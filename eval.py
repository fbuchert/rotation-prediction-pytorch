import argparse
import logging
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.metrics import *
from utils.train import rotate_tensors, ModelWrapper, NINWrapper
from utils.eval import AverageMeterSet
from datasets.custom_datasets import CustomSubset
from models.network_in_network import NetworkInNetwork


logger = logging.getLogger()


def evaluate(
        args,
        eval_loader: DataLoader,
        model: nn.Module,
        epoch: int,
        descriptor: str = "Test",
):
    """
    Evaluates current model based on the provided evaluation dataloader

    Parameters
    ----------
    args: argparse.Namespace
        Namespace with command line arguments and corresponding values
    eval_loader: torch.utils.data.DataLoader
        DataLoader objects which loads batches of evaluation dataset
    model: nn.Module
        Current model which should be evaluated on prediction task
    epoch: int
        Current epoch which is used for progress bar logging if enabled
    descriptor: str
        Descriptor which is used for progress bar logging if enabled

    Returns
    -------
    eval_tuple: namedtuple
        NamedTuple which holds all evaluation metrics such as accuracy, precision, recall, f1
    """
    meters = AverageMeterSet()

    model.eval()

    if args.pbar:
        p_bar = tqdm(range(len(eval_loader)))

    with torch.no_grad():
        for i, (inputs, _) in enumerate(eval_loader):
            inputs, rot_targets = rotate_tensors(inputs)

            inputs = inputs.to(args.device)
            rot_targets = rot_targets.to(args.device)

            # Output
            logits = model(inputs)
            loss = F.cross_entropy(logits, rot_targets, reduction="mean")

            # Compute metrics
            (top1,) = accuracy(logits, rot_targets, topk=(1,))
            meters.update("loss", loss.item(), len(inputs))
            meters.update("top1", top1.item(), len(inputs))

            if args.pbar:
                p_bar.set_description(
                    "{descriptor}: Epoch: {epoch:4}. Iter: {batch:4}/{iter:4}. Class loss: {cl:4}. Top1: {top1:4}.".format(
                        descriptor=descriptor,
                        epoch=epoch + 1,
                        batch=i + 1,
                        iter=len(eval_loader),
                        cl=meters["loss"],
                        top1=meters["top1"],
                    )
                )
                p_bar.update()
    if args.pbar:
        p_bar.close()
    logger.info(" * Prec@1 {top1.avg:.3f}".format(top1=meters["top1"]))
    return meters["loss"].avg, meters["top1"].avg


def extract_layers(
        args,
        dataset: CustomSubset,
        model: nn.Module,
        embedding_layer: str = "conv2",
        prediction_layer: str = "classifier",
        descriptor: str = "Embedding extraction"
):
    dataset.return_index = True
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    logger.info("Extracting {} and {} layers of pretrained RotNet model.".format(embedding_layer, prediction_layer))
    if isinstance(model, NetworkInNetwork):
        wrapped_model = NINWrapper(model, to_extract=[embedding_layer, prediction_layer])
    else:
        wrapped_model = ModelWrapper(model, (embedding_layer, prediction_layer))

    wrapped_model.eval()

    if args.pbar:
        p_bar = tqdm(range(len(loader)))

    index_list = []
    embedding_list = []
    prediction_list = []
    with torch.no_grad():
        for i, (samples, _, indices) in enumerate(loader):
            samples = samples.to(args.device)

            # Output
            embeddings, logits = wrapped_model(samples)

            index_list.append(indices)
            embedding_list.append(embeddings)
            prediction_list.append(torch.softmax(logits, dim=1))

            if args.pbar:
                p_bar.set_description(
                    "{descriptor}: Iter: {batch:4}/{iter:4}".format(
                        descriptor=descriptor, batch=i + 1, iter=len(loader)
                    )
                )
                p_bar.update()
    if args.pbar:
        p_bar.close()

    pretraining_save_dict = {
        "indices": torch.cat(index_list),
        "embeddings": torch.cat(embedding_list),
        "predictions": torch.cat(prediction_list),
    }
    return pretraining_save_dict


def parse_args():
    parser = argparse.ArgumentParser(description='FixMatch evaluation')

    parser.add_argument('--run-path', type=str, help='path to FixMatch run which should be evaluated.')
    parser.add_argument('--data-dir', default='./data', type=str, help='path to directory where datasets are saved.')
    parser.add_argument('--checkpoint-file', default='', type=str, help='name of .tar-checkpoint file from which model is loaded for evaluation.')
    parser.add_argument('--device', default='cpu', type=str, choices=['cpu', 'gpu'], help='device (cpu / cuda) on which evaluation is run.')
    parser.add_argument('--pbar', action='store_true', default=False, help='flag indicating whether or not to show progress bar for evaluation.')
    return parser.parse_args()


if __name__ == '__main__':
    import os
    from utils.misc import load_dataset_indices, load_args, load_state
    from augmentation.augmentations import get_normalizer
    from datasets.datasets import get_datasets, get_base_sets
    from models.model_factory import MODEL_GETTERS

    args = parse_args()
    args.device = torch.device(args.device)

    # Load arguments of run to evaluate
    run_args = load_args(args.run_path)

    # Initialize test dataset and loader
    _, test_set = get_base_sets(run_args.dataset, args.data_dir, test_transform=get_normalizer(run_args.dataset))
    test_loader = DataLoader(
        test_set,
        batch_size=run_args.batch_size,
        num_workers=run_args.num_workers,
        shuffle=False,
        pin_memory=run_args.pin_memory,
    )

    # Load trained model from specified checkpoint .tar-file containing model state dict
    model = MODEL_GETTERS[run_args.model](num_classes=run_args.num_classes)

    if args.checkpoint_file:
        saved_state = load_state(os.path.join(args.run_path, args.checkpoint_file), map_location=args.device)
    else:
        checkpoint_file = next(filter(lambda x: x.endswith('.tar'), sorted(os.listdir(args.run_path), reverse=True)))
        saved_state = load_state(os.path.join(args.run_path, checkpoint_file), map_location=args.device)

    model.load_state_dict(saved_state['model_state_dict'])
    loss, top1_acc = evaluate(run_args, test_loader, model, saved_state['epoch'])

    print(' MixMatch EVALUATION '.center(50, '-'))
    print(f'\t - Dataset {run_args.dataset}')
    print(f'\t - Model {run_args.model}')
    print(f'\t - Test metrics:')
    print(f'\t\tloss: {loss}')
    print(f'\t\ttop1_accuracy: {top1_acc}')