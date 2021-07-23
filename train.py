import os
import logging
from tqdm import tqdm

import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms


from datasets.config import IMG_SIZE
from eval import evaluate
from utils.train import get_wd_param_list, rotate_tensors
from utils.misc import *
from utils.eval import AverageMeterSet
from augmentation.augmentations import get_normalizer, get_weak_augmentation

MIN_VALIDATION_SIZE = 50

logger = logging.getLogger()


def get_transform_dict(args):
    """
    Generates dictionary with transforms for all datasets

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object that contains all command line arguments with their corresponding values
    Returns
    -------
    transform_dict: Dict
        Dictionary containing transforms for the labeled train set, unlabeled train set
        and the validation / test set
    """
    img_size = IMG_SIZE[args.dataset]
    padding = int(0.125 * img_size)
    return {
        "train": transforms.Compose(
            [
                get_weak_augmentation(img_size, padding, padding_mode="constant"),
                get_normalizer(args.dataset),
            ]
        ),
        "train_unlabeled": None,
        "test": get_normalizer(args.dataset),
    }


def get_optimizer(args, model):
    """
    Initialize and return SGD optimizer.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    model: torch.nn.Module
        torch module, i.e. neural net, which is trained using rotation prediction
    Returns
    -------
    optim: torch.optim.Optimizer
        Returns SGD optimizer which is used for model training
    """
    return SGD(
        get_wd_param_list(model),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.wd,
    )


def get_scheduler(args, optimizer):
    return MultiStepLR(
        optimizer,
        milestones=[0.3 * args.epochs, 0.6 * args.epochs, 0.8 * args.epochs],
        gamma=0.2,
    )


def train(args, model, train_loader, validation_loader, test_loader, writer, save_path):
    """
    Method for RotNet training of model based on given data loaders and parameters.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    model: torch.nn.Module
        The torch model to train
    train_loader: DataLoader
        Data loader of labeled dataset
    validation_loader: DataLoader
        Data loader of validation set.
    test_loader: DataLoader
        Data loader of test set
    writer: SummaryWriter
        SummaryWriter instance which is used to write losses as well as training / evaluation metrics
        to tensorboard summary file.
    save_path: str
        Path to which training data is saved.
    Returns
    -------
    model: torch.nn.Module
        The method returns the trained model
    ema_model: EMA
        The EMA class which maintains an exponential moving average of model parameters.
    writer: SummaryWriter
        SummaryWriter instance which is used to write losses as well as training / evaluation metrics
        to tensorboard summary file.
    """
    model.to(args.device)

    if args.use_ema:
        ema_model = EMA(model, args.ema_decay)
    else:
        ema_model = None

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    best_acc = 0
    start_epoch = 0

    if args.resume:
        state_dict = load_state(args.resume)
        model.load_state_dict(state_dict["model_state_dict"])
        if args.use_ema:
            ema_model.shadow = state_dict["ema_model_shadow"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        best_acc = state_dict["acc"]
        start_epoch = state_dict["epoch"]

    for epoch in range(start_epoch, args.epochs):
        train_total_loss = train_epoch(
            args, model, ema_model, train_loader, optimizer, scheduler, epoch
        )

        if args.use_ema:
            ema_model.assign(model)
            val_loss, val_acc_1 = evaluate(
                args, validation_loader, model, epoch, "Validation"
            )
            test_loss, test_acc_1 = evaluate(
                args, test_loader, model, epoch, "Test"
            )
            ema_model.resume(model)
        else:
            val_loss, val_acc_1 = evaluate(
                args, validation_loader, model, epoch, "Validation"
            )
            test_loss, test_acc_1 = evaluate(
                args, test_loader, model, epoch, "Test"
            )

        if scheduler:
            scheduler.step()

        writer.add_scalar("Loss/train_total", train_total_loss, epoch)
        writer.add_scalar("Loss/val_total", val_loss, epoch)
        writer.add_scalar("Accuracy/val_acc_1", val_acc_1, epoch)
        writer.add_scalar("Loss/test_total", test_loss, epoch)
        writer.add_scalar("Accuracy/test_acc_1", test_acc_1, epoch)
        writer.flush()

        if epoch % args.checkpoint_interval == 0 and args.save:
            save_state(
                epoch,
                model,
                val_acc_1,
                optimizer,
                scheduler,
                ema_model,
                save_path,
                filename=f"checkpoint_{epoch}.tar",
            )

    save_state(
        epoch,
        model,
        val_acc_1,
        optimizer,
        scheduler,
        ema_model,
        save_path,
        filename="last_model.tar",
    )
    return model, ema_model, writer


def train_epoch(args, model, ema_model, train_loader, optimizer, scheduler, epoch):
    """
    Method that executes a training epoch, i.e. a pass through all train samples in the training data loaders.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace with command line arguments and corresponding values
    model: torch.nn.Module
        Model, i.e. neural network to train using rotation prediction.
    ema_model: EMA
        The EMA class which maintains an exponential moving average of model parameters.
    train_loader: DataLoader
        Data loader fetching batches from the labeled set of data.
    optimizer: Optimizer
        Optimizer used for model training.
    scheduler: torch.optim.lr_scheduler.LambdaLR
        LambdaLR-Scheduler, which controls the learning rate using a cosine learning rate decay.
    epoch: int
        Current epoch
    Returns
    -------
    train_stats: Tuple
        The method returns a tuple containing the total, labeled and unlabeled loss.
    """
    meters = AverageMeterSet()

    iterations_per_epoch = len(train_loader)

    model.zero_grad()
    model.train()

    if args.pbar:
        p_bar = tqdm(range(iterations_per_epoch))

    for batch_idx, (samples, _) in enumerate(train_loader):
        samples, rot_targets = rotate_tensors(samples)
        loss = train_step(args, model, (samples, rot_targets), meters)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA model if configured
        if args.use_ema:
            ema_model(model)

        if args.pbar:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=iterations_per_epoch,
                    lr=scheduler.get_last_lr()[0] if scheduler else args.lr,
                )
            )
            p_bar.update()
    if args.pbar:
        p_bar.close()
    return meters["loss"].avg


def train_step(args, model, batch, meters):
    """
    Method that executes a rotation prediction training step, i.e. a single training iteration.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace with command line arguments and corresponding values
    model: torch.nn.Module
        Model, i.e. neural network to train using rotation prediction.
    batch: Tuple
        Tuple containing samples and labels of current batch
    meters: AverageMeterSet
        AverageMeterSet object which is used to track training and testing metrics (loss, accuracy, ...)
        over the entire training process.
    Returns
    -------
    loss: torch.Tensor
        Tensor containing the total rotation prediction loss used for optimization by backpropagation.
    """
    inputs, targets = batch

    inputs = inputs.to(args.device)
    targets = targets.to(args.device)

    logits = model(inputs)
    loss = F.cross_entropy(logits, targets, reduction="mean")

    meters.update("loss", loss.item(), inputs.size(0))
    return loss
