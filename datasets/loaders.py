from typing import Union, Tuple, NamedTuple, List

import numpy as np
from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
    RandomSampler,
    WeightedRandomSampler,
)


def get_sampler(
    dataset: Dataset,
    num_samples: int,
    reweighted: bool = False,
    median_freq: bool = False,
    replacement: bool = True,
):
    """
    Method that generates torch samplers.

    Parameters
    ----------
    dataset: Dataset
        Torch base dataset object from which samples are selected.
    replacement: bool
        Boolean flag indicating whether samples should be drawn with replacement or not.
    num_samples: int
        Number of samples that are drawn. Should only be specified when sampling with replacement, i.e. replacement=True
    reweighted: bool
        See get_reweighted_sampler
    median_freq: bool
        See get_reweighted_sampler

    Returns
    ----------
    sampler: Sampler (either RandomSampler or WeightedRandomSampler)
            Returns torch sampler instance, which can be used as input to a torch DataLoader. If reweighted=True,
            a WeightedRandomSampler instance is returned, while if reweighted=False a RandomSampler instance is
            returned.
    """
    if reweighted:
        return get_reweighted_sampler(dataset.targets, num_samples, replacement, median_freq)
    else:
        return get_uniform_sampler(dataset, replacement, num_samples)


def get_uniform_sampler(dataset: Dataset, replacement: bool = False, num_samples: int = None):
    """
    Method that generates samplers that randomly select samples from the dataset with equal probability.

    Parameters
    ----------
    dataset: Dataset
        Torch base dataset object from which samples are selected.
    replacement: bool
        Boolean flag indicating whether samples should be drawn with replacement or not.
    num_samples: int
        Number of samples that are drawn. Should only be specified when sampling with replacement, i.e. replacement=True

    Returns
    ----------
    random_sampler: RandomSampler
        Returns random sampler instance, which can be used as input to a torch DataLoader.
    """
    if replacement:
        return RandomSampler(dataset, replacement=replacement, num_samples=num_samples)
    else:
        return RandomSampler(dataset, replacement=replacement)


def get_reweighted_sampler(targets: List, num_samples: int, replacement: bool = True, median_freq: bool = False):
    """
    Method that generates a weighted random sampler that selects samples with a probability inversely proportional
    to their frequency (or median frequency) in the dataset. This sampling strategy can be useful when working with
    highly imbalanced datasets.

    Parameters
    ----------
    targets: List
        List of sample targets / classes based on which sampling weights are computed.
    num_samples: int
        Number of samples that are drawn. Should only be specified when sampling with replacement, i.e. replacement=True
    replacement: bool
        Boolean flag indicating whether samples should be drawn with replacement or not.
    median_freq: bool
        Boolean flag indicating whether sample weights are computed to be inversely proportional to frequency or median
        frequency of sample class.
    Returns
    ----------
    weighted_samples: WeightedRandomSampler
        Returns the weighted sampler instance, which can be used as input to a torch DataLoader.
    """
    labels, counts = np.unique(targets, return_counts=True)
    if not median_freq:
        class_weights = 1 / (counts / np.sum(counts))
    else:
        class_weights = 1 / (counts / np.median(counts))
    sample_weights = np.zeros(len(targets))
    for class_label in labels:
        sample_weights[np.array(targets) == class_label] = class_weights[
            class_label
        ]
    num_samples = num_samples if replacement else len(targets)
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=replacement)


def create_loaders(
    args,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    total_iters: int = None,
):
    """
    Parameters
    ----------
    args: argparse.Namespace
        Namespace object that contains all command line arguments with their corresponding values.
    train_dataset: Dataset
        Training dataset for RotNet training.
    validation_dataset: Dataset
        Validation dataset.
    test_dataset: Dataset
        Test dataset.
    batch_size: int
        Batch size specifying how many samples per batch are loaded. The given batch size is used for all dataloaders,
        except the unlabeled train loader - semi-supervised learning algorithms commonly use a multiple of the batch
        size used for labeled train loader for the unlabeled train loader.
    num_workers: int (default: 0)
        Number of subprocesses used for data loading. If num_workers=0, data loading is executed in the main process.
    mu: int (default: 1)
        Multiplication factor which is used to compute the batch_size for the unlabeled train dataset.
    total_iters: int (default: None)
        total_iters specifies the total number of desired training iterations per epoch. If not None,
        the product of total_iters and batch_size is used to compute the total number of samples used for training
        at every epoch.
    Returns
    ----------
    data_loaders: Tuple[Tuple[DataLoader, DataLoader], DataLoader, DataLoader]
        Returns a tuple of all data loaders required for semi-supervised learning.
    """
    num_labeled_samples = (
        len(train_dataset) if total_iters is None else total_iters * batch_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=get_sampler(
            train_dataset,
            num_samples=num_labeled_samples,
            reweighted=args.weighted_sampling,
        ),
        num_workers=num_workers,
        drop_last=False,
        pin_memory=args.pin_memory
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
    )
    return (
        train_loader,
        validation_loader,
        test_loader,
    )
