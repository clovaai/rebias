#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.kinetics_tools.kinetics import Kinetics

# Supported datasets.
_DATASET_CATALOG = {"kinetics50": Kinetics, "mimetics50": Kinetics,
                    "kinetics10": Kinetics, "mimetics10": Kinetics,}


def construct_loader(root, split, logger, anno_file,
                     dataset_name='kinetics', batch_size=64,
                     num_gpus=1, num_workers=24, pin_memory=True):
    """
    :param root: root path
    :param split: dataset split ('train','val','test')
    :param logger:
    :param dataset_name:
    :param batch_size:
    :param num_gpus:
    :param num_workers:
    :param pin_memory:
    :return:
    """

    assert split in ["train", "val", "test"]
    if split in ["train"]:
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        shuffle = False
        drop_last = False
    assert (
        dataset_name in _DATASET_CATALOG.keys()
    ), "Dataset '{}' is not supported".format(dataset_name)

    # Construct the dataset
    dataset = _DATASET_CATALOG[dataset_name](root, split, logger,
                                             dataset_name=dataset_name,
                                             anno_file=anno_file)

    # Create a sampler for multi-process training
    # sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    sampler = None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
