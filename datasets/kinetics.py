"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Dataset for the action recognition benchmarks.
We use the official implemenation of SlowFast by Facebook research.
https://github.com/facebookresearch/SlowFast
"""
import torch

from datasets.kinetics_tools.loader import construct_loader


def get_kinetics_dataloader(root,
                            split='train',
                            logger=None,
                            anno_file=None,
                            dataset_name='kinetics50',
                            batch_size=16):
    return construct_loader(root, split, logger,
                            anno_file=anno_file,
                            dataset_name=dataset_name,
                            num_gpus=torch.cuda.device_count(),
                            batch_size=batch_size)
