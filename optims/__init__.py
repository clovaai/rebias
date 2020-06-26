"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Opitmizers for the training.
"""
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from adamp import AdamP


__optim__ = ['Adam', 'AdamP']
__scheduler__ = ['StepLR', 'CosineAnnealingLR']

__all__ = ['Adam', 'AdamP', 'StepLR', 'CosineAnnealingLR', 'get_optim', 'get_scheduler']


def get_optim(params, optim_name, optim_config=None):
    if optim_name not in __optim__:
        raise KeyError(optim_name)

    optim = globals()[optim_name]
    if not optim_config:
        optim_config = {'lr': 1e-2, 'weight_decay': 1e-4}
    return optim(params, **optim_config)


def get_scheduler(optimizer, scheduler_name, scheduler_config=None):
    if scheduler_name not in __scheduler__:
        raise KeyError(scheduler_name)

    scheduler = globals()[scheduler_name]

    if not scheduler_config:
        scheduler_config = {}
    return scheduler(optimizer, **scheduler_config)
