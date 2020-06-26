"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Distance-based objective functions.
Re-implemented for the compatibility with other losses
"""
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """ A simple mean squared error (MSE) implementation.
    """
    def __init__(self, reduction='mean', **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, **kwargs):
        return F.mse_loss(input, target, reduction=self.reduction)


class L1Loss(nn.Module):
    """ A simple mean absolute error (MAE) implementation.
    """
    def __init__(self, reduction='mean', **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, **kwargs):
        return F.l1_loss(input, target, reduction=self.reduction)
