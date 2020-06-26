#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
            self,
            dim_in,
            num_classes,
            pool_size,
            dropout_rate=0.0,
            feature_position='post',
            act_func="softmax",
            final_bottleneck_dim=None
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
                len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # setting final bottleneck after GAP (e.g., 2048 -> final_bottleck_dim -> num_classes)
        if final_bottleneck_dim:
            self.final_bottleneck_dim = final_bottleneck_dim
            self.final_bottleneck = nn.Conv3d(sum(dim_in), final_bottleneck_dim,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              bias=False)
            self.final_bottleneck_bn = nn.BatchNorm3d(final_bottleneck_dim,
                                                      eps=1e-5,
                                                      momentum=0.1)
            self.final_bottleneck_act = nn.ReLU(inplace=True)
            dim_in = final_bottleneck_dim
        else:
            self.final_bottleneck_dim = None
            dim_in = sum(dim_in)


        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        self.feature_position = feature_position

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
                len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []

        # Perform final bottleneck
        if self.final_bottleneck_dim:
            for pathway in range(self.num_pathways):
                inputs[pathway] = self.final_bottleneck(inputs[pathway])
                inputs[pathway] = self.final_bottleneck_bn(inputs[pathway])
                inputs[pathway] = self.final_bottleneck_act(inputs[pathway])

        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))


        h = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = h.permute((0, 2, 3, 4, 1))

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        if self.feature_position == 'final_bottleneck':
            h = x.mean([1, 2, 3])
            h = h.view(h.shape[0], -1)

        x = self.projection(x)
        if self.feature_position == 'logit':
            h = x

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x, h
