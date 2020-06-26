"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Datasets used for the ``unbaised'' benchmarks
- Biased-MNIST: synthetic bias with background colours.
- 9-Class ImageNet: realistic bias where the unbiased performances are
    computed by the proxy texture labels (by texture clustering).
- Kinetics-10: a subset of Kinetics dataset, where the unbiased performances are
    measured by ``Mimetics'' dataset.
    Weinzaepfel, Philippe, and Grégory Rogez. "Mimetics: Towards Understanding Human Actions Out of Context." arXiv preprint arXiv:1912.07249 (2019).
    https://europe.naverlabs.com/research/computer-vision/mimetics/
"""
from datasets.colour_mnist import get_biased_mnist_dataloader
try:
    from datasets.kinetics import get_kinetics_dataloader
except ImportError:
    print('failed to import kinetics, please install library from')
    print('https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md')
from datasets.imagenet import get_imagenet_dataloader


__all__ = ['get_biased_mnist_dataloader',
           'get_kinetics_dataloader',
           'get_imagenet_dataloader']
