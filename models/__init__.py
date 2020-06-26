"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Target architectures and intentionally biased architectures for three benchmarks
- MNIST: deep stacked convolutional networks with different kernel size, i.e., 7 (target) and 1 (biased).
- ImageNet: ResNet-18 (target) and BagNet-18 (biased).
- Kinetics: spatial-temporal 3D-ResNet (target), spatial-only 2D-ResNet (biased).
"""
try:
    from models.action_models import ResNet3D
except ImportError:
    print('failed to import kinetics, please install library from')
    print('https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md')
    ResNet3D = None
from models.imagenet_models import resnet18, bagnet18
from models.mnist_models import SimpleConvNet
from models.rebias_models import ReBiasModels


__all__ = ['ReBiasModels',
           'resnet18', 'bagnet18',
           'SimpleConvNet',
           'ResNet3D']
