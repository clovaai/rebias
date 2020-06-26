"""Criterions for de-biased representations.

This module contains three different types of criterions.
- HSIC: independence-based criterion used by ReBias (ours).
- Distance: L2 and L1 losses.
- Comparison methods: RUBi and LearnedMixin for comparisons.
"""
from criterions.hsic import RbfHSIC, MinusRbfHSIC
from criterions.dist import L1Loss, MSELoss
from criterions.comparison_methods import RUBi, LearnedMixin

__all__ = ['RbfHSIC', 'MinusRbfHSIC',
           'L1Loss', 'MSELoss',
           'RUBi', 'LearnedMixin']


def get_criterion(criterion_name):
    """return the criterion (nn.Module) by the given name (str)
    Possible criterions:
        ['RbfHSIC', 'MinusRbfHSIC', 'L1Loss', 'MSELoss', 'RUBi', 'LearnedMixin']
    """
    return globals()[criterion_name]
