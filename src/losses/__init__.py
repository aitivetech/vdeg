"""Loss functions for multi-task restoration."""

from .core import MultiTaskLoss, PerceptualLoss
from .gan import GANLoss, FeatureMatchingLoss
from .multitask_gan import MultiTaskGANLoss

__all__ = [
    'MultiTaskLoss',
    'PerceptualLoss',
    'GANLoss',
    'FeatureMatchingLoss',
    'MultiTaskGANLoss',
]
