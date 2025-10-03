"""Loss functions for training."""

from .perceptual import PerceptualLoss
from .mixed import MixedLoss
from .colorization import ColorizationLoss
from .colorization_v2 import ColorizationLossSimple
from .colorization_v3 import ColorizationLossEnhanced
from .colorization_antiaverage import ColorizationAntiAverageLoss
from .colorization_weighted import ColorizationWeightedLoss

__all__ = [
    "PerceptualLoss",
    "MixedLoss",
    "ColorizationLoss",
    "ColorizationLossSimple",
    "ColorizationLossEnhanced",
    "ColorizationAntiAverageLoss",
    "ColorizationWeightedLoss",
]
