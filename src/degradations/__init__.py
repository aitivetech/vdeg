"""Degradation functions for creating training data."""

from .base import DegradationPipeline
from .noise import GaussianNoise, PoissonNoise
from .compression import JPEGCompression
from .blur import GaussianBlur, MotionBlur
from .downscale import Downscale
from .color import Grayscale, ReduceDynamicRange
from .super_resolution import SuperResolutionDegradation
from .multitask import MultiTaskDegradation, TaskSpecificDegradation

__all__ = [
    "DegradationPipeline",
    "GaussianNoise",
    "PoissonNoise",
    "JPEGCompression",
    "GaussianBlur",
    "MotionBlur",
    "Downscale",
    "Grayscale",
    "ReduceDynamicRange",
    "SuperResolutionDegradation",
    "MultiTaskDegradation",
    "TaskSpecificDegradation",
]
