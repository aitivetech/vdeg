"""
Multi-task loss for simultaneous super-resolution, artifact removal, and colorization.

Re-exports core loss implementations for backward compatibility.
"""

from .core import MultiTaskLoss, PerceptualLoss

__all__ = ['MultiTaskLoss', 'PerceptualLoss']
