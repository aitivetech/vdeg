"""Base classes for degradations."""

from typing import Callable

import torch


class DegradationPipeline:
    """
    Composable pipeline of degradation functions.

    Applies multiple degradations in sequence.
    """

    def __init__(self, *degradations: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Initialize pipeline.

        Args:
            degradations: Sequence of degradation functions
        """
        self.degradations = degradations

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all degradations in sequence.

        Args:
            x: Input tensor of shape (T, C, H, W)

        Returns:
            Degraded tensor of same shape
        """
        for degradation in self.degradations:
            x = degradation(x)
        return x
