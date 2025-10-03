"""Color-related degradations."""

import torch
from kornia.color import rgb_to_grayscale


class Grayscale:
    """Convert to grayscale for colorization tasks."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB to grayscale.

        Args:
            x: Input tensor of shape (T, C, H, W) with C=3

        Returns:
            Grayscale tensor of shape (T, 3, H, W) (replicated to 3 channels)
        """
        T_dim, C, H, W = x.shape

        # Convert to grayscale
        gray = rgb_to_grayscale(x.reshape(T_dim, C, H, W))  # (T, 1, H, W)

        # Replicate to 3 channels for consistency
        return gray.repeat(1, 3, 1, 1)


class ReduceDynamicRange:
    """Reduce dynamic range for SDR to HDR tasks."""

    def __init__(self, gamma: float = 2.2, max_value: float = 0.8) -> None:
        """
        Initialize dynamic range reduction.

        Args:
            gamma: Gamma correction factor
            max_value: Maximum value to clip to (simulates SDR limit)
        """
        self.gamma = gamma
        self.max_value = max_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce dynamic range.

        Args:
            x: Input tensor of shape (T, C, H, W) in range [0, 1]

        Returns:
            Tensor with reduced dynamic range
        """
        # Apply gamma correction and clipping
        x_sdr = torch.pow(x, 1.0 / self.gamma)
        x_sdr = torch.clamp(x_sdr, 0.0, self.max_value)

        # Normalize back to [0, 1]
        x_sdr = x_sdr / self.max_value

        return x_sdr
