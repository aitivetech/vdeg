"""Downscaling degradation for super-resolution tasks."""

import torch
import torch.nn.functional as F


class Downscale:
    """Downscale and upscale images to simulate low-resolution input."""

    def __init__(self, scale_factor: int = 2, mode: str = "bilinear") -> None:
        """
        Initialize downscale degradation.

        Args:
            scale_factor: Factor to downscale by (e.g., 2 = half resolution)
            mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
        """
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1, got {scale_factor}")
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply downscaling and upscaling.

        Args:
            x: Input tensor of shape (T, C, H, W)

        Returns:
            Low-resolution tensor upscaled to original size
        """
        if self.scale_factor == 1:
            return x

        T_dim, C, H, W = x.shape

        # Downscale
        x_down = F.interpolate(
            x.reshape(T_dim * C, 1, H, W),
            scale_factor=1.0 / self.scale_factor,
            mode=self.mode,
            align_corners=False if self.mode != "nearest" else None,
        )

        # Upscale back to original size
        x_up = F.interpolate(
            x_down,
            size=(H, W),
            mode=self.mode,
            align_corners=False if self.mode != "nearest" else None,
        )

        return x_up.reshape(T_dim, C, H, W)
