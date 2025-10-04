"""Super-resolution degradation that actually reduces resolution."""

import torch
import torch.nn.functional as F


class SuperResolutionDegradation:
    """Downscale images for super-resolution training (no upscaling back)."""

    def __init__(self, scale_factor: int = 2, mode: str = "bilinear") -> None:
        """
        Initialize super-resolution degradation.

        Args:
            scale_factor: Factor to downscale by (e.g., 4 = quarter resolution)
            mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
        """
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1, got {scale_factor}")
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply downscaling only (for super-resolution).

        Args:
            x: Input tensor of shape (T, C, H, W)

        Returns:
            Low-resolution tensor of shape (T, C, H/scale, W/scale)
        """
        if self.scale_factor == 1:
            return x

        T_dim, C, H, W = x.shape

        # Downscale only (don't upscale back)
        x_down = F.interpolate(
            x.reshape(T_dim * C, 1, H, W),
            scale_factor=1.0 / self.scale_factor,
            mode=self.mode,
            align_corners=False if self.mode != "nearest" else None,
        )

        _, _, H_new, W_new = x_down.shape
        return x_down.reshape(T_dim, C, H_new, W_new)
