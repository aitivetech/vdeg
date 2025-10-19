"""Super-resolution degradation with bicubic upscaling."""

import torch
import torch.nn.functional as F


class SuperResolutionDegradation:
    """
    Downscale then upscale images for super-resolution training.

    Creates a low-quality upscaled input that the model must refine.
    Input and output are the same size (matching target).
    """

    def __init__(self, scale_factor: int = 2, mode: str = "bicubic") -> None:
        """
        Initialize super-resolution degradation.

        Args:
            scale_factor: Factor to downscale and upscale by (e.g., 4 = downscale to 1/4 then upscale back)
            mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
        """
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1, got {scale_factor}")
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply downscaling then upscaling (creates low-quality version at same size).

        Args:
            x: Input tensor of shape (T, C, H, W)

        Returns:
            Low-quality upscaled tensor of shape (T, C, H, W) - same size as input
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

        # Upscale back to original size (creates low-quality version)
        x_up = F.interpolate(
            x_down,
            size=(H, W),
            mode=self.mode,
            align_corners=False if self.mode != "nearest" else None,
        )

        return x_up.reshape(T_dim, C, H, W)
