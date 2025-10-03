"""Blur degradations."""

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d, motion_blur


class GaussianBlur:
    """Apply Gaussian blur."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0) -> None:
        """
        Initialize Gaussian blur degradation.

        Args:
            kernel_size: Size of Gaussian kernel (must be odd)
            sigma: Standard deviation of Gaussian kernel
        """
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur.

        Args:
            x: Input tensor of shape (T, C, H, W)

        Returns:
            Blurred tensor
        """
        T_dim, C, H, W = x.shape

        # Reshape to (T*C, 1, H, W) for processing
        x_reshaped = x.reshape(T_dim * C, 1, H, W)

        # Apply Gaussian blur
        blurred = gaussian_blur2d(
            x_reshaped,
            kernel_size=(self.kernel_size, self.kernel_size),
            sigma=(self.sigma, self.sigma),
        )

        # Reshape back
        return blurred.reshape(T_dim, C, H, W)


class MotionBlur:
    """Apply motion blur."""

    def __init__(self, kernel_size: int = 9, angle: float = 0.0, direction: float = 0.0) -> None:
        """
        Initialize motion blur degradation.

        Args:
            kernel_size: Size of motion blur kernel (must be odd)
            angle: Angle of motion in degrees
            direction: Direction of motion (-1.0 to 1.0)
        """
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply motion blur.

        Args:
            x: Input tensor of shape (T, C, H, W)

        Returns:
            Blurred tensor
        """
        T_dim, C, H, W = x.shape

        # Reshape to (T*C, 1, H, W) for processing
        x_reshaped = x.reshape(T_dim * C, 1, H, W)

        # Apply motion blur
        blurred = motion_blur(
            x_reshaped,
            kernel_size=self.kernel_size,
            angle=self.angle,
            direction=self.direction,
        )

        # Reshape back
        return blurred.reshape(T_dim, C, H, W)
