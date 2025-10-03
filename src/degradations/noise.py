"""Noise degradations."""

import torch


class GaussianNoise:
    """Add Gaussian noise to images."""

    def __init__(self, sigma: float = 0.1) -> None:
        """
        Initialize Gaussian noise degradation.

        Args:
            sigma: Standard deviation of noise (in [0, 1] range)
        """
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise.

        Args:
            x: Input tensor of shape (T, C, H, W) in range [0, 1]

        Returns:
            Noisy tensor clipped to [0, 1]
        """
        noise = torch.randn_like(x) * self.sigma
        return torch.clamp(x + noise, 0.0, 1.0)


class PoissonNoise:
    """Add Poisson (shot) noise to images."""

    def __init__(self, scale: float = 10.0) -> None:
        """
        Initialize Poisson noise degradation.

        Args:
            scale: Scale factor for Poisson distribution (higher = less noise)
        """
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Poisson noise.

        Args:
            x: Input tensor of shape (T, C, H, W) in range [0, 1]

        Returns:
            Noisy tensor clipped to [0, 1]
        """
        # Scale up, apply Poisson noise, scale back
        scaled = x * self.scale
        noisy = torch.poisson(scaled) / self.scale
        return torch.clamp(noisy, 0.0, 1.0)
