"""Loss function for colorization tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab

from .perceptual import PerceptualLoss


class ColorizationLoss(nn.Module):
    """
    Improved loss for colorization that separates luminance and chrominance.

    Key improvements:
    - Focuses on AB channels (chrominance) in LAB space
    - Uses Huber loss to handle outliers better
    - Adds color diversity regularization to prevent mode collapse
    - Higher weight on perceptual loss for semantic understanding
    """

    def __init__(
        self,
        rgb_weight: float = 0.5,
        perceptual_weight: float = 1.0,
        ab_weight: float = 2.0,
        huber_delta: float = 0.1,
        diversity_weight: float = 0.1,
        perceptual_net: str = "vgg",
        device: str = "cuda",
    ) -> None:
        """
        Initialize colorization loss.

        Args:
            rgb_weight: Weight for RGB MSE loss (lower than others)
            perceptual_weight: Weight for perceptual loss (higher for semantics)
            ab_weight: Weight for AB channels in LAB space (highest for color)
            huber_delta: Delta parameter for Huber loss (smoother than MSE)
            diversity_weight: Weight for color diversity regularization
            perceptual_net: Network for perceptual loss ('vgg' recommended for colorization)
            device: Device to run on
        """
        super().__init__()
        self.rgb_weight = rgb_weight
        self.perceptual_weight = perceptual_weight
        self.ab_weight = ab_weight
        self.huber_delta = huber_delta
        self.diversity_weight = diversity_weight
        self.perceptual = PerceptualLoss(net=perceptual_net, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute colorization loss.

        Args:
            pred: Predicted images of shape (B, C, H, W) in range [0, 1]
            target: Target images of shape (B, C, H, W) in range [0, 1]

        Returns:
            Dictionary with loss components
        """
        # RGB loss (basic reconstruction)
        rgb_loss = F.mse_loss(pred, target)

        # Perceptual loss (semantic understanding)
        perceptual_loss = self.perceptual(pred, target)

        # Convert to LAB color space
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)

        # Separate L (luminance) and AB (chrominance) channels
        # L is in range [0, 100], AB are in range [-128, 128]
        pred_l = pred_lab[:, 0:1, :, :]
        pred_ab = pred_lab[:, 1:3, :, :]
        target_l = target_lab[:, 0:1, :, :]
        target_ab = target_lab[:, 1:3, :, :]

        # Focus on AB channels (chrominance) with Huber loss for robustness
        # Normalize AB to similar scale as L
        pred_ab_norm = pred_ab / 128.0
        target_ab_norm = target_ab / 128.0

        ab_loss = F.smooth_l1_loss(pred_ab_norm, target_ab_norm, beta=self.huber_delta)

        # Color diversity regularization to prevent mode collapse
        # Encourages the model to use a wider range of colors
        diversity_loss = self._color_diversity_loss(pred_ab)

        # Combined loss
        total_loss = (
            self.rgb_weight * rgb_loss
            + self.perceptual_weight * perceptual_loss
            + self.ab_weight * ab_loss
            + self.diversity_weight * diversity_loss
        )

        return {
            "total": total_loss,
            "rgb": rgb_loss,
            "perceptual": perceptual_loss,
            "ab_chrominance": ab_loss,
            "diversity": diversity_loss,
        }

    def _color_diversity_loss(self, ab: torch.Tensor) -> torch.Tensor:
        """
        Compute color diversity loss to prevent mode collapse.

        Penalizes low variance in color predictions to encourage
        the model to use a fuller range of colors.

        Args:
            ab: AB channels of shape (B, 2, H, W)

        Returns:
            Diversity loss (lower variance = higher loss)
        """
        # Compute variance across spatial dimensions
        # Flatten spatial dimensions
        B, C, H, W = ab.shape
        ab_flat = ab.reshape(B, C, H * W)

        # Compute variance for each channel in each image
        variance = torch.var(ab_flat, dim=2).mean()

        # Clamp variance to prevent extreme values
        variance = torch.clamp(variance, min=1.0, max=10000.0)

        # Return negative log variance (encourages higher variance)
        # Clamping prevents this from exploding during training
        return -torch.log(variance)
