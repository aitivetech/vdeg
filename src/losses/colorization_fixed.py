"""
FIXED Loss function for colorization tasks.

Critical fix: Diversity loss was backwards - it was PENALIZING color!
This version removes the buggy diversity loss entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab

from .perceptual import PerceptualLoss


class ColorizationLossFixed(nn.Module):
    """
    FIXED colorization loss WITHOUT buggy diversity component.

    The diversity loss in the original was backwards - it penalized color!
    This simpler version works much better.
    """

    def __init__(
        self,
        rgb_weight: float = 0.5,
        perceptual_weight: float = 1.0,
        ab_weight: float = 2.0,
        perceptual_net: str = "vgg",
        device: str = "cuda",
    ) -> None:
        """
        Initialize colorization loss.

        Args:
            rgb_weight: Weight for RGB MSE loss
            perceptual_weight: Weight for perceptual loss
            ab_weight: Weight for AB channels in LAB space
            perceptual_net: Network for perceptual loss
            device: Device to run on
        """
        super().__init__()
        self.rgb_weight = rgb_weight
        self.perceptual_weight = perceptual_weight
        self.ab_weight = ab_weight
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

        # Focus on AB channels (chrominance)
        pred_ab = pred_lab[:, 1:3, :, :]
        target_ab = target_lab[:, 1:3, :, :]

        # Normalize AB to similar scale
        pred_ab_norm = pred_ab / 128.0
        target_ab_norm = target_ab / 128.0

        ab_loss = F.mse_loss(pred_ab_norm, target_ab_norm)

        # Combined loss (NO DIVERSITY - it was buggy!)
        total_loss = (
            self.rgb_weight * rgb_loss
            + self.perceptual_weight * perceptual_loss
            + self.ab_weight * ab_loss
        )

        return {
            "total": total_loss,
            "rgb": rgb_loss,
            "perceptual": perceptual_loss,
            "ab_chrominance": ab_loss,
        }
