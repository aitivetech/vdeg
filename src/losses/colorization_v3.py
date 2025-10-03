"""Enhanced colorization loss with better color diversity."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab

from .perceptual import PerceptualLoss


class ColorizationLossEnhanced(nn.Module):
    """
    Enhanced colorization loss that prevents purple/sepia tones.

    Key improvements over simple version:
    - Weighted LAB channels (A and B get more emphasis than L)
    - Color saturation loss to encourage vibrant colors
    - Reduced perceptual weight to allow more color freedom
    """

    def __init__(
        self,
        perceptual_weight: float = 0.5,  # Lower to allow more color variation
        l_weight: float = 0.3,           # Lower weight on luminance
        ab_weight: float = 3.0,          # Much higher weight on chrominance
        saturation_weight: float = 0.5,  # Encourage colorful outputs
        perceptual_net: str = "alex",
        device: str = "cuda",
    ) -> None:
        """
        Initialize enhanced colorization loss.

        Args:
            perceptual_weight: Weight for perceptual loss (lower = more freedom)
            l_weight: Weight for L (luminance) channel
            ab_weight: Weight for AB (chrominance) channels - highest!
            saturation_weight: Weight for saturation loss
            perceptual_net: Network for perceptual loss
            device: Device to run on
        """
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.l_weight = l_weight
        self.ab_weight = ab_weight
        self.saturation_weight = saturation_weight
        self.perceptual = PerceptualLoss(net=perceptual_net, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute enhanced colorization loss.

        Args:
            pred: Predicted images of shape (B, C, H, W) in range [0, 1]
            target: Target images of shape (B, C, H, W) in range [0, 1]

        Returns:
            Dictionary with loss components
        """
        # Perceptual loss (reduced weight)
        perceptual_loss = self.perceptual(pred, target)

        # Convert to LAB
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)

        # Normalize LAB to [0, 1]
        pred_lab_norm = self._normalize_lab(pred_lab)
        target_lab_norm = self._normalize_lab(target_lab)

        # Separate L and AB channels
        pred_l = pred_lab_norm[:, 0:1, :, :]
        pred_ab = pred_lab_norm[:, 1:3, :, :]
        target_l = target_lab_norm[:, 0:1, :, :]
        target_ab = target_lab_norm[:, 1:3, :, :]

        # L (luminance) loss - lower weight
        l_loss = F.mse_loss(pred_l, target_l)

        # AB (chrominance) loss - HIGHEST weight
        ab_loss = F.mse_loss(pred_ab, target_ab)

        # Saturation loss - encourage vibrant colors
        # Compute color saturation (distance from gray in AB space)
        pred_saturation = torch.sqrt(
            (pred_lab[:, 1] - 0) ** 2 + (pred_lab[:, 2] - 0) ** 2
        ).mean()
        target_saturation = torch.sqrt(
            (target_lab[:, 1] - 0) ** 2 + (target_lab[:, 2] - 0) ** 2
        ).mean()

        # Penalize if prediction is less saturated than target
        saturation_loss = F.mse_loss(pred_saturation, target_saturation)

        # Combined loss with emphasis on chrominance
        total_loss = (
            self.perceptual_weight * perceptual_loss
            + self.l_weight * l_loss
            + self.ab_weight * ab_loss
            + self.saturation_weight * saturation_loss
        )

        return {
            "total": total_loss,
            "perceptual": perceptual_loss,
            "l_luminance": l_loss,
            "ab_chrominance": ab_loss,
            "saturation": saturation_loss,
        }

    def _normalize_lab(self, lab: torch.Tensor) -> torch.Tensor:
        """
        Normalize LAB values to [0, 1] range.

        Args:
            lab: LAB tensor with L in [0, 100], AB in [-128, 128]

        Returns:
            Normalized LAB tensor in [0, 1]
        """
        lab_norm = lab.clone()
        lab_norm[:, 0:1] = lab[:, 0:1] / 100.0
        lab_norm[:, 1:3] = (lab[:, 1:3] + 128.0) / 256.0
        return lab_norm
