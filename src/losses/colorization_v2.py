"""Simplified but effective loss function for colorization tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab

from .perceptual import PerceptualLoss


class ColorizationLossSimple(nn.Module):
    """
    Simplified colorization loss that actually works.

    Key principles:
    - Strong perceptual loss for semantic guidance
    - LAB space loss for color accuracy
    - No fancy diversity tricks that can destabilize training
    """

    def __init__(
        self,
        perceptual_weight: float = 1.0,
        lab_weight: float = 1.0,
        ab_weight: float = 2.0,
        perceptual_net: str = "alex",
        device: str = "cuda",
    ) -> None:
        """
        Initialize colorization loss.

        Args:
            perceptual_weight: Weight for perceptual loss
            lab_weight: Weight for full LAB loss
            ab_weight: Additional weight specifically for AB channels
            perceptual_net: Network for perceptual loss ('alex' or 'vgg')
            device: Device to run on
        """
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.lab_weight = lab_weight
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
        # Perceptual loss in RGB space
        perceptual_loss = self.perceptual(pred, target)

        # Convert to LAB
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)

        # Normalize LAB to similar scale as RGB [0,1]
        # L: [0, 100] -> [0, 1]
        # A, B: [-128, 128] -> [-1, 1] -> [0, 1]
        pred_lab_norm = pred_lab.clone()
        pred_lab_norm[:, 0:1] = pred_lab[:, 0:1] / 100.0
        pred_lab_norm[:, 1:3] = (pred_lab[:, 1:3] + 128.0) / 256.0

        target_lab_norm = target_lab.clone()
        target_lab_norm[:, 0:1] = target_lab[:, 0:1] / 100.0
        target_lab_norm[:, 1:3] = (target_lab[:, 1:3] + 128.0) / 256.0

        # Full LAB loss (normalized)
        lab_loss = F.mse_loss(pred_lab_norm, target_lab_norm)

        # Extra emphasis on AB (chrominance) channels (normalized)
        pred_ab_norm = pred_lab_norm[:, 1:3, :, :]
        target_ab_norm = target_lab_norm[:, 1:3, :, :]
        ab_loss = F.mse_loss(pred_ab_norm, target_ab_norm)

        # Combined loss
        total_loss = (
            self.perceptual_weight * perceptual_loss
            + self.lab_weight * lab_loss
            + self.ab_weight * ab_loss
        )

        return {
            "total": total_loss,
            "perceptual": perceptual_loss,
            "lab": lab_loss,
            "ab": ab_loss,
        }
