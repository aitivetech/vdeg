"""
Core loss functions for multi-task restoration.

Provides unified loss implementations with perceptual losses (LPIPS),
pixel-level losses, and LAB color space losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from kornia.color import rgb_to_lab
from typing import Literal


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using LPIPS (Learned Perceptual Image Patch Similarity).

    Uses pretrained networks to compare high-level feature representations
    between images, which correlates better with human perception than pixel-level losses.
    """

    def __init__(
        self,
        net: Literal["alex", "vgg", "squeeze"] = "alex",
        device: str = "cuda",
    ) -> None:
        """
        Initialize perceptual loss.

        Args:
            net: Network backbone ('alex', 'vgg', 'squeeze')
                 - alex: AlexNet (fastest, default)
                 - vgg: VGG16 (higher quality)
                 - squeeze: SqueezeNet (smallest)
            device: Device to run on
        """
        super().__init__()
        self.model = lpips.LPIPS(net=net).to(device).eval()

        # Freeze perceptual model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted images (B, C, H, W) in range [0, 1]
            target: Target images (B, C, H, W) in range [0, 1]

        Returns:
            Perceptual loss (scalar)
        """
        # LPIPS expects inputs in range [-1, 1]
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0

        # Compute perceptual distance
        loss = self.model(pred_scaled, target_scaled)
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Unified multi-task loss for all restoration tasks.

    Supports:
    - Super-resolution: RGB reconstruction + perceptual
    - Artifact removal: RGB reconstruction + perceptual
    - Colorization: LAB AB channels + perceptual

    All tasks share a single perceptual loss network for efficiency.
    """

    def __init__(
        self,
        # Task weights
        rgb_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        ab_weight: float = 2.0,
        # Perceptual settings
        perceptual_net: Literal["alex", "vgg", "squeeze"] = "alex",
        device: str = "cuda",
        # Task enables
        enable_colorization: bool = True,
        enable_super_resolution: bool = True,
    ) -> None:
        """
        Initialize multi-task loss.

        Args:
            rgb_weight: Weight for RGB MSE loss (SR + artifact removal)
            perceptual_weight: Weight for perceptual loss (all tasks)
            ab_weight: Weight for AB chrominance loss (colorization)
            perceptual_net: Network for perceptual loss
            device: Device to run on
            enable_colorization: Enable colorization loss component
            enable_super_resolution: Enable SR loss component
        """
        super().__init__()
        self.rgb_weight = rgb_weight
        self.perceptual_weight = perceptual_weight
        self.ab_weight = ab_weight
        self.enable_colorization = enable_colorization
        self.enable_super_resolution = enable_super_resolution

        # Shared perceptual loss
        self.perceptual = PerceptualLoss(net=perceptual_net, device=device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        task_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            pred: Predicted images (B, C, H, W) in range [0, 1]
            target: Target images (B, C, H, W) in range [0, 1]
            task_weights: Optional per-batch task weights override

        Returns:
            Dictionary with loss components:
            - total: Combined weighted loss
            - rgb: RGB MSE loss (if enabled)
            - perceptual: Perceptual loss
            - ab_chrominance: AB channel loss (if enabled)
        """
        # Use provided weights or defaults
        if task_weights is None:
            task_weights = {
                'rgb': self.rgb_weight,
                'perceptual': self.perceptual_weight,
                'ab': self.ab_weight,
            }

        losses = {}
        total_loss = torch.tensor(0.0, device=pred.device)

        # 1. RGB reconstruction (for SR + artifact removal)
        if self.enable_super_resolution:
            rgb_loss = F.mse_loss(pred, target)
            losses['rgb'] = rgb_loss
            total_loss = total_loss + task_weights.get('rgb', self.rgb_weight) * rgb_loss

        # 2. Perceptual loss (shared across all tasks)
        perceptual_loss = self.perceptual(pred, target)
        losses['perceptual'] = perceptual_loss
        total_loss = total_loss + task_weights.get('perceptual', self.perceptual_weight) * perceptual_loss

        # 3. LAB AB channel loss (for colorization)
        if self.enable_colorization:
            # Convert to LAB color space
            pred_lab = rgb_to_lab(pred)
            target_lab = rgb_to_lab(target)

            # Extract AB channels (chrominance)
            pred_ab = pred_lab[:, 1:3, :, :]
            target_ab = target_lab[:, 1:3, :, :]

            # Normalize AB to [0, 1] (AB typically in [-128, 128])
            pred_ab_norm = pred_ab / 128.0
            target_ab_norm = target_ab / 128.0

            ab_loss = F.mse_loss(pred_ab_norm, target_ab_norm)
            losses['ab_chrominance'] = ab_loss
            total_loss = total_loss + task_weights.get('ab', self.ab_weight) * ab_loss

        losses['total'] = total_loss
        return losses

    def set_task_weights(
        self,
        rgb_weight: float | None = None,
        perceptual_weight: float | None = None,
        ab_weight: float | None = None,
    ) -> None:
        """
        Dynamically adjust task weights during training.

        Useful for curriculum learning or task balancing.

        Args:
            rgb_weight: New RGB weight (None to keep current)
            perceptual_weight: New perceptual weight (None to keep current)
            ab_weight: New AB weight (None to keep current)
        """
        if rgb_weight is not None:
            self.rgb_weight = rgb_weight
        if perceptual_weight is not None:
            self.perceptual_weight = perceptual_weight
        if ab_weight is not None:
            self.ab_weight = ab_weight
