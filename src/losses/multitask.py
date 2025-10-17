"""
Multi-task loss for simultaneous super-resolution, artifact removal, and colorization.

This loss combines three tasks:
1. Super-resolution + artifact removal (MSE + perceptual on RGB)
2. Colorization (LAB color space + perceptual on AB channels)
3. Combined perceptual loss for semantic coherence

The loss automatically handles task weighting and normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab

from .perceptual import PerceptualLoss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for super-resolution + artifact removal + colorization.

    This loss combines:
    - RGB reconstruction (MSE) for overall quality
    - Perceptual loss for semantic understanding
    - LAB AB channel loss for vibrant colorization

    All tasks share perceptual features for efficiency.
    """

    def __init__(
        self,
        rgb_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        ab_weight: float = 2.0,
        perceptual_net: str = "vgg",
        device: str = "cuda",
        enable_colorization: bool = True,
        enable_super_resolution: bool = True,
    ) -> None:
        """
        Initialize multi-task loss.

        Args:
            rgb_weight: Weight for RGB MSE loss (artifact removal + SR)
            perceptual_weight: Weight for perceptual loss (semantic coherence)
            ab_weight: Weight for AB chrominance loss (colorization)
            perceptual_net: Network for perceptual loss ('alex', 'vgg', 'squeeze')
            device: Device to run on
            enable_colorization: Enable colorization loss component
            enable_super_resolution: Enable super-resolution loss component
        """
        super().__init__()
        self.rgb_weight = rgb_weight
        self.perceptual_weight = perceptual_weight
        self.ab_weight = ab_weight
        self.enable_colorization = enable_colorization
        self.enable_super_resolution = enable_super_resolution

        # Shared perceptual loss network
        self.perceptual = PerceptualLoss(net=perceptual_net, device=device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        task_weights: dict[str, float] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            pred: Predicted images of shape (B, C, H, W) in range [0, 1]
            target: Target images of shape (B, C, H, W) in range [0, 1]
            task_weights: Optional per-batch task weights
                         {'rgb': float, 'perceptual': float, 'ab': float}

        Returns:
            Dictionary with loss components:
            - total: Combined weighted loss
            - rgb: RGB MSE loss
            - perceptual: Perceptual loss
            - ab_chrominance: AB channel loss (if colorization enabled)
        """
        # Default task weights
        if task_weights is None:
            task_weights = {
                'rgb': self.rgb_weight,
                'perceptual': self.perceptual_weight,
                'ab': self.ab_weight
            }

        losses = {}
        total_loss = 0.0

        # 1. RGB reconstruction loss (for SR + artifact removal)
        if self.enable_super_resolution:
            rgb_loss = F.mse_loss(pred, target)
            losses['rgb'] = rgb_loss
            total_loss += task_weights.get('rgb', self.rgb_weight) * rgb_loss

        # 2. Perceptual loss (shared across all tasks for efficiency)
        perceptual_loss = self.perceptual(pred, target)
        losses['perceptual'] = perceptual_loss
        total_loss += task_weights.get('perceptual', self.perceptual_weight) * perceptual_loss

        # 3. LAB AB channel loss (for colorization)
        if self.enable_colorization:
            # Convert to LAB color space
            pred_lab = rgb_to_lab(pred)
            target_lab = rgb_to_lab(target)

            # Extract AB channels (chrominance)
            pred_ab = pred_lab[:, 1:3, :, :]
            target_ab = target_lab[:, 1:3, :, :]

            # Normalize AB to [0, 1] range (AB typically in [-128, 128])
            pred_ab_norm = pred_ab / 128.0
            target_ab_norm = target_ab / 128.0

            ab_loss = F.mse_loss(pred_ab_norm, target_ab_norm)
            losses['ab_chrominance'] = ab_loss
            total_loss += task_weights.get('ab', self.ab_weight) * ab_loss

        losses['total'] = total_loss
        return losses

    def set_task_weights(
        self,
        rgb_weight: float | None = None,
        perceptual_weight: float | None = None,
        ab_weight: float | None = None
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


class AdaptiveMultiTaskLoss(MultiTaskLoss):
    """
    Adaptive multi-task loss with automatic task balancing.

    Uses gradient magnitude balancing to ensure all tasks contribute equally.
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Learnable task weights (log variance)
        # Initialize to log(weight) for stability
        self.log_var_rgb = nn.Parameter(torch.tensor([0.0]))
        self.log_var_perceptual = nn.Parameter(torch.tensor([0.0]))
        self.log_var_ab = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        task_weights: dict[str, float] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Compute adaptive multi-task loss with learned weighting.

        The learned weights automatically balance task contributions.
        """
        # Compute individual losses
        losses = {}

        # RGB loss
        if self.enable_super_resolution:
            rgb_loss = F.mse_loss(pred, target)
            losses['rgb'] = rgb_loss
            # Adaptive weighting: loss / (2 * var) + log(var)
            precision_rgb = torch.exp(-self.log_var_rgb)
            weighted_rgb = precision_rgb * rgb_loss + self.log_var_rgb
        else:
            weighted_rgb = 0.0

        # Perceptual loss
        perceptual_loss = self.perceptual(pred, target)
        losses['perceptual'] = perceptual_loss
        precision_perceptual = torch.exp(-self.log_var_perceptual)
        weighted_perceptual = precision_perceptual * perceptual_loss + self.log_var_perceptual

        # AB loss
        if self.enable_colorization:
            pred_lab = rgb_to_lab(pred)
            target_lab = rgb_to_lab(target)
            pred_ab = pred_lab[:, 1:3, :, :] / 128.0
            target_ab = target_lab[:, 1:3, :, :] / 128.0
            ab_loss = F.mse_loss(pred_ab, target_ab)
            losses['ab_chrominance'] = ab_loss
            precision_ab = torch.exp(-self.log_var_ab)
            weighted_ab = precision_ab * ab_loss + self.log_var_ab
        else:
            weighted_ab = 0.0

        # Total adaptive loss
        total_loss = weighted_rgb + weighted_perceptual + weighted_ab
        losses['total'] = total_loss

        # Log learned weights for monitoring
        losses['weight_rgb'] = torch.exp(-self.log_var_rgb)
        losses['weight_perceptual'] = torch.exp(-self.log_var_perceptual)
        losses['weight_ab'] = torch.exp(-self.log_var_ab)

        return losses
