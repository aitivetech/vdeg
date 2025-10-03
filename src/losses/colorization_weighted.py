"""Weighted MSE loss for colorization - simple but effective.

Based on the insight that rare colors should be weighted higher than common ones.
This prevents the model from always predicting common colors (which causes sepia/gray).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab


class ColorizationWeightedLoss(nn.Module):
    """
    Simple weighted MSE loss for colorization.

    Key insight: Weight pixels by how "rare" their color is.
    Common colors (gray/brown/sepia) get LOW weight.
    Rare colors (saturated blues/greens/reds) get HIGH weight.

    This naturally encourages the model to predict diverse, saturated colors
    instead of playing it safe with common neutral colors.
    """

    def __init__(
        self,
        device: str = "cuda",
    ) -> None:
        """
        Initialize weighted colorization loss.

        Args:
            device: Device to run on
        """
        super().__init__()
        self.device = device

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute weighted MSE loss.

        Args:
            pred: Predicted RGB of shape (B, 3, H, W) in [0, 1]
            target: Target RGB of shape (B, 3, H, W) in [0, 1]

        Returns:
            Loss dictionary
        """
        # Convert to LAB
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)

        # Extract AB channels
        target_ab = target_lab[:, 1:3, :, :]  # (B, 2, H, W)

        # Compute per-pixel weights based on color saturation
        # More saturated colors = higher weight
        # This makes the model care MORE about getting vibrant colors right
        saturation = torch.sqrt(target_ab[:, 0] ** 2 + target_ab[:, 1] ** 2)  # (B, H, W)

        # Normalize saturation to [0, 1]
        saturation_norm = saturation / 181.0

        # Create weights: saturated colors get higher weight
        # Add 0.5 as base so even gray pixels have some weight
        weights = 0.5 + saturation_norm  # Range: [0.5, 1.5]

        # Expand weights for all channels
        weights = weights.unsqueeze(1)  # (B, 1, H, W)

        # Weighted MSE loss in RGB space
        mse_per_pixel = (pred - target) ** 2  # (B, 3, H, W)
        weighted_mse = mse_per_pixel * weights
        rgb_loss = weighted_mse.mean()

        # Also compute loss in LAB space with same weights
        # Normalize LAB
        pred_lab_norm = self._normalize_lab(pred_lab)
        target_lab_norm = self._normalize_lab(target_lab)

        lab_mse_per_pixel = (pred_lab_norm - target_lab_norm) ** 2  # (B, 3, H, W)
        weighted_lab_mse = lab_mse_per_pixel * weights
        lab_loss = weighted_lab_mse.mean()

        # Simple combination
        total_loss = rgb_loss + lab_loss

        return {
            "total": total_loss,
            "rgb": rgb_loss,
            "lab": lab_loss,
        }

    def _normalize_lab(self, lab: torch.Tensor) -> torch.Tensor:
        """Normalize LAB to [0, 1]."""
        lab_norm = lab.clone()
        lab_norm[:, 0:1] = lab[:, 0:1] / 100.0
        lab_norm[:, 1:3] = (lab[:, 1:3] + 128.0) / 256.0
        return lab_norm
