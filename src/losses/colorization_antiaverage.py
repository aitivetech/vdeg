"""Anti-averaging colorization loss.

Prevents mode collapse to sepia/gray tones without complex classification.
Uses a combination of losses that discourage averaging colors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab


class ColorizationAntiAverageLoss(nn.Module):
    """
    Anti-averaging colorization loss.

    Key insight: Sepia/gray happens when model averages multiple valid colors.
    Solution: Penalize predictions that look like averages (low saturation, neutral hue).

    Uses:
    1. Standard LAB loss (with proper normalization)
    2. Saturation push (heavily penalize desaturated predictions)
    3. Hue diversity (encourage varied hues, not just one dominant hue)
    4. Color confidence (penalize wishy-washy predictions)
    """

    def __init__(
        self,
        lab_weight: float = 1.0,
        saturation_weight: float = 2.0,  # HIGH - push for saturated colors
        diversity_weight: float = 1.0,   # NEW - encourage diverse hues
        confidence_weight: float = 1.0,  # NEW - penalize averaging
        device: str = "cuda",
    ) -> None:
        """
        Initialize anti-averaging loss.

        Args:
            lab_weight: Weight for standard LAB MSE
            saturation_weight: Weight for saturation push (high = more vibrant)
            diversity_weight: Weight for hue diversity
            confidence_weight: Weight for color confidence
            device: Device
        """
        super().__init__()
        self.lab_weight = lab_weight
        self.saturation_weight = saturation_weight
        self.diversity_weight = diversity_weight
        self.confidence_weight = confidence_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute anti-averaging loss.

        Args:
            pred: Predicted RGB of shape (B, 3, H, W) in [0, 1]
            target: Target RGB of shape (B, 3, H, W) in [0, 1]

        Returns:
            Loss dictionary
        """
        # Convert to LAB
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)

        # Normalize
        pred_lab_norm = self._normalize_lab(pred_lab)
        target_lab_norm = self._normalize_lab(target_lab)

        # 1. Standard LAB loss
        lab_loss = F.mse_loss(pred_lab_norm, target_lab_norm)

        # 2. Saturation loss - STRONG penalty for desaturation
        pred_saturation = torch.sqrt(pred_lab[:, 1] ** 2 + pred_lab[:, 2] ** 2)
        target_saturation = torch.sqrt(target_lab[:, 1] ** 2 + target_lab[:, 2] ** 2)

        # Normalize saturation to [0, 1] range (max possible ~181 for sqrt(128^2 + 128^2))
        pred_saturation_norm = pred_saturation / 181.0
        target_saturation_norm = target_saturation / 181.0

        # MSE on normalized saturation
        saturation_loss = F.mse_loss(pred_saturation_norm, target_saturation_norm)

        # 3. Hue diversity loss - penalize if all pixels have similar hue
        # Compute hue angle for each pixel
        pred_hue = torch.atan2(pred_lab[:, 2], pred_lab[:, 1])  # (B, H, W) in range [-pi, pi]
        target_hue = torch.atan2(target_lab[:, 2], target_lab[:, 1])

        # Normalize hue to [0, 1] (from [-pi, pi])
        pred_hue_norm = (pred_hue + 3.14159) / (2 * 3.14159)
        target_hue_norm = (target_hue + 3.14159) / (2 * 3.14159)

        # Compute hue variance within each image
        pred_hue_var = torch.var(pred_hue_norm.reshape(pred_hue_norm.size(0), -1), dim=1).mean()
        target_hue_var = torch.var(target_hue_norm.reshape(target_hue_norm.size(0), -1), dim=1).mean()

        # Penalize if predicted hue variance is much lower than target
        # (indicates mode collapse to single color)
        diversity_loss = F.mse_loss(pred_hue_var.unsqueeze(0), target_hue_var.unsqueeze(0))

        # 4. Color confidence loss - penalize "neutral" predictions
        # Measure distance from gray (0, 0) in AB space
        pred_ab = pred_lab[:, 1:3, :, :]
        target_ab = target_lab[:, 1:3, :, :]

        # L2 distance from gray (normalized to [0, 1])
        pred_distance_from_gray = torch.sqrt(pred_ab[:, 0] ** 2 + pred_ab[:, 1] ** 2) / 181.0
        target_distance_from_gray = torch.sqrt(target_ab[:, 0] ** 2 + target_ab[:, 1] ** 2) / 181.0

        # Heavily penalize if prediction is closer to gray than target
        confidence_loss = F.relu(target_distance_from_gray - pred_distance_from_gray).mean()

        # Combined loss
        total_loss = (
            self.lab_weight * lab_loss
            + self.saturation_weight * saturation_loss
            + self.diversity_weight * diversity_loss
            + self.confidence_weight * confidence_loss
        )

        return {
            "total": total_loss,
            "lab": lab_loss,
            "saturation": saturation_loss,
            "diversity": diversity_loss,
            "confidence": confidence_loss,
        }

    def _normalize_lab(self, lab: torch.Tensor) -> torch.Tensor:
        """Normalize LAB to [0, 1]."""
        lab_norm = lab.clone()
        lab_norm[:, 0:1] = lab[:, 0:1] / 100.0
        lab_norm[:, 1:3] = (lab[:, 1:3] + 128.0) / 256.0
        return lab_norm
