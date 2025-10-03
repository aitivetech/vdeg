"""Mixed loss combining multiple objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perceptual import PerceptualLoss


class MixedLoss(nn.Module):
    """
    Mixed loss combining MSE and perceptual loss.

    Useful for general restoration tasks.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        perceptual_net: str = "alex",
        device: str = "cuda",
    ) -> None:
        """
        Initialize mixed loss.

        Args:
            mse_weight: Weight for MSE loss
            perceptual_weight: Weight for perceptual loss
            perceptual_net: Network to use for perceptual loss
            device: Device to run on
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual = PerceptualLoss(net=perceptual_net, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute mixed loss.

        Args:
            pred: Predicted images of shape (B, C, H, W)
            target: Target images of shape (B, C, H, W)

        Returns:
            Dictionary with 'total', 'mse', and 'perceptual' losses
        """
        # MSE loss
        mse_loss = F.mse_loss(pred, target)

        # Perceptual loss
        perceptual_loss = self.perceptual(pred, target)

        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss + self.perceptual_weight * perceptual_loss
        )

        return {
            "total": total_loss,
            "mse": mse_loss,
            "perceptual": perceptual_loss,
        }
