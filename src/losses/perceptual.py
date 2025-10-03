"""Perceptual loss using LPIPS."""

import torch
import torch.nn as nn
from lpips import LPIPS


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using LPIPS (Learned Perceptual Image Patch Similarity).

    Measures perceptual similarity between images using a pretrained network.
    """

    def __init__(self, net: str = "alex", device: str = "cuda") -> None:
        """
        Initialize perceptual loss.

        Args:
            net: Network to use ('alex', 'vgg', 'squeeze')
            device: Device to run on
        """
        super().__init__()
        self.lpips = LPIPS(net=net).to(device)
        self.lpips.eval()

        # Freeze LPIPS network
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted images of shape (B, C, H, W) in range [0, 1]
            target: Target images of shape (B, C, H, W) in range [0, 1]

        Returns:
            Scalar loss value
        """
        # LPIPS expects images in range [-1, 1]
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0

        return self.lpips(pred_scaled, target_scaled).mean()
