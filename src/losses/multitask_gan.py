"""
Multi-task GAN loss for adversarial training across SR, artifact removal, and colorization.

Combines content losses (RGB, perceptual, AB) with adversarial losses.
"""

import torch
import torch.nn as nn

from .multitask import MultiTaskLoss, AdaptiveMultiTaskLoss
from .gan import GANLoss


class MultiTaskGANLoss(nn.Module):
    """
    Multi-task GAN loss combining content and adversarial objectives.

    This loss enables adversarial training for multi-task restoration:
    - Content loss (from MultiTaskLoss): RGB + perceptual + AB
    - Adversarial loss: helps with realism for SR and colorization
    """

    def __init__(
        self,
        content_loss: MultiTaskLoss | AdaptiveMultiTaskLoss,
        gan_loss: GANLoss,
        content_weight: float = 1.0,
        gan_weight: float = 0.1,
    ):
        """
        Initialize multi-task GAN loss.

        Args:
            content_loss: Multi-task content loss (RGB + perceptual + AB)
            gan_loss: GAN loss for adversarial training
            content_weight: Weight for content loss
            gan_weight: Weight for adversarial loss
        """
        super().__init__()
        self.content_loss = content_loss
        self.gan_loss = gan_loss
        self.content_weight = content_weight
        self.gan_weight = gan_weight

    def generator_loss_combined(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fake_pred: torch.Tensor,
        task_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined generator loss (content + adversarial).

        Args:
            pred: Generated images (B, C, H, W)
            target: Target images (B, C, H, W)
            fake_pred: Discriminator predictions for generated images
            task_weights: Optional per-batch task weights

        Returns:
            Dictionary with loss components:
            - total: Combined loss
            - content: Content loss from MultiTaskLoss
            - adversarial: Adversarial loss
            - rgb, perceptual, ab_chrominance: Individual content losses
        """
        # Content loss (RGB + perceptual + AB)
        content_result = self.content_loss(pred, target, task_weights)

        # Adversarial loss (fool discriminator)
        adv_loss = self.gan_loss.generator_loss(fake_pred)

        # Combined loss
        total_loss = (
            self.content_weight * content_result["total"]
            + self.gan_weight * adv_loss
        )

        # Return all components for logging
        result = {
            "total": total_loss,
            "content": content_result["total"],
            "adversarial": adv_loss,
        }

        # Add individual content losses
        for key, value in content_result.items():
            if key != "total":
                result[key] = value

        return result

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator loss.

        Args:
            real_pred: Discriminator predictions for real images
            fake_pred: Discriminator predictions for fake images

        Returns:
            Discriminator loss
        """
        return self.gan_loss.discriminator_loss(real_pred, fake_pred)

    def set_weights(
        self,
        content_weight: float | None = None,
        gan_weight: float | None = None,
    ) -> None:
        """
        Dynamically adjust loss weights.

        Args:
            content_weight: New content weight (None to keep current)
            gan_weight: New GAN weight (None to keep current)
        """
        if content_weight is not None:
            self.content_weight = content_weight
        if gan_weight is not None:
            self.gan_weight = gan_weight
