"""
GAN loss functions for adversarial training.

Supports multiple GAN loss types and includes helper functions for
stable GAN training, particularly for colorization tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class GANLoss(nn.Module):
    """
    GAN loss with support for different loss types.

    Supports:
    - vanilla: Original GAN loss with BCE
    - lsgan: Least Squares GAN (more stable)
    - hinge: Hinge loss (used in many modern GANs)
    """

    def __init__(
        self,
        loss_type: Literal["vanilla", "lsgan", "hinge"] = "lsgan",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ):
        """
        Initialize GAN loss.

        Args:
            loss_type: Type of GAN loss to use
            real_label: Label value for real images
            fake_label: Label value for fake images
        """
        super().__init__()
        self.loss_type = loss_type
        self.real_label = real_label
        self.fake_label = fake_label

        if loss_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_type == "lsgan":
            self.loss = nn.MSELoss()
        elif loss_type == "hinge":
            self.loss = None  # Hinge loss is computed directly
        else:
            raise ValueError(f"Unknown GAN loss type: {loss_type}")

    def get_target_tensor(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Create target tensor with same shape as prediction.

        Args:
            prediction: Discriminator output
            is_real: Whether target should be real or fake

        Returns:
            Target tensor filled with real or fake labels
        """
        target_value = self.real_label if is_real else self.fake_label
        return torch.full_like(prediction, target_value, requires_grad=False)

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Compute GAN loss.

        Args:
            prediction: Discriminator output (logits, not probabilities)
            is_real: Whether the target should be real or fake

        Returns:
            GAN loss value
        """
        if self.loss_type == "hinge":
            if is_real:
                # For real images: max(0, 1 - D(x))
                loss = F.relu(1.0 - prediction).mean()
            else:
                # For fake images: max(0, 1 + D(G(z)))
                loss = F.relu(1.0 + prediction).mean()
            return loss
        else:
            target = self.get_target_tensor(prediction, is_real)
            return self.loss(prediction, target)

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator loss (real + fake).

        Args:
            real_pred: Discriminator output for real images
            fake_pred: Discriminator output for fake images

        Returns:
            Total discriminator loss
        """
        real_loss = self.forward(real_pred, is_real=True)
        fake_loss = self.forward(fake_pred, is_real=False)
        return (real_loss + fake_loss) * 0.5

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss.

        Generator tries to fool discriminator by making fake_pred look real.

        Args:
            fake_pred: Discriminator output for generated images

        Returns:
            Generator adversarial loss
        """
        return self.forward(fake_pred, is_real=True)


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for GAN training stability.

    Matches intermediate feature representations from the discriminator
    between real and fake images. This can stabilize training and improve
    quality.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        real_features: list[torch.Tensor],
        fake_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature matching loss.

        Args:
            real_features: List of feature tensors from discriminator on real images
            fake_features: List of feature tensors from discriminator on fake images

        Returns:
            Feature matching loss (L1 distance between features)
        """
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        return loss / len(real_features)


class ColorizationGANLoss(nn.Module):
    """
    Combined loss for colorization with GAN.

    Combines:
    - Content loss (perceptual + pixel-level)
    - Adversarial loss (GAN)
    - Optional feature matching loss
    """

    def __init__(
        self,
        content_loss: nn.Module,
        gan_loss: GANLoss,
        content_weight: float = 1.0,
        gan_weight: float = 0.1,
        feature_matching_weight: float = 0.0,
    ):
        """
        Initialize combined colorization GAN loss.

        Args:
            content_loss: Content loss module (e.g., perceptual + LAB loss)
            gan_loss: GAN loss module
            content_weight: Weight for content loss
            gan_weight: Weight for adversarial loss (usually much smaller)
            feature_matching_weight: Weight for feature matching loss (optional)
        """
        super().__init__()
        self.content_loss = content_loss
        self.gan_loss = gan_loss
        self.content_weight = content_weight
        self.gan_weight = gan_weight
        self.feature_matching_weight = feature_matching_weight

        if feature_matching_weight > 0:
            self.feature_matching = FeatureMatchingLoss()

    def generator_loss_combined(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        discriminator_pred: torch.Tensor,
        real_features: list[torch.Tensor] = None,
        fake_features: list[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined generator loss.

        Args:
            pred: Generated colorized images
            target: Ground truth color images
            discriminator_pred: Discriminator output for generated images
            real_features: Discriminator features for real images (for feature matching)
            fake_features: Discriminator features for fake images (for feature matching)

        Returns:
            Dictionary with loss components
        """
        # Content loss (perceptual + pixel-level)
        content_loss_dict = self.content_loss(pred, target)
        content_loss_value = content_loss_dict["total"]

        # Adversarial loss
        adv_loss = self.gan_loss.generator_loss(discriminator_pred)

        # Combined loss
        total_loss = (
            self.content_weight * content_loss_value +
            self.gan_weight * adv_loss
        )

        # Feature matching loss (optional)
        if self.feature_matching_weight > 0 and real_features is not None:
            fm_loss = self.feature_matching(real_features, fake_features)
            total_loss += self.feature_matching_weight * fm_loss
        else:
            fm_loss = torch.tensor(0.0, device=pred.device)

        # Build result dictionary
        result = {
            "total": total_loss,
            "content": content_loss_value,
            "adversarial": adv_loss,
            "feature_matching": fm_loss,
        }

        # Add detailed content loss components
        for key, value in content_loss_dict.items():
            if key != "total":
                result[f"content_{key}"] = value

        return result
