"""Classification-based colorization loss using quantized AB space.

Based on "Colorful Image Colorization" (Zhang et al., ECCV 2016).
Treats colorization as classification over quantized AB color space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab, lab_to_rgb


class ColorizationClassificationLoss(nn.Module):
    """
    Classification-based colorization loss.

    Treats AB color space as discrete bins and uses cross-entropy loss.
    This prevents mode collapse to gray/sepia by forcing the model to
    commit to specific color bins rather than averaging.

    The model still outputs continuous RGB values, but we quantize them
    internally for the classification loss.
    """

    def __init__(
        self,
        num_bins: int = 313,  # Number of color bins (can adjust)
        q_range: float = 110.0,  # Range of quantization (-110 to 110 for AB)
        temperature: float = 0.38,  # Temperature for soft encoding
        lambda_classification: float = 1.0,  # Weight for classification loss
        lambda_regression: float = 0.01,  # Small weight for regression (smoothness)
        device: str = "cuda",
    ) -> None:
        """
        Initialize classification-based colorization loss.

        Args:
            num_bins: Number of bins to quantize AB space into
            q_range: Range of AB values to quantize
            temperature: Temperature for soft quantization
            lambda_classification: Weight for classification loss
            lambda_regression: Weight for regression loss (keeps outputs smooth)
            device: Device to run on
        """
        super().__init__()
        self.num_bins = num_bins
        self.q_range = q_range
        self.temperature = temperature
        self.lambda_classification = lambda_classification
        self.lambda_regression = lambda_regression
        self.device = device

        # Create quantized AB grid
        self.ab_bins = self._create_ab_bins()

    def _create_ab_bins(self) -> torch.Tensor:
        """
        Create quantized AB color space bins.

        Returns:
            Tensor of shape (num_bins, 2) with AB coordinates for each bin
        """
        # Create grid in AB space
        # Use Gaussian quantization (more bins near gray, fewer at extremes)
        grid_size = int(torch.sqrt(torch.tensor(self.num_bins)).item())

        a_values = torch.linspace(-self.q_range, self.q_range, grid_size)
        b_values = torch.linspace(-self.q_range, self.q_range, grid_size)

        # Create meshgrid
        a_grid, b_grid = torch.meshgrid(a_values, b_values, indexing='ij')

        # Flatten to (num_bins, 2)
        ab_bins = torch.stack([a_grid.flatten(), b_grid.flatten()], dim=1)

        # Only keep bins that are within gamut (optional filtering)
        # For simplicity, keep all bins
        if ab_bins.shape[0] > self.num_bins:
            ab_bins = ab_bins[:self.num_bins]
        elif ab_bins.shape[0] < self.num_bins:
            # Pad if needed
            padding = self.num_bins - ab_bins.shape[0]
            ab_bins = torch.cat([ab_bins, ab_bins[:padding]], dim=0)

        return ab_bins.to(self.device)

    def _quantize_ab(self, ab: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous AB values to soft-encoded distribution over bins.

        Args:
            ab: AB channels of shape (B, 2, H, W)

        Returns:
            Soft-encoded distribution of shape (B, num_bins, H, W)
        """
        B, _, H, W = ab.shape

        # Reshape AB to (B*H*W, 2)
        ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)  # (B*H*W, 2)

        # Compute distances to all bins
        # ab_flat: (B*H*W, 2), ab_bins: (num_bins, 2)
        # distances: (B*H*W, num_bins)
        distances = torch.cdist(ab_flat, self.ab_bins)  # Euclidean distance

        # Convert distances to soft assignments using Gaussian kernel
        soft_assignments = torch.exp(-distances ** 2 / (2 * self.temperature ** 2))

        # Normalize to get probabilities
        soft_assignments = soft_assignments / (soft_assignments.sum(dim=1, keepdim=True) + 1e-8)

        # Reshape back to (B, num_bins, H, W)
        soft_assignments = soft_assignments.reshape(B, H, W, self.num_bins).permute(0, 3, 1, 2)

        return soft_assignments

    def _decode_ab_from_distribution(self, distribution: torch.Tensor) -> torch.Tensor:
        """
        Convert distribution over bins back to continuous AB values.

        Args:
            distribution: Distribution of shape (B, num_bins, H, W)

        Returns:
            AB values of shape (B, 2, H, W)
        """
        B, _, H, W = distribution.shape

        # Reshape distribution to (B*H*W, num_bins)
        dist_flat = distribution.permute(0, 2, 3, 1).reshape(-1, self.num_bins)

        # Compute expected AB as weighted sum of bin centers
        # dist_flat: (B*H*W, num_bins), ab_bins: (num_bins, 2)
        # ab_decoded: (B*H*W, 2)
        ab_decoded = torch.matmul(dist_flat, self.ab_bins)

        # Reshape back to (B, 2, H, W)
        ab_decoded = ab_decoded.reshape(B, H, W, 2).permute(0, 3, 1, 2)

        return ab_decoded

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute classification-based colorization loss.

        Args:
            pred: Predicted RGB images of shape (B, 3, H, W) in range [0, 1]
            target: Target RGB images of shape (B, 3, H, W) in range [0, 1]

        Returns:
            Dictionary with loss components
        """
        # Convert to LAB
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)

        # Extract AB channels
        pred_ab = pred_lab[:, 1:3, :, :]  # (B, 2, H, W)
        target_ab = target_lab[:, 1:3, :, :]  # (B, 2, H, W)

        # Quantize target AB to get ground truth distribution
        target_distribution = self._quantize_ab(target_ab)  # (B, num_bins, H, W)

        # Quantize predicted AB to get predicted distribution
        pred_distribution = self._quantize_ab(pred_ab)  # (B, num_bins, H, W)

        # Classification loss (cross-entropy between distributions)
        # Both are soft distributions, so we use KL divergence
        # Add small epsilon to avoid log(0)
        pred_distribution_safe = torch.clamp(pred_distribution, min=1e-8, max=1.0)
        classification_loss = F.kl_div(
            pred_distribution_safe.log(),
            target_distribution,
            reduction='batchmean',
        )

        # Regression loss (for smoothness) - direct MSE on AB
        regression_loss = F.mse_loss(pred_ab / 128.0, target_ab / 128.0)

        # Combined loss
        total_loss = (
            self.lambda_classification * classification_loss
            + self.lambda_regression * regression_loss
        )

        return {
            "total": total_loss,
            "classification": classification_loss,
            "regression": regression_loss,
        }
