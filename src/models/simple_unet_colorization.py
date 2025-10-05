"""
Simple U-Net model optimized for COLORIZATION.

Key difference from simple_unet.py:
- NO sigmoid activation (causes grayscale mode collapse)
- Uses Tanh + rescale to [0,1] for better gradient flow
- Better for GAN training
"""

import torch
import torch.nn as nn


class SimpleUNetColorization(nn.Module):
    """
    U-Net architecture optimized for colorization tasks.

    CRITICAL: No sigmoid activation! Uses Tanh instead.
    Sigmoid causes mode collapse to grayscale because sigmoid(0) = 0.5.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_frames: int = 1,
    ) -> None:
        """Initialize U-Net for colorization."""
        super().__init__()

        self.num_frames = num_frames
        self.input_shape = (num_frames, in_channels, 256, 256)

        total_in_channels = num_frames * in_channels

        # Encoder
        self.enc1 = self._conv_block(total_in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = self._conv_block(base_channels * 16, base_channels * 8)

        self.upconv3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        # Output layer - NO SIGMOID!
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W) in range [0, 1]
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output with Tanh activation, then rescale to [0, 1]
        # Tanh outputs [-1, 1], we map to [0, 1]
        # This gives better gradients than sigmoid for colorization
        out = self.out(dec1)
        out = torch.tanh(out)  # [-1, 1]
        out = (out + 1.0) / 2.0  # [0, 1]

        return out
