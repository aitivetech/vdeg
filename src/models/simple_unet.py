"""Simple U-Net model for restoration tasks."""

import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    """
    Simple U-Net architecture for image/video restoration.

    Accepts input of shape (B, T, C, H, W) and outputs (B, C, H, W).
    For images, T=1. For videos, processes all T frames and outputs middle frame.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_frames: int = 1,
    ) -> None:
        """
        Initialize U-Net.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_channels: Base number of channels
            num_frames: Number of temporal frames (T dimension)
        """
        super().__init__()

        self.num_frames = num_frames
        self.input_shape = (num_frames, in_channels, 256, 256)  # For ONNX export

        # Calculate total input channels (T * C)
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

        # Output layer
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
            Output tensor of shape (B, C, H, W)
        """
        B, T, C, H, W = x.shape

        # Flatten temporal dimension into channels: (B, T, C, H, W) -> (B, T*C, H, W)
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

        # Output with Sigmoid activation to constrain to [0, 1]
        out = self.out(dec1)
        out = torch.sigmoid(out)

        return out
