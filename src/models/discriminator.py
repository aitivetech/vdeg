"""
PatchGAN Discriminator for colorization quality assessment.

Based on the pix2pix discriminator architecture, adapted for colorization.
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator that classifies image patches as real or fake.

    This discriminator outputs a matrix of values rather than a single scalar,
    where each value represents the "realness" of a 70x70 patch of the input.
    This is more effective for image quality assessment than global discrimination.

    For colorization, the discriminator receives:
    - Grayscale input (L channel) concatenated with color channels (AB or RGB)
    - Learns to distinguish real colorization from generated colorization
    """

    def __init__(
        self,
        input_channels: int = 6,  # 3 (grayscale) + 3 (color)
        num_filters: int = 64,
        num_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        """
        Initialize PatchGAN discriminator.

        Args:
            input_channels: Number of input channels (grayscale + color)
            num_filters: Number of filters in first layer
            num_layers: Number of downsampling layers
            use_spectral_norm: Use spectral normalization for training stability
        """
        super().__init__()

        # Use spectral norm for better training stability (from NoGAN research)
        norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x

        layers = []

        # Initial layer: no normalization
        layers.append(
            nn.Sequential(
                norm_layer(nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        # Downsampling layers with instance normalization
        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(
                nn.Sequential(
                    norm_layer(nn.Conv2d(
                        num_filters * nf_mult_prev,
                        num_filters * nf_mult,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    )),
                    nn.InstanceNorm2d(num_filters * nf_mult),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        # Additional layer with stride 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        layers.append(
            nn.Sequential(
                norm_layer(nn.Conv2d(
                    num_filters * nf_mult_prev,
                    num_filters * nf_mult,
                    kernel_size=4,
                    stride=1,
                    padding=1
                )),
                nn.InstanceNorm2d(num_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        # Output layer: produces patch predictions
        layers.append(
            norm_layer(nn.Conv2d(
                num_filters * nf_mult,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            ))
        )

        self.model = nn.Sequential(*layers)

    def forward(self, grayscale: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of discriminator.

        Args:
            grayscale: Grayscale input of shape (B, C, H, W) - usually 3 channels (RGB grayscale)
            color: Color prediction/target of shape (B, C, H, W) - 3 channels

        Returns:
            Patch predictions of shape (B, 1, H', W') where each value represents
            the "realness" of a patch. Values are not bounded (use with BCE loss + sigmoid).
        """
        # Concatenate grayscale and color as input
        x = torch.cat([grayscale, color], dim=1)  # (B, 6, H, W)
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator that operates at multiple image resolutions.

    This can help capture both fine details and overall color coherence.
    Uses multiple PatchDiscriminators at different scales.
    """

    def __init__(
        self,
        input_channels: int = 6,
        num_filters: int = 64,
        num_discriminators: int = 2,
        num_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        """
        Initialize multi-scale discriminator.

        Args:
            input_channels: Number of input channels
            num_filters: Base number of filters
            num_discriminators: Number of discriminators at different scales
            num_layers: Number of layers in each discriminator
            use_spectral_norm: Use spectral normalization
        """
        super().__init__()

        self.num_discriminators = num_discriminators

        # Create multiple discriminators
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(
                input_channels=input_channels,
                num_filters=num_filters,
                num_layers=num_layers,
                use_spectral_norm=use_spectral_norm,
            )
            for _ in range(num_discriminators)
        ])

        # Downsampling for multi-scale input
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, grayscale: torch.Tensor, color: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass of multi-scale discriminator.

        Args:
            grayscale: Grayscale input
            color: Color prediction/target

        Returns:
            List of patch predictions from each scale, from finest to coarsest
        """
        results = []

        gray_scaled = grayscale
        color_scaled = color

        for i, discriminator in enumerate(self.discriminators):
            results.append(discriminator(gray_scaled, color_scaled))

            # Downsample for next scale (except for last discriminator)
            if i < self.num_discriminators - 1:
                gray_scaled = self.downsample(gray_scaled)
                color_scaled = self.downsample(color_scaled)

        return results
