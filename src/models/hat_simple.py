"""
Simplified HAT: Hybrid Attention Transformer for Image Restoration

Simplified version without OCAB for easier integration.
Based on: https://arxiv.org/abs/2309.05239

Adapted for BxTxCxHxW format (T=1 for images) -> BxCxHxW output
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    """Channel attention (squeeze-and-excitation)."""

    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    """Channel Attention Block (conv + channel attention)."""

    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    """MLP feed-forward network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HybridAttentionBlock(nn.Module):
    """Simplified Hybrid Attention Block.

    Combines convolutional features with channel attention and MLP.
    No window attention to avoid resolution dependencies.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.,
        drop: float = 0.,
        drop_path: float = 0.,
        compress_ratio: int = 3,
        squeeze_factor: int = 30
    ):
        super().__init__()
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.conv_block = CAB(dim, compress_ratio, squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Conv + channel attention branch
        shortcut = x
        conv_x = self.conv_block(x)  # (B, C, H, W)

        # Add residual
        x = shortcut + self.drop_path(conv_x)

        # MLP branch (process as tokens)
        x_tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_tokens = x_tokens + self.drop_path(self.mlp(self.norm2(x_tokens)))
        x = x_tokens.transpose(1, 2).view(B, C, H, W)

        return x


class ResidualGroup(nn.Module):
    """Residual Group of Hybrid Attention Blocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        mlp_ratio: float = 4.,
        drop: float = 0.,
        drop_path: float = 0.,
        compress_ratio: int = 3,
        squeeze_factor: int = 30
    ):
        super().__init__()
        self.dim = dim

        self.blocks = nn.ModuleList([
            HybridAttentionBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor
            )
            for i in range(depth)
        ])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        res = x
        for blk in self.blocks:
            res = blk(res)
        res = self.conv(res)
        return x + res


class Upsample(nn.Sequential):
    """Upsample module using pixel shuffle."""

    def __init__(self, scale: int, num_feat: int):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class HATSimple(nn.Module):
    """Simplified Hybrid Attention Transformer.

    Adapted for BxTxCxHxW format (T=1 for images) -> BxCxHxW output.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 96,
        depths: list = [6, 6, 6, 6],
        mlp_ratio: float = 4.,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        num_frames: int = 1,  # For compatibility with our framework
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_frames = num_frames
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # Store input shape for ONNX export
        self.input_shape = (num_frames, in_channels, 256, 256)

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build Residual Groups
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)

        # Reconstruction (same-size output)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_final = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C, H, W) where T=1 for images

        Returns:
            Output tensor of shape (B, C, H_out, W_out)
        """
        # Handle BxTxCxHxW format -> BxCxHxW
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            assert T == 1, f"HATSimple expects T=1, got T={T}"
            x = x.squeeze(1)  # (B, C, H, W)

        # Shallow feature extraction
        x = self.conv_first(x)

        # Deep feature extraction
        res = x
        for layer in self.layers:
            res = layer(res)

        res = self.conv_after_body(res)
        x = x + res

        # Reconstruction (same size as input)
        x = self.conv_before_final(x)
        x = self.conv_last(x)

        # Ensure output is in [0, 1] range
        x = torch.sigmoid(x)

        return x


def hat_simple_s(**kwargs):
    """HAT-Simple Small model (same-size input/output)."""
    # Remove upscale if provided (no longer used)
    kwargs.pop('upscale', None)
    return HATSimple(
        embed_dim=64,
        depths=[6, 6, 6, 6],
        **kwargs
    )


def hat_simple_m(**kwargs):
    """HAT-Simple Medium model (same-size input/output)."""
    # Remove upscale if provided (no longer used)
    kwargs.pop('upscale', None)
    return HATSimple(
        embed_dim=96,
        depths=[6, 6, 6, 6],
        **kwargs
    )


def hat_simple_l(**kwargs):
    """HAT-Simple Large model (same-size input/output)."""
    # Remove upscale if provided (no longer used)
    kwargs.pop('upscale', None)
    return HATSimple(
        embed_dim=128,
        depths=[6, 6, 6, 6, 6, 6],
        **kwargs
    )
