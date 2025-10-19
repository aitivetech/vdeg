
"""
HAT: Hybrid Attention Transformer for Image Restoration

Based on:
    "Activating More Pixels in Image Super-Resolution Transformer"
    https://arxiv.org/abs/2309.05239
    https://github.com/XPixelGroup/HAT

Adapted for BxTxCxHxW format (T=1 for images) -> BxCxHxW output
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange


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


def window_partition(x, window_size: int):
    """Partition into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """Reverse window partition.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: tuple,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAB(nn.Module):
    """Hybrid Attention Block.

    Combines window-based self-attention with convolutional features and channel attention.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        compress_ratio: int = 3,
        squeeze_factor: int = 30
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.conv_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.conv_block = CAB(dim, compress_ratio, squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, x_size):
        """
        Args:
            x: Input features (B, H*W, C)
            x_size: (H, W)
        """
        H, W = x_size
        if x.ndim != 3:
            raise ValueError(f"HAB expected 3D tensor (B, L, C), got shape {x.shape}")
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # Conv branch
        conv_x = self.conv_block(shortcut.view(B, H, W, C).permute(0, 3, 1, 2).contiguous())
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x) + self.conv_scale * conv_x
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class OCAB(nn.Module):
    """Overlapping Cross-Attention Block."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        window_size: int,
        overlap_ratio: float,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        mlp_ratio: float = 2.,
        drop_path: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2
        )

        # Relative position
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        coords_h_overlapping = torch.arange(self.overlap_win_size)
        coords_w_overlapping = torch.arange(self.overlap_win_size)
        coords_overlapping = torch.stack(torch.meshgrid([coords_h_overlapping, coords_w_overlapping], indexing='ij'))
        coords_overlapping_flatten = torch.flatten(coords_overlapping, 1)

        relative_coords = coords_flatten[:, :, None] - coords_overlapping_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.overlap_win_size - 1
        relative_coords[:, :, 1] += self.overlap_win_size - 1
        relative_coords[:, :, 0] *= self.window_size + self.overlap_win_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, x_size):
        """
        Args:
            x: Input features (B, H*W, C)
            x_size: (H, W)
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        qkv = self.qkv(x).reshape(B, H, W, 3, C).permute(3, 0, 4, 1, 2)
        q = qkv[0].permute(0, 2, 3, 1)  # B, H, W, C
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # B, 2*C, H, W

        # Overlapping cross-attention
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)
        kv_windows = self.unfold(kv)
        kv_windows = rearrange(
            kv_windows,
            'b (nc ch owh oww) nw -> (b nw) (owh oww) (nc ch)',
            nc=2,
            ch=C,
            owh=self.overlap_win_size,
            oww=self.overlap_win_size
        ).contiguous()
        k_windows, v_windows = kv_windows.chunk(2, dim=-1)

        b_, nq, _ = q_windows.shape
        _, nk, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)
        k = k_windows.reshape(b_, nk, self.num_heads, d).permute(0, 2, 1, 3)
        v = v_windows.reshape(b_, nk, self.num_heads, d).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W).view(B, H * W, C)
        x = self.proj(x)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class AttenBlocks(nn.Module):
    """A series of attention blocks for one RHAG."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        overlap_ratio: float = 0.5
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                HAB(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor
                )
            )

        self.overlap_attn = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path[-1] if isinstance(drop_path, list) else drop_path
        )

    def forward(self, x, x_size):
        if x.ndim != 3:
            raise ValueError(f"AttenBlocks expected 3D tensor, got shape {x.shape}")
        for i, blk in enumerate(self.blocks):
            x = blk(x, x_size)
            if x.ndim != 3:
                raise ValueError(f"After HAB block {i}: expected 3D tensor, got shape {x.shape}")
        x = self.overlap_attn(x, x_size)
        if x.ndim != 3:
            raise ValueError(f"After OCAB: expected 3D tensor, got shape {x.shape}")
        return x


class RHAG(nn.Module):
    """Residual Hybrid Attention Group (RHAG)."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        overlap_ratio: float = 0.5
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.blocks = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            overlap_ratio=overlap_ratio
        )

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        if x.ndim != 3:
            raise ValueError(f"RHAG expected 3D tensor, got shape {x.shape}")
        res = self.blocks(x, x_size)
        # Conv residual
        res = res.transpose(1, 2).view(-1, self.dim, x_size[0], x_size[1])
        res = self.conv(res)
        res = res.view(-1, self.dim, x_size[0] * x_size[1]).transpose(1, 2)
        return x + res


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size: int = 224, patch_size: int = 1, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if patch_size == 1:
            self.proj = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        return x


class PatchUnEmbed(nn.Module):
    """Patch to Image."""

    def __init__(self, img_size: int = 224, patch_size: int = 1, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


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


class HAT(nn.Module):
    """Hybrid Attention Transformer.

    Adapted for BxTxCxHxW format (T=1 for images) -> BxCxHxW output.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 96,
        depths: list = [6, 6, 6, 6],
        num_heads: list = [6, 6, 6, 6],
        window_size: int = 7,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        overlap_ratio: float = 0.5,
        upscale: int = 4,
        img_size: int = 64,
        patch_size: int = 1,
        num_frames: int = 1,  # For compatibility with our framework
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_frames = num_frames
        self.upscale = upscale
        self.img_size = img_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim
        )

        self.patches_resolution = self.patch_embed.patches_resolution

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build Residual Hybrid Attention Groups (RHAG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                overlap_ratio=overlap_ratio
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)

        # Reconstruction
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(upscale, embed_dim)
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
            assert T == 1, f"HAT expects T=1, got T={T}"
            x = x.squeeze(1)  # (B, C, H, W)
        else:
            B, C, H, W = x.shape

        # Ensure image size is divisible by window_size
        # Pad if necessary
        mod_pad_h = (self.window_size - H % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - W % self.window_size) % self.window_size

        if mod_pad_h != 0 or mod_pad_w != 0:
            x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            _, _, H_pad, W_pad = x.shape
        else:
            H_pad, W_pad = H, W

        # Compute x_size for the patches
        x_size = (H_pad, W_pad)

        # Shallow feature extraction
        x = self.patch_embed(x)

        # Debug: check shape after patch_embed
        if x.ndim != 3:
            raise ValueError(f"After patch_embed: expected 3D tensor, got shape {x.shape}")

        # Deep feature extraction
        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        # Reconstruction
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)

        # Remove padding if it was added
        if mod_pad_h != 0 or mod_pad_w != 0:
            H_out = H * self.upscale
            W_out = W * self.upscale
            x = x[:, :, :H_out, :W_out]

        # Ensure output is in [0, 1] range
        x = torch.sigmoid(x)

        return x


def hat_s(**kwargs):
    """HAT-S (Small) model."""
    return HAT(
        embed_dim=64,
        depths=[6, 6, 6, 6],
        num_heads=[4, 4, 4, 4],
        **kwargs
    )


def hat_m(**kwargs):
    """HAT-M (Medium) model."""
    return HAT(
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        **kwargs
    )


def hat_l(**kwargs):
    """HAT-L (Large) model."""
    return HAT(
        embed_dim=128,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[8, 8, 8, 8, 8, 8],
        **kwargs
    )
