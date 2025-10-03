"""Compression artifacts degradation."""

import io
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms as T


class JPEGCompression:
    """Simulate JPEG compression artifacts."""

    def __init__(self, quality: int = 50) -> None:
        """
        Initialize JPEG compression degradation.

        Args:
            quality: JPEG quality (1-100, lower = more artifacts)
        """
        if not 1 <= quality <= 100:
            raise ValueError(f"Quality must be in [1, 100], got {quality}")
        self.quality = quality

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply JPEG compression.

        Args:
            x: Input tensor of shape (T, C, H, W) in range [0, 1]

        Returns:
            Compressed tensor
        """
        T_dim, C, H, W = x.shape
        result = torch.zeros_like(x)

        for t in range(T_dim):
            frame = x[t]  # [C, H, W]

            # Convert to PIL Image
            frame_pil = T.ToPILImage()(frame)

            # Compress and decompress
            buffer = io.BytesIO()
            frame_pil.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            frame_pil = Image.open(buffer)

            # Convert back to tensor
            result[t] = T.ToTensor()(frame_pil)

        return result
