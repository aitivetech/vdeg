"""Image restoration dataset."""

from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class ImageRestorationDataset(Dataset):
    """
    Dataset for image restoration tasks.

    Returns tensors in TxCxHxW format where T=1 for images.
    The degradation function transforms the high-quality image into degraded input.
    """

    def __init__(
        self,
        root_dir: str | Path,
        target_size: tuple[int, int],
        degradation_fn: Callable[[torch.Tensor], torch.Tensor],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        limit: int | None = None,
    ) -> None:
        """
        Initialize image dataset.

        Args:
            root_dir: Root directory containing images (searched recursively)
            target_size: Target size (height, width) for both input and target
            degradation_fn: Function to degrade high-quality images (same size in/out)
            transform: Optional additional transforms after degradation
            extensions: Valid image file extensions
            limit: Maximum number of images to load (None = all)
        """
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.degradation_fn = degradation_fn
        self.transform = transform

        # Find all image files recursively
        self.image_paths = sorted([
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in extensions
        ])

        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir}")

        if limit is not None:
            self.image_paths = self.image_paths[:limit]

        # Normalization transform
        self.normalize = T.Compose([
            T.ToTensor(),  # Converts to [0, 1] and CHW format
        ])

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            input: Degraded image tensor of shape (1, C, H, W)
            target: Original image tensor of shape (C, H, W)
        """
        # Load and normalize image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Resize to target size using high-quality Lanczos resampling
        # Maintain aspect ratio by cropping to fit
        image = self._resize_and_crop(image, self.target_size)

        # Convert to tensor [C, H, W] in range [0, 1]
        target = self.normalize(image)

        # Apply degradation to create input
        # Add temporal dimension: [C, H, W] -> [1, C, H, W]
        target_with_time = target.unsqueeze(0)
        degraded_with_time = self.degradation_fn(target_with_time)

        # Apply additional transforms if any
        if self.transform is not None:
            degraded_with_time = self.transform(degraded_with_time)

        return degraded_with_time, target

    def _resize_and_crop(self, image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        """
        Resize and center crop image to target size without distortion.

        Args:
            image: PIL Image
            target_size: (height, width)

        Returns:
            Resized and cropped image
        """
        target_h, target_w = target_size
        img_w, img_h = image.size

        # Calculate scaling to maintain aspect ratio
        scale = max(target_h / img_h, target_w / img_w)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h

        return image.crop((left, top, right, bottom))
