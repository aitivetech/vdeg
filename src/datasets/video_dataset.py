"""Video restoration dataset."""

from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder


class VideoRestorationDataset(Dataset):
    """
    Dataset for video restoration tasks.

    Returns tensors in TxCxHxW format where T is the number of frames.
    The target is the middle frame of the clip.
    """

    def __init__(
        self,
        root_dir: str | Path,
        target_size: tuple[int, int],
        target_fps: float,
        num_frames: int,
        clips_per_video: int,
        degradation_fn: Callable[[torch.Tensor], torch.Tensor],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        extensions: tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov", ".webm"),
    ) -> None:
        """
        Initialize video dataset.

        Args:
            root_dir: Root directory containing videos (searched recursively)
            target_size: Target size (height, width) for resizing
            target_fps: Target frame rate for resampling
            num_frames: Number of frames per clip (must be odd)
            clips_per_video: Number of clips to sample from each video
            degradation_fn: Function to degrade high-quality frames
            transform: Optional additional transforms after degradation
            extensions: Valid video file extensions
        """
        if num_frames % 2 == 0:
            raise ValueError(f"num_frames must be odd, got {num_frames}")

        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.target_fps = target_fps
        self.num_frames = num_frames
        self.clips_per_video = clips_per_video
        self.degradation_fn = degradation_fn
        self.transform = transform

        # Find all video files recursively
        self.video_paths = sorted([
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in extensions
        ])

        if not self.video_paths:
            raise ValueError(f"No videos found in {root_dir}")

    def __len__(self) -> int:
        """Return total number of clips."""
        return len(self.video_paths) * self.clips_per_video

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            input: Degraded video clip tensor of shape (T, C, H, W)
            target: Middle frame tensor of shape (C, H, W)
        """
        video_idx = idx // self.clips_per_video
        clip_idx = idx % self.clips_per_video

        video_path = str(self.video_paths[video_idx])

        # Load video
        decoder = VideoDecoder(video_path)

        # Get video metadata
        metadata = decoder.metadata
        total_frames = metadata["num_frames"]
        original_fps = metadata["fps"]

        # Calculate frame sampling rate
        frame_step = max(1, int(original_fps / self.target_fps))

        # Calculate available frame range for sampling
        available_frames = total_frames // frame_step
        if available_frames < self.num_frames:
            # If video is too short, use all frames with repetition
            start_frame = 0
        else:
            # Evenly space clips across video
            clip_spacing = (available_frames - self.num_frames) // max(1, self.clips_per_video - 1)
            start_frame = min(clip_idx * clip_spacing, available_frames - self.num_frames)

        # Extract frames
        frames = []
        for i in range(self.num_frames):
            frame_idx = (start_frame + i) * frame_step
            frame_idx = min(frame_idx, total_frames - 1)  # Clamp to valid range

            # Decode frame and get data
            frame_data = decoder.get_frame_at(frame_idx)
            frame = frame_data.data  # Tensor in [H, W, C] format

            # Resize and normalize
            frame = self._resize_and_crop(frame, self.target_size)
            frames.append(frame)

        # Stack frames: [T, H, W, C] -> [T, C, H, W]
        clip = torch.stack(frames, dim=0).permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        clip = clip.float() / 255.0

        # Extract middle frame as target
        middle_idx = self.num_frames // 2
        target = clip[middle_idx]  # [C, H, W]

        # Apply degradation
        degraded = self.degradation_fn(clip)

        # Apply additional transforms if any
        if self.transform is not None:
            degraded = self.transform(degraded)

        return degraded, target

    def _resize_and_crop(self, frame: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """
        Resize and center crop frame to target size without distortion.

        Args:
            frame: Frame tensor in [H, W, C] format
            target_size: (height, width)

        Returns:
            Resized and cropped frame in [H, W, C] format
        """
        target_h, target_w = target_size
        h, w, c = frame.shape

        # Calculate scaling to maintain aspect ratio
        scale = max(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize using interpolation
        # Convert to [C, H, W] for interpolation
        frame = frame.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        frame = torch.nn.functional.interpolate(
            frame,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        frame = frame.squeeze(0).permute(1, 2, 0)  # [H, W, C]

        # Center crop
        top = (new_h - target_h) // 2
        left = (new_w - target_w) // 2
        frame = frame[top : top + target_h, left : left + target_w]

        return frame
