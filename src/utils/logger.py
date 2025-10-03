"""Logging utilities for training."""

import sys
from pathlib import Path
from typing import Any

import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


class Logger:
    """
    Logger for training with console and TensorBoard output.

    Handles metrics logging, image logging, and progress tracking.
    """

    def __init__(self, log_dir: str | Path, experiment_name: str) -> None:
        """
        Initialize logger.

        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.writer = SummaryWriter(str(self.log_dir / "tensorboard"))

    def log_info(self, message: str) -> None:
        """
        Log informational message to console.

        Args:
            message: Message to log
        """
        print(f"[INFO] {message}", flush=True)

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        """
        Log metrics to console and TensorBoard.

        Args:
            metrics: Dictionary of metric names and values
            step: Current step/iteration
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}{name}", value, step)

        # Log to console (summary line)
        metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        print(f"[METRICS] Step {step} | {metric_str}", flush=True)

    def log_images(
        self,
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        step: int,
        max_images: int = 8,
    ) -> None:
        """
        Log comparison images to TensorBoard.

        Args:
            inputs: Input images of shape (B, C, H, W)
            predictions: Predicted images of shape (B, C, H, W)
            targets: Target images of shape (B, C, H, W)
            step: Current step/iteration
            max_images: Maximum number of images to log
        """
        # Limit number of images
        n = min(max_images, inputs.size(0))
        inputs = inputs[:n]
        predictions = predictions[:n]
        targets = targets[:n]

        # Clamp to [0, 1] for visualization
        inputs = torch.clamp(inputs, 0.0, 1.0)
        predictions = torch.clamp(predictions, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)

        # Create comparison grid
        # Stack horizontally: [input | prediction | target]
        comparison = torch.cat([inputs, predictions, targets], dim=3)  # Concatenate along width

        # Create grid
        grid = make_grid(comparison, nrow=1, padding=2, normalize=False)

        # Log to TensorBoard
        self.writer.add_image("comparisons", grid, step)

    def log_experiment_info(self, info: dict[str, Any]) -> None:
        """
        Log experiment configuration and info to console.

        Args:
            info: Dictionary of experiment information
        """
        self.log_info("=" * 80)
        self.log_info(f"Experiment: {self.experiment_name}")
        self.log_info("=" * 80)

        for key, value in info.items():
            self.log_info(f"{key}: {value}")

        self.log_info("=" * 80)

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
