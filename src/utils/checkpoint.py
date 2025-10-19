"""Checkpoint management."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages model checkpoints.

    Saves checkpoints, keeps track of best models, and handles cleanup.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 5,
        keep_best: int = 3,
    ) -> None:
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of regular checkpoints to keep
            keep_best: Number of best checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.best_scores: list[tuple[float, Path]] = []  # (score, path)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        scaler: Optional[torch.amp.GradScaler],
        ema: Optional[Any],
        epoch: int,
        step: int,
        loss: float,
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            scaler: Gradient scaler for mixed precision
            ema: EMA model state
            epoch: Current epoch
            step: Current step
            loss: Current loss value
            metrics: Current metrics
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": ema.state_dict() if ema is not None else None,
            "loss": loss,
            "metrics": metrics,
        }

        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"best_step_{step}.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"

        torch.save(checkpoint, checkpoint_path)

        # Update best checkpoints list
        if is_best:
            self.best_scores.append((loss, checkpoint_path))
            self.best_scores.sort(key=lambda x: x[0])  # Sort by loss (lower is better)

            # Remove excess best checkpoints
            while len(self.best_scores) > self.keep_best:
                _, path_to_remove = self.best_scores.pop()
                if path_to_remove.exists():
                    path_to_remove.unlink()

        # Clean up old regular checkpoints
        if not is_best:
            self._cleanup_old_checkpoints()

        return checkpoint_path

    def save_model_only(
        self,
        model: nn.Module,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save only the model weights (for export).

        Args:
            model: Model to save
            name: Name for the saved file
            metadata: Optional metadata to save alongside model

        Returns:
            Path to saved model
        """
        model_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(model.state_dict(), model_path)

        # Save metadata if provided
        if metadata is not None:
            metadata_path = self.checkpoint_dir / f"{name}_metadata.json"
            self._save_metadata(metadata_path, metadata)

        return model_path

    def _save_metadata(self, path: Path, metadata: dict[str, Any]) -> None:
        """
        Save metadata as JSON file.

        Args:
            path: Path to save metadata
            metadata: Metadata dictionary
        """
        # Add timestamp
        metadata['saved_at'] = datetime.now().isoformat()

        # Convert Path objects to strings
        metadata_serializable = {}
        for key, value in metadata.items():
            if isinstance(value, Path):
                metadata_serializable[key] = str(value)
            elif isinstance(value, (int, float, str, bool, list, dict)):
                metadata_serializable[key] = value
            else:
                metadata_serializable[key] = str(value)

        with open(path, 'w') as f:
            json.dump(metadata_serializable, f, indent=2)

    def get_best_checkpoint_metadata(self) -> dict[str, Any] | None:
        """
        Get metadata for the best checkpoint.

        Returns:
            Metadata dictionary or None if no best checkpoint exists
        """
        if not self.best_scores:
            return None

        best_loss, best_path = self.best_scores[0]
        return {
            'checkpoint_path': str(best_path),
            'loss': best_loss,
        }

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        ema: Optional[Any] = None,
    ) -> dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            scaler: Optional scaler to load state into
            ema: Optional EMA to load state into

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint["model"])

        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        if scaler is not None and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        if ema is not None and checkpoint["ema"] is not None:
            ema.load_state_dict(checkpoint["ema"])

        return checkpoint

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        # Get all regular checkpoint files
        checkpoints = sorted(
            [p for p in self.checkpoint_dir.glob("checkpoint_step_*.pt")],
            key=lambda p: p.stat().st_mtime,
        )

        # Remove oldest checkpoints
        while len(checkpoints) > self.max_checkpoints:
            checkpoint = checkpoints.pop(0)
            if checkpoint.exists():
                checkpoint.unlink()
