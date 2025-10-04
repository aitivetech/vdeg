"""Main trainer class for restoration models."""

from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from ..utils import CheckpointManager, EMA, Logger, export_to_onnx


class Trainer:
    """
    Trainer for image/video restoration models.

    Features:
    - Mixed precision training (AMP)
    - Gradient clipping and accumulation
    - EMA for model parameters
    - Checkpointing with best model tracking
    - TensorBoard logging
    - Multi-GPU support
    - torch.compile support
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | Callable,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        experiment_dir: str | Path = "./experiments",
        experiment_id: str = "exp_001",
        use_amp: bool = True,
        gradient_clip: Optional[float] = 1.0,
        gradient_accumulation_steps: int = 1,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        use_compile: bool = False,
        multi_gpu: bool = False,
        log_interval: int = 10,
        checkpoint_interval: int = 1000,
        image_log_interval: int = 500,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Model to train
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device to train on
            experiment_dir: Root directory for experiments
            experiment_id: Unique experiment identifier
            use_amp: Use automatic mixed precision
            gradient_clip: Gradient clipping value (None to disable)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            use_ema: Use exponential moving average
            ema_decay: EMA decay rate
            use_compile: Use torch.compile for faster training
            multi_gpu: Use DataParallel for multi-GPU
            log_interval: Steps between metric logging
            checkpoint_interval: Steps between checkpoints
            image_log_interval: Steps between image logging
        """
        self.device = device
        self.gradient_clip = gradient_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.image_log_interval = image_log_interval

        # Setup experiment directories
        exp_path = Path(experiment_dir) / experiment_id
        self.checkpoint_dir = exp_path / "checkpoints"
        self.log_dir = exp_path / "logs"
        self.export_dir = exp_path / "exports"

        for dir_path in [self.checkpoint_dir, self.log_dir, self.export_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup model
        self.model = model.to(device)

        if multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.logger.log_info(f"Using {torch.cuda.device_count()} GPUs")

        if use_compile:
            self.model = torch.compile(self.model)

        # Setup loss and optimizer
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Setup mixed precision
        self.use_amp = use_amp
        self.scaler = GradScaler("cuda") if use_amp else None

        # Setup EMA
        self.use_ema = use_ema
        self.ema = EMA(self.model, decay=ema_decay) if use_ema else None

        # Setup metrics
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        # Setup utilities
        self.logger = Logger(self.log_dir, experiment_id)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")

        # Scheduler placeholder (can be set externally)
        self.scheduler: Optional[Any] = None

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        self.current_epoch = epoch

        epoch_metrics = {
            "loss": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
        }

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            # inputs: (B, T, C, H, W), targets: (B, C, H, W)
            self.input_shape = inputs.shape[1:]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            with autocast("cuda", enabled=self.use_amp):
                predictions = self.model(inputs)  # (B, C, H, W)
                loss_output = self.loss_fn(predictions, targets)

                # Handle both dict and scalar loss outputs
                if isinstance(loss_output, dict):
                    loss = loss_output["total"]
                else:
                    loss = loss_output

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update EMA
                if self.use_ema:
                    self.ema.update()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            # Compute metrics
            with torch.no_grad():
                psnr = self.psnr_metric(predictions, targets)
                ssim = self.ssim_metric(predictions, targets)

            # Accumulate metrics
            batch_loss = loss.item() * self.gradient_accumulation_steps
            epoch_metrics["loss"] += batch_loss
            epoch_metrics["psnr"] += psnr.item()
            epoch_metrics["ssim"] += ssim.item()

            # Update progress bar
            pbar.set_postfix({
                "loss": batch_loss,
                "psnr": psnr.item(),
                "ssim": ssim.item(),
            })

            # Logging
            if self.global_step % self.log_interval == 0:
                metrics = {
                    "loss": batch_loss,
                    "psnr": psnr.item(),
                    "ssim": ssim.item(),
                }
                if isinstance(loss_output, dict):
                    for key, value in loss_output.items():
                        if key != "total":
                            metrics[f"loss_{key}"] = value.item()

                self.logger.log_metrics(metrics, self.global_step, prefix="train/")

            # Image logging
            if self.global_step % self.image_log_interval == 0:
                self.logger.log_images(
                    inputs[:, 0] if inputs.dim() == 5 else inputs,  # First frame for videos
                    predictions,
                    targets,
                    self.global_step,
                )

            # Checkpointing
            if self.global_step % self.checkpoint_interval == 0:
                self._save_checkpoint(batch_loss, {"psnr": psnr.item(), "ssim": ssim.item()})

        # Calculate average metrics
        num_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def _save_checkpoint(self, loss: float, metrics: dict[str, float]) -> None:
        """Save checkpoint and handle best model tracking."""
        is_best = loss < self.best_loss
        if is_best:
            self.best_loss = loss

        # Get actual model (unwrap DataParallel if needed)
        model_to_save = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )

        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model_to_save,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.ema,
            self.current_epoch,
            self.global_step,
            loss,
            metrics,
            is_best=is_best,
        )

        self.logger.log_info(
            f"Checkpoint saved: {checkpoint_path} (best={is_best})"
        )

        # Save model weights
        self.checkpoint_manager.save_model_only(model_to_save, f"model_step_{self.global_step}")

        # Save EMA model weights
        if self.use_ema:
            self.ema.apply_shadow()
            self.checkpoint_manager.save_model_only(
                model_to_save, f"model_ema_step_{self.global_step}"
            )
            self.ema.restore()

        # Export to ONNX
        try:
            # Get input shape from model (assuming it has an input_shape attribute)
            # Otherwise use a default
            if hasattr(model_to_save, "input_shape"):
                input_shape = model_to_save.input_shape
            elif hasattr(self, "input_shape"):
                input_shape = tuple(self.input_shape)
            else:
                input_shape = (1, 3, 256, 256)  # Default T, C, H, W

            export_path = self.export_dir / f"model_step_{self.global_step}.onnx"
            export_to_onnx(model_to_save, export_path, input_shape)

            if self.use_ema:
                self.ema.apply_shadow()
                export_path_ema = self.export_dir / f"model_ema_step_{self.global_step}.onnx"
                export_to_onnx(model_to_save, export_path_ema, input_shape)
                self.ema.restore()

            self.logger.log_info(f"ONNX export saved: {export_path}")
        except Exception as e:
            self.logger.log_info(f"ONNX export failed: {e}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load checkpoint and resume training."""
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.ema,
        )

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["step"]
        self.best_loss = checkpoint["loss"]

        self.logger.log_info(
            f"Resumed from checkpoint: epoch={self.current_epoch}, step={self.global_step}"
        )

    def log_experiment_info(self, **kwargs: Any) -> None:
        """Log experiment configuration."""
        info = {
            "Experiment ID": Path(self.checkpoint_dir).parent.name,
            "Device": self.device,
            "Mixed Precision": self.use_amp,
            "Gradient Clipping": self.gradient_clip,
            "Gradient Accumulation": self.gradient_accumulation_steps,
            "EMA": self.use_ema,
            **kwargs,
        }
        self.logger.log_experiment_info(info)
