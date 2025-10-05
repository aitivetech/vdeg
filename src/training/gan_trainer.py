"""
GAN Trainer for colorization with NoGAN training support.

Extends the base Trainer class to support adversarial training with
a discriminator network. Implements NoGAN training phases.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Literal

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from ..losses import GANLoss, ColorizationGANLoss
from ..utils import CheckpointManager, EMA, Logger, export_to_onnx


class GANTrainer:
    """
    Trainer for colorization models with GAN and NoGAN training.

    NoGAN Training Phases:
    1. "pretrain": Train generator only with content loss
    2. "critic": Train discriminator only on generated images
    3. "gan": Train both generator and discriminator adversarially
    4. Can cycle between phases for best results

    Features (inherited from base Trainer):
    - Mixed precision training (AMP)
    - Gradient clipping and accumulation
    - EMA for model parameters
    - Checkpointing with best model tracking
    - TensorBoard logging
    - Multi-GPU support
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        content_loss: nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        experiment_dir: str | Path = "./experiments",
        experiment_id: str = "colorization_gan_001",
        # GAN settings
        gan_loss_type: Literal["vanilla", "lsgan", "hinge"] = "lsgan",
        gan_weight: float = 0.1,
        content_weight: float = 1.0,
        feature_matching_weight: float = 0.0,
        # Training settings
        use_amp: bool = True,
        gradient_clip: Optional[float] = 1.0,
        gradient_accumulation_steps: int = 1,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        use_compile: bool = False,
        multi_gpu: bool = False,
        # Logging settings
        log_interval: int = 10,
        checkpoint_interval: int = 1000,
        image_log_interval: int = 500,
        # NoGAN settings
        critic_threshold: float = 0.5,  # Discriminator loss threshold before training generator
    ) -> None:
        """
        Initialize GAN trainer.

        Args:
            generator: Generator model (colorization network)
            discriminator: Discriminator model (PatchGAN)
            content_loss: Content loss module (e.g., perceptual + LAB loss)
            generator_optimizer: Optimizer for generator
            discriminator_optimizer: Optimizer for discriminator
            device: Device to train on
            experiment_dir: Root directory for experiments
            experiment_id: Unique experiment identifier
            gan_loss_type: Type of GAN loss ('vanilla', 'lsgan', 'hinge')
            gan_weight: Weight for adversarial loss
            content_weight: Weight for content loss
            feature_matching_weight: Weight for feature matching loss
            use_amp: Use automatic mixed precision
            gradient_clip: Gradient clipping value
            gradient_accumulation_steps: Number of gradient accumulation steps
            use_ema: Use exponential moving average for generator
            ema_decay: EMA decay rate
            use_compile: Use torch.compile
            multi_gpu: Use DataParallel for multi-GPU
            log_interval: Steps between metric logging
            checkpoint_interval: Steps between checkpoints
            image_log_interval: Steps between image logging
            critic_threshold: NoGAN critic loss threshold
        """
        self.device = device
        self.gradient_clip = gradient_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.image_log_interval = image_log_interval
        self.critic_threshold = critic_threshold

        # Setup experiment directories
        exp_path = Path(experiment_dir) / experiment_id
        self.checkpoint_dir = exp_path / "checkpoints"
        self.log_dir = exp_path / "logs"
        self.export_dir = exp_path / "exports"

        for dir_path in [self.checkpoint_dir, self.log_dir, self.export_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup models
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        if multi_gpu and torch.cuda.device_count() > 1:
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)

        if use_compile:
            self.generator = torch.compile(self.generator)
            self.discriminator = torch.compile(self.discriminator)

        # Setup losses
        self.gan_loss = GANLoss(loss_type=gan_loss_type)
        self.combined_loss = ColorizationGANLoss(
            content_loss=content_loss,
            gan_loss=self.gan_loss,
            content_weight=content_weight,
            gan_weight=gan_weight,
            feature_matching_weight=feature_matching_weight,
        )
        self.content_loss = content_loss

        # Setup optimizers
        self.gen_optimizer = generator_optimizer
        self.disc_optimizer = discriminator_optimizer

        # Setup mixed precision
        self.use_amp = use_amp
        self.gen_scaler = GradScaler("cuda") if use_amp else None
        self.disc_scaler = GradScaler("cuda") if use_amp else None

        # Setup EMA for generator
        self.use_ema = use_ema
        self.ema = EMA(self.generator, decay=ema_decay) if use_ema else None

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
        self.training_phase: Literal["pretrain", "critic", "gan"] = "pretrain"

        # Scheduler placeholders
        self.gen_scheduler: Optional[Any] = None
        self.disc_scheduler: Optional[Any] = None

    def set_phase(self, phase: Literal["pretrain", "critic", "gan"]) -> None:
        """
        Set training phase for NoGAN training.

        Args:
            phase: Training phase ('pretrain', 'critic', or 'gan')
        """
        self.training_phase = phase
        self.logger.log_info(f"Training phase set to: {phase}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """
        Train for one epoch based on current phase.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number

        Returns:
            Dictionary of average metrics for the epoch
        """
        if self.training_phase == "pretrain":
            return self._train_epoch_pretrain(dataloader, epoch)
        elif self.training_phase == "critic":
            return self._train_epoch_critic(dataloader, epoch)
        elif self.training_phase == "gan":
            return self._train_epoch_gan(dataloader, epoch)
        else:
            raise ValueError(f"Unknown training phase: {self.training_phase}")

    def _train_epoch_pretrain(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train generator only with content loss (NoGAN phase 1)."""
        self.generator.train()
        self.discriminator.eval()  # Not used in pretrain
        self.current_epoch = epoch

        epoch_metrics = {
            "loss": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
        }

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [PRETRAIN]")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # inputs: (B, T, C, H, W) - grayscale, targets: (B, C, H, W) - color
            self.input_shape = inputs.shape[1:]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            with autocast("cuda", enabled=self.use_amp):
                predictions = self.generator(inputs)
                loss_output = self.content_loss(predictions, targets)

                if isinstance(loss_output, dict):
                    loss = loss_output["total"]
                else:
                    loss = loss_output

                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.gen_scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.gradient_clip is not None:
                    if self.use_amp:
                        self.gen_scaler.unscale_(self.gen_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), self.gradient_clip
                    )

                if self.use_amp:
                    self.gen_scaler.step(self.gen_optimizer)
                    self.gen_scaler.update()
                else:
                    self.gen_optimizer.step()

                self.gen_optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

                if self.gen_scheduler is not None:
                    self.gen_scheduler.step()

                self.global_step += 1

            # Compute metrics
            with torch.no_grad():
                psnr = self.psnr_metric(predictions, targets)
                ssim = self.ssim_metric(predictions, targets)

            batch_loss = loss.item() * self.gradient_accumulation_steps
            epoch_metrics["loss"] += batch_loss
            epoch_metrics["psnr"] += psnr.item()
            epoch_metrics["ssim"] += ssim.item()

            pbar.set_postfix({
                "loss": batch_loss,
                "psnr": psnr.item(),
                "phase": "pretrain",
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

                self.logger.log_metrics(metrics, self.global_step, prefix="pretrain/")

            # Image logging
            if self.global_step % self.image_log_interval == 0:
                self.logger.log_images(
                    inputs[:, 0] if inputs.dim() == 5 else inputs,
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

    def _train_epoch_critic(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train discriminator only on generated images (NoGAN phase 2)."""
        self.generator.eval()
        self.discriminator.train()
        self.current_epoch = epoch

        epoch_metrics = {
            "disc_loss": 0.0,
            "disc_real": 0.0,
            "disc_fake": 0.0,
        }

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [CRITIC]")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Get grayscale version for discriminator input
            # For colorization: inputs are already grayscale, targets are color
            grayscale = inputs[:, 0] if inputs.dim() == 5 else inputs

            # Generate fake colorization
            with torch.no_grad():
                fake_color = self.generator(inputs)

            # Forward pass through discriminator
            with autocast("cuda", enabled=self.use_amp):
                # Real images
                real_pred = self.discriminator(grayscale, targets)
                # Fake images
                fake_pred = self.discriminator(grayscale, fake_color.detach())

                # Discriminator loss
                disc_loss = self.gan_loss.discriminator_loss(real_pred, fake_pred)
                disc_loss = disc_loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.disc_scaler.scale(disc_loss).backward()
            else:
                disc_loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.gradient_clip is not None:
                    if self.use_amp:
                        self.disc_scaler.unscale_(self.disc_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), self.gradient_clip
                    )

                if self.use_amp:
                    self.disc_scaler.step(self.disc_optimizer)
                    self.disc_scaler.update()
                else:
                    self.disc_optimizer.step()

                self.disc_optimizer.zero_grad()

                if self.disc_scheduler is not None:
                    self.disc_scheduler.step()

                self.global_step += 1

            batch_disc_loss = disc_loss.item() * self.gradient_accumulation_steps
            epoch_metrics["disc_loss"] += batch_disc_loss
            epoch_metrics["disc_real"] += real_pred.mean().item()
            epoch_metrics["disc_fake"] += fake_pred.mean().item()

            pbar.set_postfix({
                "disc_loss": batch_disc_loss,
                "phase": "critic",
            })

            # Logging
            if self.global_step % self.log_interval == 0:
                self.logger.log_metrics({
                    "disc_loss": batch_disc_loss,
                    "disc_real": real_pred.mean().item(),
                    "disc_fake": fake_pred.mean().item(),
                }, self.global_step, prefix="critic/")

        # Calculate average metrics
        num_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def _train_epoch_gan(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train both generator and discriminator adversarially (NoGAN phase 3)."""
        self.generator.train()
        self.discriminator.train()
        self.current_epoch = epoch

        epoch_metrics = {
            "gen_loss": 0.0,
            "disc_loss": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
        }

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [GAN]")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            self.input_shape = inputs.shape[1:]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            grayscale = inputs[:, 0] if inputs.dim() == 5 else inputs

            # ==================== Train Discriminator ====================
            # Generate fake images
            with torch.no_grad():
                fake_color = self.generator(inputs)

            with autocast("cuda", enabled=self.use_amp):
                real_pred = self.discriminator(grayscale, targets)
                fake_pred = self.discriminator(grayscale, fake_color.detach())
                disc_loss = self.gan_loss.discriminator_loss(real_pred, fake_pred)
                disc_loss = disc_loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.disc_scaler.scale(disc_loss).backward()
            else:
                disc_loss.backward()

            # ==================== Train Generator ====================
            with autocast("cuda", enabled=self.use_amp):
                fake_color = self.generator(inputs)
                fake_pred = self.discriminator(grayscale, fake_color)

                # Combined generator loss (content + adversarial)
                gen_loss_dict = self.combined_loss.generator_loss_combined(
                    fake_color, targets, fake_pred
                )
                gen_loss = gen_loss_dict["total"]
                gen_loss = gen_loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.gen_scaler.scale(gen_loss).backward()
            else:
                gen_loss.backward()

            # ==================== Update Weights ====================
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Update discriminator
                if self.gradient_clip is not None:
                    if self.use_amp:
                        self.disc_scaler.unscale_(self.disc_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), self.gradient_clip
                    )

                if self.use_amp:
                    self.disc_scaler.step(self.disc_optimizer)
                    self.disc_scaler.update()
                else:
                    self.disc_optimizer.step()

                self.disc_optimizer.zero_grad()

                # Update generator
                if self.gradient_clip is not None:
                    if self.use_amp:
                        self.gen_scaler.unscale_(self.gen_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), self.gradient_clip
                    )

                if self.use_amp:
                    self.gen_scaler.step(self.gen_optimizer)
                    self.gen_scaler.update()
                else:
                    self.gen_optimizer.step()

                self.gen_optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

                if self.gen_scheduler is not None:
                    self.gen_scheduler.step()
                if self.disc_scheduler is not None:
                    self.disc_scheduler.step()

                self.global_step += 1

            # Compute metrics
            with torch.no_grad():
                psnr = self.psnr_metric(fake_color, targets)
                ssim = self.ssim_metric(fake_color, targets)

            batch_gen_loss = gen_loss.item() * self.gradient_accumulation_steps
            batch_disc_loss = disc_loss.item() * self.gradient_accumulation_steps
            epoch_metrics["gen_loss"] += batch_gen_loss
            epoch_metrics["disc_loss"] += batch_disc_loss
            epoch_metrics["psnr"] += psnr.item()
            epoch_metrics["ssim"] += ssim.item()

            pbar.set_postfix({
                "gen": batch_gen_loss,
                "disc": batch_disc_loss,
                "psnr": psnr.item(),
                "phase": "gan",
            })

            # Logging
            if self.global_step % self.log_interval == 0:
                metrics = {
                    "gen_loss": batch_gen_loss,
                    "disc_loss": batch_disc_loss,
                    "psnr": psnr.item(),
                    "ssim": ssim.item(),
                }
                for key, value in gen_loss_dict.items():
                    if key != "total":
                        metrics[f"gen_{key}"] = value.item()

                self.logger.log_metrics(metrics, self.global_step, prefix="gan/")

            # Image logging
            if self.global_step % self.image_log_interval == 0:
                self.logger.log_images(
                    grayscale,
                    fake_color,
                    targets,
                    self.global_step,
                )

            # Checkpointing
            if self.global_step % self.checkpoint_interval == 0:
                self._save_checkpoint(batch_gen_loss, {"psnr": psnr.item(), "ssim": ssim.item()})

        # Calculate average metrics
        num_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def _save_checkpoint(self, loss: float, metrics: dict[str, float]) -> None:
        """Save checkpoint for both generator and discriminator."""
        is_best = loss < self.best_loss
        if is_best:
            self.best_loss = loss

        # Get actual models (unwrap DataParallel if needed)
        gen_to_save = (
            self.generator.module if isinstance(self.generator, nn.DataParallel) else self.generator
        )
        disc_to_save = (
            self.discriminator.module if isinstance(self.discriminator, nn.DataParallel) else self.discriminator
        )

        # Save checkpoint with both models
        checkpoint = {
            "epoch": self.current_epoch,
            "step": self.global_step,
            "phase": self.training_phase,
            "generator_state_dict": gen_to_save.state_dict(),
            "discriminator_state_dict": disc_to_save.state_dict(),
            "gen_optimizer_state_dict": self.gen_optimizer.state_dict(),
            "disc_optimizer_state_dict": self.disc_optimizer.state_dict(),
            "loss": loss,
            "metrics": metrics,
        }

        if self.gen_scheduler is not None:
            checkpoint["gen_scheduler_state_dict"] = self.gen_scheduler.state_dict()
        if self.disc_scheduler is not None:
            checkpoint["disc_scheduler_state_dict"] = self.disc_scheduler.state_dict()
        if self.use_amp:
            checkpoint["gen_scaler_state_dict"] = self.gen_scaler.state_dict()
            checkpoint["disc_scaler_state_dict"] = self.disc_scaler.state_dict()
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)

        self.logger.log_info(f"Checkpoint saved: {checkpoint_path} (best={is_best})")

        # Save generator weights only
        gen_path = self.checkpoint_dir / f"generator_step_{self.global_step}.pt"
        torch.save(gen_to_save.state_dict(), gen_path)

        # Save EMA generator
        if self.use_ema:
            self.ema.apply_shadow()
            gen_ema_path = self.checkpoint_dir / f"generator_ema_step_{self.global_step}.pt"
            torch.save(gen_to_save.state_dict(), gen_ema_path)
            self.ema.restore()

        # Export to ONNX
        try:
            if hasattr(gen_to_save, "input_shape"):
                input_shape = gen_to_save.input_shape
            elif hasattr(self, "input_shape"):
                input_shape = tuple(self.input_shape)
            else:
                input_shape = (1, 3, 256, 256)

            export_path = self.export_dir / f"generator_step_{self.global_step}.onnx"
            export_to_onnx(gen_to_save, export_path, input_shape)

            if self.use_ema:
                self.ema.apply_shadow()
                export_path_ema = self.export_dir / f"generator_ema_step_{self.global_step}.onnx"
                export_to_onnx(gen_to_save, export_path_ema, input_shape)
                self.ema.restore()

            self.logger.log_info(f"ONNX export saved: {export_path}")
        except Exception as e:
            self.logger.log_info(f"ONNX export failed: {e}")

    def log_experiment_info(self, **kwargs: Any) -> None:
        """Log experiment configuration."""
        info = {
            "Experiment ID": Path(self.checkpoint_dir).parent.name,
            "Device": self.device,
            "Mixed Precision": self.use_amp,
            "Gradient Clipping": self.gradient_clip,
            "Gradient Accumulation": self.gradient_accumulation_steps,
            "EMA": self.use_ema,
            "Training Phase": self.training_phase,
            **kwargs,
        }
        self.logger.log_experiment_info(info)
