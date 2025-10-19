"""
Balanced Multi-Task GAN Trainer.

Extends GANTrainer with balanced discriminator updates for multi-task learning
(super-resolution + artifact removal + colorization).

Key features from balanced GAN training:
- Lower discriminator learning rate (5x slower than generator)
- Adaptive discriminator updates based on advantage threshold
- Optional discriminator update frequency control
"""

from pathlib import Path
from typing import Any, Optional, Literal

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from ..losses import GANLoss, MultiTaskGANLoss
from ..utils import CheckpointManager, EMA, Logger, export_to_onnx


class BalancedMultiTaskGANTrainer:
    """
    Balanced GAN trainer for multi-task restoration.

    Implements NoGAN training phases with balanced discriminator to prevent overpowering:
    1. "pretrain": Train generator only with content loss
    2. "critic": Train discriminator only on generated images
    3. "gan": Train both with balanced discriminator updates

    Balance mechanisms:
    - Slower discriminator learning rate (5x slower than generator)
    - Adaptive updates: skip disc update if advantage > threshold
    - Optional update frequency control
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
        experiment_id: str = "multitask_gan_001",
        # GAN settings
        gan_loss_type: Literal["vanilla", "lsgan", "hinge"] = "lsgan",
        gan_weight: float = 0.5,
        content_weight: float = 1.0,
        # Balanced discriminator settings
        disc_update_frequency: int = 1,
        use_adaptive_disc: bool = True,
        disc_advantage_threshold: float = 0.7,
        # Training settings
        use_amp: bool = True,
        gradient_clip: Optional[float] = 1.0,
        gradient_accumulation_steps: int = 1,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        use_compile: bool = False,
        multi_gpu: bool = False,
        # Logging settings
        log_interval: int = 100,
        checkpoint_interval: int = 1000,
        image_log_interval: int = 100,
    ) -> None:
        """
        Initialize balanced multi-task GAN trainer.

        Args:
            generator: Generator model (HAT for multi-task restoration)
            discriminator: Discriminator model (PatchGAN)
            content_loss: Content loss module (MultiTaskLoss)
            generator_optimizer: Optimizer for generator
            discriminator_optimizer: Optimizer for discriminator (slower LR!)
            device: Device to train on
            experiment_dir: Root directory for experiments
            experiment_id: Unique experiment identifier
            gan_loss_type: Type of GAN loss ('vanilla', 'lsgan', 'hinge')
            gan_weight: Weight for adversarial loss
            content_weight: Weight for content loss
            disc_update_frequency: Train disc every N generator updates
            use_adaptive_disc: Enable adaptive discriminator updates
            disc_advantage_threshold: Skip disc update if Real-Fake gap > this
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
        """
        self.device = device
        self.gradient_clip = gradient_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.image_log_interval = image_log_interval

        # Balanced discriminator settings
        self.disc_update_frequency = disc_update_frequency
        self.use_adaptive_disc = use_adaptive_disc
        self.disc_advantage_threshold = disc_advantage_threshold
        self.step_counter = 0

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
        self.combined_loss = MultiTaskGANLoss(
            content_loss=content_loss,
            gan_loss=self.gan_loss,
            content_weight=content_weight,
            gan_weight=gan_weight,
        )
        self.content_loss = content_loss
        self.content_weight = content_weight
        self.gan_weight = gan_weight

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
        """Set training phase for NoGAN training."""
        self.training_phase = phase
        self.logger.log_info(f"Training phase set to: {phase}")

    def amp_context(self):
        """Return AMP autocast context."""
        return autocast("cuda", enabled=self.use_amp)

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
        self.discriminator.eval()
        self.current_epoch = epoch

        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0

        pbar = tqdm(dataloader, desc=f"[PRETRAIN] Epoch {epoch}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            self.input_shape = inputs.shape[1:]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            with self.amp_context():
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
            epoch_loss += batch_loss
            epoch_psnr += psnr.item()
            epoch_ssim += ssim.item()

            pbar.set_postfix({
                "loss": batch_loss,
                "psnr": psnr.item(),
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

        num_batches = len(dataloader)
        return {
            "loss": epoch_loss / num_batches,
            "psnr": epoch_psnr / num_batches,
            "ssim": epoch_ssim / num_batches,
        }

    def _train_epoch_critic(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train discriminator only (NoGAN phase 2)."""
        self.generator.eval()
        self.discriminator.train()
        self.current_epoch = epoch

        epoch_disc_loss = 0.0
        epoch_disc_real = 0.0
        epoch_disc_fake = 0.0

        pbar = tqdm(dataloader, desc=f"[CRITIC] Epoch {epoch}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Extract input for discriminator (first frame if video)
            disc_input = inputs[:, 0] if inputs.dim() == 5 else inputs

            # Generate fake images
            with torch.no_grad():
                fake_output = self.generator(inputs)

            # For super-resolution, discriminator compares HR images
            # For colorization, it compares at same resolution
            # We need to resize disc_input to match target resolution
            if disc_input.shape[-2:] != targets.shape[-2:]:
                disc_input_resized = torch.nn.functional.interpolate(
                    disc_input, size=targets.shape[-2:], mode='bilinear', align_corners=False
                )
            else:
                disc_input_resized = disc_input

            # Forward pass through discriminator
            with self.amp_context():
                real_pred = self.discriminator(disc_input_resized, targets)
                fake_pred = self.discriminator(disc_input_resized, fake_output.detach())
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
            epoch_disc_loss += batch_disc_loss
            epoch_disc_real += real_pred.mean().item()
            epoch_disc_fake += fake_pred.mean().item()

            pbar.set_postfix({
                "disc_loss": batch_disc_loss,
                "real": real_pred.mean().item(),
                "fake": fake_pred.mean().item(),
            })

            # Logging
            if self.global_step % self.log_interval == 0:
                self.logger.log_metrics({
                    "disc_loss": batch_disc_loss,
                    "disc_real": real_pred.mean().item(),
                    "disc_fake": fake_pred.mean().item(),
                }, self.global_step, prefix="critic/")

        num_batches = len(dataloader)
        return {
            "disc_loss": epoch_disc_loss / num_batches,
            "disc_real": epoch_disc_real / num_batches,
            "disc_fake": epoch_disc_fake / num_batches,
        }

    def _train_epoch_gan(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train both generator and discriminator with BALANCED updates (NoGAN phase 3)."""
        self.generator.train()
        self.discriminator.train()
        self.current_epoch = epoch

        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        epoch_content_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_disc_real = 0.0
        epoch_disc_fake = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0

        disc_updates = 0
        disc_skips = 0

        pbar = tqdm(dataloader, desc=f"[GAN] Epoch {epoch}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            self.input_shape = inputs.shape[1:]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Extract input for discriminator
            disc_input = inputs[:, 0] if inputs.dim() == 5 else inputs

            # ==================== TRAIN GENERATOR ====================
            self.gen_optimizer.zero_grad()

            with self.amp_context():
                # Generate output
                fake_output = self.generator(inputs)

                # For super-resolution, resize disc_input to match output resolution
                if disc_input.shape[-2:] != targets.shape[-2:]:
                    disc_input_resized = torch.nn.functional.interpolate(
                        disc_input, size=targets.shape[-2:], mode='bilinear', align_corners=False
                    )
                else:
                    disc_input_resized = disc_input

                # Content loss
                content_result = self.content_loss(fake_output, targets)
                content_loss = content_result['total']

                # Adversarial loss
                fake_pred = self.discriminator(disc_input_resized, fake_output)
                adv_loss = self.gan_loss.generator_loss(fake_pred)

                # Combined loss
                gen_loss = self.content_weight * content_loss + self.gan_weight * adv_loss

            # Backward generator
            if self.use_amp:
                self.gen_scaler.scale(gen_loss).backward()
                if self.gradient_clip > 0:
                    self.gen_scaler.unscale_(self.gen_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.gradient_clip)
                self.gen_scaler.step(self.gen_optimizer)
                self.gen_scaler.update()
            else:
                gen_loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.gradient_clip)
                self.gen_optimizer.step()

            if self.gen_scheduler is not None:
                self.gen_scheduler.step()

            # ==================== TRAIN DISCRIMINATOR (BALANCED) ====================
            should_update_disc = False

            # Check update frequency
            if self.step_counter % self.disc_update_frequency == 0:
                should_update_disc = True

                # Check adaptive threshold if enabled
                if self.use_adaptive_disc:
                    with torch.no_grad():
                        real_pred = self.discriminator(disc_input_resized, targets)
                        fake_pred_check = self.discriminator(disc_input_resized, fake_output.detach())

                        real_score = real_pred.mean().item()
                        fake_score = fake_pred_check.mean().item()
                        advantage = real_score - fake_score

                        # Skip if discriminator is too strong
                        if advantage > self.disc_advantage_threshold:
                            should_update_disc = False
                            disc_skips += 1

            if should_update_disc:
                self.disc_optimizer.zero_grad()

                with self.amp_context():
                    # Real predictions
                    real_pred = self.discriminator(disc_input_resized, targets)

                    # Fake predictions (detach generator)
                    fake_pred = self.discriminator(disc_input_resized, fake_output.detach())

                    # Discriminator loss
                    disc_loss = self.gan_loss.discriminator_loss(real_pred, fake_pred)

                # Backward discriminator
                if self.use_amp:
                    self.disc_scaler.scale(disc_loss).backward()
                    if self.gradient_clip > 0:
                        self.disc_scaler.unscale_(self.disc_optimizer)
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip)
                    self.disc_scaler.step(self.disc_optimizer)
                    self.disc_scaler.update()
                else:
                    disc_loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip)
                    self.disc_optimizer.step()

                if self.disc_scheduler is not None:
                    self.disc_scheduler.step()

                disc_updates += 1
                epoch_disc_loss += disc_loss.item()
                epoch_disc_real += real_pred.mean().item()
                epoch_disc_fake += fake_pred.mean().item()

            # Update EMA
            if self.use_ema:
                self.ema.update()

            # Calculate metrics
            with torch.no_grad():
                psnr = self.psnr_metric(fake_output, targets)
                ssim = self.ssim_metric(fake_output, targets)

            # Accumulate metrics
            epoch_gen_loss += gen_loss.item()
            epoch_content_loss += content_loss.item()
            epoch_adv_loss += adv_loss.item()
            epoch_psnr += psnr.item()
            epoch_ssim += ssim.item()

            # Update progress bar
            real_score = epoch_disc_real / max(disc_updates, 1)
            fake_score = epoch_disc_fake / max(disc_updates, 1)
            advantage = real_score - fake_score

            pbar.set_postfix({
                'G': f"{gen_loss.item():.4f}",
                'D': f"{disc_loss.item():.4f}" if should_update_disc else "skip",
                'Adv': f"{advantage:.3f}",
                'PSNR': f"{psnr.item():.1f}",
            })

            # Logging
            if self.global_step % self.log_interval == 0:
                metrics = {
                    "gen_loss": gen_loss.item(),
                    "content_loss": content_loss.item(),
                    "adv_loss": adv_loss.item(),
                    "psnr": psnr.item(),
                    "ssim": ssim.item(),
                    "disc_advantage": advantage,
                    "disc_updates": disc_updates,
                    "disc_skips": disc_skips,
                }
                if should_update_disc:
                    metrics["disc_loss"] = disc_loss.item()
                    metrics["disc_real"] = real_pred.mean().item()
                    metrics["disc_fake"] = fake_pred.mean().item()

                # Add individual content losses
                for key, value in content_result.items():
                    if key != "total":
                        metrics[f"loss_{key}"] = value.item()

                self.logger.log_metrics(metrics, self.global_step, prefix="gan/")

            # Image logging
            if self.global_step % self.image_log_interval == 0:
                self.logger.log_images(
                    disc_input_resized,
                    fake_output,
                    targets,
                    self.global_step,
                )

            # Checkpointing
            if self.global_step % self.checkpoint_interval == 0:
                self._save_checkpoint(gen_loss.item(), {"psnr": psnr.item(), "ssim": ssim.item()})

            self.step_counter += 1
            self.global_step += 1

        n_batches = len(dataloader)
        n_disc_updates = max(disc_updates, 1)

        return {
            'gen_loss': epoch_gen_loss / n_batches,
            'disc_loss': epoch_disc_loss / n_disc_updates,
            'content_loss': epoch_content_loss / n_batches,
            'adv_loss': epoch_adv_loss / n_batches,
            'disc_real': epoch_disc_real / n_disc_updates,
            'disc_fake': epoch_disc_fake / n_disc_updates,
            'disc_advantage': (epoch_disc_real - epoch_disc_fake) / n_disc_updates,
            'disc_updates': disc_updates,
            'disc_skips': disc_skips,
            'psnr': epoch_psnr / n_batches,
            'ssim': epoch_ssim / n_batches,
        }

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

        # Only save EMA generator weights (better quality)
        if self.use_ema:
            self.ema.apply_shadow()
            gen_ema_name = f"generator_ema_step_{self.global_step}"

            # Prepare metadata for best checkpoint
            metadata = {
                "step": self.global_step,
                "epoch": self.current_epoch,
                "phase": self.training_phase,
                "loss": loss,
                "metrics": metrics,
                "is_best": is_best,
            }

            # Save EMA generator with metadata
            self.checkpoint_manager.save_model_only(
                gen_to_save,
                gen_ema_name,
                metadata=metadata if is_best else None,
            )

            # Export EMA generator to ONNX
            try:
                if hasattr(gen_to_save, "input_shape"):
                    input_shape = gen_to_save.input_shape
                elif hasattr(self, "input_shape"):
                    input_shape = tuple(self.input_shape)
                else:
                    input_shape = (1, 3, 256, 256)

                export_path_ema = self.export_dir / f"generator_ema_step_{self.global_step}.onnx"
                export_to_onnx(gen_to_save, export_path_ema, input_shape)

                # Save ONNX metadata for best checkpoint
                if is_best:
                    onnx_metadata = metadata.copy()
                    onnx_metadata["onnx_path"] = str(export_path_ema)
                    onnx_metadata["pytorch_path"] = str(self.checkpoint_dir / f"{gen_ema_name}.pt")
                    metadata_path = self.checkpoint_dir / "best_model_info.json"
                    self.checkpoint_manager._save_metadata(metadata_path, onnx_metadata)

                self.logger.log_info(f"ONNX export saved: {export_path_ema}")
            except Exception as e:
                self.logger.log_info(f"ONNX export failed: {e}")

            self.ema.restore()

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
            "Balanced Disc": True,
            "Disc Update Freq": self.disc_update_frequency,
            "Adaptive Disc": self.use_adaptive_disc,
            "Disc Threshold": self.disc_advantage_threshold,
            **kwargs,
        }
        self.logger.log_experiment_info(info)
