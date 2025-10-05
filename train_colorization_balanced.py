"""
BALANCED GAN Training script for image colorization.

This version fixes the discriminator overpowering issue by:
- Lower discriminator learning rate (5x slower than generator)
- Optional discriminator training frequency control
- Careful monitoring of discriminator/generator balance
- Adaptive discriminator updates based on performance gap

Key insight from debugging:
- Discriminator was achieving perfect separation (0.97 vs 0.01) by epoch 3
- This caused generator to lose color (AB variance dropped from 75 to 17)
- Solution: Keep discriminator weaker so it provides useful gradients
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import ImageRestorationDataset
from src.degradations import Grayscale
from src.losses import ColorizationLossFixed
from src.models import SimpleUNetColorization, PatchDiscriminator
from src.training import GANTrainer

# =============================================================================
# CONFIGURATION - BALANCED GAN SETTINGS
# =============================================================================

# Experiment settings
EXPERIMENT_ID = "colorization_balanced_gan_001"
EXPERIMENT_DIR = "./experiments"

# Dataset settings
DATASET_PATH = "/media/bglueck/Data/datasets/laion_high/laion-output"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
NUM_WORKERS = 4
LIMIT = 20000  # Full dataset

# Model settings
MODEL_TYPE = "SimpleUNet"
MODEL_CHANNELS = 96
NUM_FRAMES = 1
UPSCALE_FACTOR = 1

# Discriminator settings
DISC_NUM_FILTERS = 64
DISC_NUM_LAYERS = 3
USE_SPECTRAL_NORM = True

# Training settings - NoGAN phases with BALANCED updates
PRETRAIN_EPOCHS = 10
CRITIC_EPOCHS = 2
GAN_EPOCHS = 10
NUM_CYCLES = 5

# Learning rates - CRITICAL BALANCE
GEN_LEARNING_RATE = 1e-4
DISC_LEARNING_RATE = 2e-5  # 5x SLOWER than generator (was 2e-4)
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 1000

# Discriminator training frequency
# Train discriminator only every N generator updates
DISC_UPDATE_FREQUENCY = 1  # 1 = every step, 2 = every other step, etc.

# Adaptive discriminator updates
# Stop training discriminator if it gets too strong
USE_ADAPTIVE_DISC = True
DISC_ADVANTAGE_THRESHOLD = 0.7  # If Real-Fake gap > 0.7, skip disc update

# Gradient settings
USE_GRADIENT_ACCUMULATION = True
ACCUMULATION_STEPS = 1
GRADIENT_CLIP = 1.0

# Loss settings
GAN_LOSS_TYPE = "lsgan"
CONTENT_WEIGHT = 0.5
GAN_WEIGHT = 0.5
PERCEPTUAL_WEIGHT = 0.5
LAB_WEIGHT = 1.0
AB_WEIGHT = 3.0

# Trainer settings
USE_AMP = True
USE_EMA = True
EMA_DECAY = 0.999
USE_COMPILE = False
MULTI_GPU = False

# Logging and checkpointing
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 1000
IMAGE_LOG_INTERVAL = 50

# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =============================================================================
# BALANCED GAN TRAINER - EXTENDS BASE GANTRAINER
# =============================================================================

class BalancedGANTrainer(GANTrainer):
    """GAN trainer with discriminator balancing to prevent overpowering."""

    def __init__(self, *args, disc_update_frequency: int = 1,
                 use_adaptive_disc: bool = True,
                 disc_advantage_threshold: float = 0.7,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.disc_update_frequency = disc_update_frequency
        self.use_adaptive_disc = use_adaptive_disc
        self.disc_advantage_threshold = disc_advantage_threshold
        self.step_counter = 0

        # Store weights for manual loss computation
        self.content_weight = kwargs.get('content_weight', 1.0)
        self.gan_weight = kwargs.get('gan_weight', 0.1)

    def amp_context(self):
        """Return AMP autocast context."""
        from torch.amp import autocast
        return autocast("cuda", enabled=self.use_amp)

    def _train_epoch_gan(self, dataloader: DataLoader, epoch: int) -> dict:
        """GAN training with balanced discriminator updates."""
        self.generator.train()
        self.discriminator.train()

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

        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"[GAN] Epoch {epoch}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Extract grayscale for discriminator
            # Input shape: (B, T, C, H, W) where T=1 for images, C=3 (replicated grayscale)
            # We need all 3 channels of first frame: (B, 3, H, W)
            grayscale = inputs[:, 0, :, :, :]  # B,T,C,H,W -> B,C,H,W

            # =============================================
            # TRAIN GENERATOR
            # =============================================
            self.gen_optimizer.zero_grad()

            with self.amp_context():
                # Generate fake colorization
                fake_color = self.generator(inputs)

                # Content loss
                content_result = self.content_loss(fake_color, targets)
                content_loss = content_result['total']

                # Adversarial loss
                fake_pred = self.discriminator(grayscale, fake_color)
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

            # =============================================
            # TRAIN DISCRIMINATOR (WITH BALANCING)
            # =============================================
            should_update_disc = False

            # Check update frequency
            if self.step_counter % self.disc_update_frequency == 0:
                should_update_disc = True

                # Check adaptive threshold if enabled
                if self.use_adaptive_disc:
                    with torch.no_grad():
                        real_pred = self.discriminator(grayscale, targets)
                        fake_pred_check = self.discriminator(grayscale, fake_color.detach())

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
                    real_pred = self.discriminator(grayscale, targets)

                    # Fake predictions (detach generator)
                    fake_pred = self.discriminator(grayscale, fake_color.detach())

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
                psnr = 10 * torch.log10(1 / torch.mean((fake_color - targets) ** 2))
                from torchmetrics.functional import structural_similarity_index_measure as ssim_func
                ssim = ssim_func(fake_color, targets, data_range=1.0)

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
                self._save_checkpoint(gen_loss.item(), {"psnr": psnr.item(), "ssim": ssim.item()})

            self.step_counter += 1
            self.global_step += 1

        n_batches = len(dataloader)
        n_disc_updates = max(disc_updates, 1)

        metrics = {
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

        return metrics


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main() -> None:
    """Main training function with BALANCED GAN."""
    print("=" * 80)
    print("BALANCED GAN COLORIZATION TRAINING")
    print("=" * 80)
    print("\n🔧 BALANCE FIXES APPLIED:")
    print(f"  • Discriminator LR: {DISC_LEARNING_RATE} (5x slower than generator)")
    print(f"  • Disc update frequency: Every {DISC_UPDATE_FREQUENCY} step(s)")
    print(f"  • Adaptive disc: {'ON' if USE_ADAPTIVE_DISC else 'OFF'}")
    if USE_ADAPTIVE_DISC:
        print(f"  • Advantage threshold: {DISC_ADVANTAGE_THRESHOLD} (skip if Real-Fake > this)")
    print("=" * 80)

    # Create grayscale degradation
    degradation = Grayscale()

    # Create dataset
    print(f"\nLoading dataset from: {DATASET_PATH}")
    dataset = ImageRestorationDataset(
        root_dir=DATASET_PATH,
        target_size=IMAGE_SIZE,
        degradation_fn=degradation,
        upscale_factor=UPSCALE_FACTOR,
        limit=LIMIT,
    )
    print(f"Dataset size: {len(dataset)} images")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Batches per epoch: {len(dataloader)}")

    # Create generator
    print(f"\nInitializing generator: {MODEL_TYPE}")
    generator = SimpleUNetColorization(
        in_channels=3,
        out_channels=3,
        base_channels=MODEL_CHANNELS,
        num_frames=NUM_FRAMES,
    )
    gen_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {gen_params:,}")

    # Create discriminator
    print(f"\nInitializing discriminator: PatchGAN")
    discriminator = PatchDiscriminator(
        input_channels=6,  # grayscale + color
        num_filters=DISC_NUM_FILTERS,
        num_layers=DISC_NUM_LAYERS,
        use_spectral_norm=USE_SPECTRAL_NORM,
    )
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator parameters: {disc_params:,}")

    # Create content loss
    content_loss = ColorizationLossFixed(
        rgb_weight=0.3,
        perceptual_weight=PERCEPTUAL_WEIGHT,
        ab_weight=AB_WEIGHT,
        perceptual_net="vgg",
        device=DEVICE,
    )

    # Create optimizers
    gen_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=GEN_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=DISC_LEARNING_RATE,  # 5x slower!
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    # Create learning rate schedulers
    total_pretrain_steps = PRETRAIN_EPOCHS * len(dataloader)
    total_gan_steps = GAN_EPOCHS * len(dataloader) * NUM_CYCLES

    def gen_lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        else:
            total_steps = total_pretrain_steps + total_gan_steps
            progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    def disc_lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        else:
            total_steps = total_gan_steps
            progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, gen_lr_lambda)
    disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, disc_lr_lambda)

    # Create BALANCED GAN trainer
    trainer = BalancedGANTrainer(
        generator=generator,
        discriminator=discriminator,
        content_loss=content_loss,
        generator_optimizer=gen_optimizer,
        discriminator_optimizer=disc_optimizer,
        device=DEVICE,
        experiment_dir=EXPERIMENT_DIR,
        experiment_id=EXPERIMENT_ID,
        gan_loss_type=GAN_LOSS_TYPE,
        gan_weight=GAN_WEIGHT,
        content_weight=CONTENT_WEIGHT,
        use_amp=USE_AMP,
        gradient_clip=GRADIENT_CLIP,
        gradient_accumulation_steps=ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1,
        use_ema=USE_EMA,
        ema_decay=EMA_DECAY,
        use_compile=USE_COMPILE,
        multi_gpu=MULTI_GPU,
        log_interval=LOG_INTERVAL,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        image_log_interval=IMAGE_LOG_INTERVAL,
        # Balancing parameters
        disc_update_frequency=DISC_UPDATE_FREQUENCY,
        use_adaptive_disc=USE_ADAPTIVE_DISC,
        disc_advantage_threshold=DISC_ADVANTAGE_THRESHOLD,
    )

    trainer.gen_scheduler = gen_scheduler
    trainer.disc_scheduler = disc_scheduler

    # Log experiment info
    experiment_info = {
        "ModelType": MODEL_TYPE,
        "Dataset": DATASET_PATH,
        "DatasetSize": len(dataset),
        "ImageSize": IMAGE_SIZE,
        "BatchSize": BATCH_SIZE,
        "GenLearningRate": GEN_LEARNING_RATE,
        "DiscLearningRate": DISC_LEARNING_RATE,
        "DiscLRRatio": f"1:{GEN_LEARNING_RATE/DISC_LEARNING_RATE:.1f}",
        "DiscUpdateFreq": DISC_UPDATE_FREQUENCY,
        "AdaptiveDisc": USE_ADAPTIVE_DISC,
        "AdvantageThreshold": DISC_ADVANTAGE_THRESHOLD,
        "GANWeight": GAN_WEIGHT,
        "ContentWeight": CONTENT_WEIGHT,
        "PretrainEpochs": PRETRAIN_EPOCHS,
        "CriticEpochs": CRITIC_EPOCHS,
        "GANEpochs": GAN_EPOCHS,
        "NumCycles": NUM_CYCLES,
        "BALANCED_VERSION": True,
    }

    trainer.log_experiment_info(**experiment_info)

    print("\nStarting NoGAN training with BALANCED GAN...")
    print("=" * 80)

    # Phase 1: PRETRAIN
    print("\n" + "=" * 80)
    print("PHASE 1: PRETRAIN GENERATOR (Content Loss Only)")
    print("=" * 80)
    trainer.set_phase("pretrain")

    for epoch in range(PRETRAIN_EPOCHS):
        epoch_metrics = trainer.train_epoch(dataloader, epoch)
        print(f"\nPretrain Epoch {epoch} Summary:")
        print(f"  Loss: {epoch_metrics['loss']:.6f}")
        print(f"  PSNR: {epoch_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {epoch_metrics['ssim']:.4f}")
        print("-" * 80)

    # NoGAN Cycles
    for cycle in range(NUM_CYCLES):
        print("\n" + "=" * 80)
        print(f"CYCLE {cycle + 1}/{NUM_CYCLES}")
        print("=" * 80)

        # Phase 2: CRITIC
        print("\n" + "=" * 80)
        print("PHASE 2: PRETRAIN CRITIC (Discriminator Only)")
        print("=" * 80)
        trainer.set_phase("critic")

        for epoch in range(CRITIC_EPOCHS):
            epoch_metrics = trainer.train_epoch(dataloader, epoch)
            print(f"\nCritic Epoch {epoch} Summary:")
            print(f"  Disc Loss: {epoch_metrics['disc_loss']:.6f}")
            print(f"  Disc Real: {epoch_metrics['disc_real']:.4f}")
            print(f"  Disc Fake: {epoch_metrics['disc_fake']:.4f}")
            print("-" * 80)

        # Phase 3: BALANCED GAN
        print("\n" + "=" * 80)
        print("PHASE 3: BALANCED ADVERSARIAL TRAINING")
        print(f"🎨 Discriminator is BALANCED (slower LR, adaptive updates)")
        print("=" * 80)
        trainer.set_phase("gan")

        for epoch in range(GAN_EPOCHS):
            epoch_metrics = trainer.train_epoch(dataloader, epoch)
            advantage = epoch_metrics.get('disc_advantage', 0)
            disc_updates = epoch_metrics.get('disc_updates', 0)
            disc_skips = epoch_metrics.get('disc_skips', 0)

            print(f"\nGAN Epoch {epoch} Summary:")
            print(f"  Gen Loss: {epoch_metrics['gen_loss']:.6f}")
            print(f"  Disc Loss: {epoch_metrics['disc_loss']:.6f}")
            print(f"  Disc Advantage: {advantage:.4f} (Real-Fake gap)")
            print(f"  Disc Updates: {disc_updates}, Skips: {disc_skips}")
            print(f"  PSNR: {epoch_metrics['psnr']:.2f} dB")
            print(f"  SSIM: {epoch_metrics['ssim']:.4f}")
            print("-" * 80)

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Logs saved to: {trainer.log_dir}")
    print(f"\n🔍 Run diagnose_simple.py to check color variance!")
    print("=" * 80)


if __name__ == "__main__":
    main()
