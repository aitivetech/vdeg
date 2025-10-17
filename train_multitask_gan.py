"""
Balanced Multi-Task GAN Training with NoGAN strategy.

Trains a HAT model on multiple tasks (SR + artifact removal + colorization)
using balanced adversarial training for improved realism.

NoGAN Training Strategy:
1. PRETRAIN: Train generator with content loss only
2. CRITIC: Train discriminator on generated images
3. GAN: Train both with balanced discriminator
4. Cycle between phases for best results

Balanced Discriminator:
- 5x slower learning rate than generator
- Adaptive updates: skip if discriminator gets too strong
- Prevents discriminator overpowering seen in colorization
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import ImageRestorationDataset
from src.degradations import (
    GaussianNoise,
    JPEGCompression,
    GaussianBlur,
    Grayscale,
    SuperResolutionDegradation,
    MultiTaskDegradation,
)
from src.losses import MultiTaskLoss, GANLoss
from src.models import PatchDiscriminator, hat_simple_s, hat_simple_m, hat_simple_l
from src.training import BalancedMultiTaskGANTrainer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment settings
EXPERIMENT_ID = "multitask_gan_balanced_001"
EXPERIMENT_DIR = "./experiments"

# Dataset settings
DATASET_PATH = "/media/bglueck/Data/datasets/laion_high/laion-output"
LIMIT = 1000
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_WORKERS = 4

# Model settings
MODEL_TYPE = "hat_simple_s"  # "hat_simple_s", "hat_simple_m", "hat_simple_l"
UPSCALE_FACTOR = 1
NUM_FRAMES = 1

# Discriminator settings
DISC_NUM_FILTERS = 64
DISC_NUM_LAYERS = 3
USE_SPECTRAL_NORM = True

# Training settings - NoGAN phases with BALANCED discriminator
PRETRAIN_EPOCHS = 10
CRITIC_EPOCHS = 2
GAN_EPOCHS = 10
NUM_CYCLES = 5  # Number of NoGAN cycles

# Learning rates - CRITICAL BALANCE
GEN_LEARNING_RATE = 1e-4
DISC_LEARNING_RATE = 2e-5  # 5x SLOWER than generator (key to balance!)
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 1000

# Discriminator balancing
DISC_UPDATE_FREQUENCY = 1  # 1 = every step, 2 = every other step
USE_ADAPTIVE_DISC = True  # Enable adaptive discriminator updates
DISC_ADVANTAGE_THRESHOLD = 0.7  # Skip disc update if Real-Fake gap > 0.7

# Gradient settings
USE_GRADIENT_ACCUMULATION = True
ACCUMULATION_STEPS = 2
GRADIENT_CLIP = 1.0

# Task settings
ENABLE_SUPER_RESOLUTION = False
ENABLE_ARTIFACT_REMOVAL = False
ENABLE_COLORIZATION = True

# Degradation probabilities
DOWNSCALE_PROB = 1.0
NOISE_PROB = 0.7
BLUR_PROB = 0.5
COMPRESSION_PROB = 0.8
GRAYSCALE_PROB = 1.0

# Degradation parameters
NOISE_SIGMA = 0.05
JPEG_QUALITY = 70
BLUR_KERNEL_SIZE = 3
BLUR_SIGMA = 0.5

# Loss settings
GAN_LOSS_TYPE = "lsgan"  # "vanilla", "lsgan", "hinge"
CONTENT_WEIGHT = 1.0
GAN_WEIGHT = 0.5
RGB_WEIGHT = 1.0
PERCEPTUAL_WEIGHT = 0.5
AB_WEIGHT = 2.0
PERCEPTUAL_NET = "vgg"

# Trainer settings
USE_AMP = True
USE_EMA = True
EMA_DECAY = 0.999
USE_COMPILE = False
MULTI_GPU = False

# Logging and checkpointing
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 5000
IMAGE_LOG_INTERVAL = 100

# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SETUP
# =============================================================================

def main() -> None:
    """Main multi-task GAN training function."""
    print("=" * 80)
    print("BALANCED MULTI-TASK GAN TRAINING")
    print("=" * 80)
    print("\n🔧 BALANCE FIXES APPLIED:")
    print(f"  • Discriminator LR: {DISC_LEARNING_RATE} (5x slower than generator)")
    print(f"  • Disc update frequency: Every {DISC_UPDATE_FREQUENCY} step(s)")
    print(f"  • Adaptive disc: {'ON' if USE_ADAPTIVE_DISC else 'OFF'}")
    if USE_ADAPTIVE_DISC:
        print(f"  • Advantage threshold: {DISC_ADVANTAGE_THRESHOLD} (skip if Real-Fake > this)")
    print("\n📋 Enabled tasks:")
    print(f"  • Super-resolution: {ENABLE_SUPER_RESOLUTION} ({UPSCALE_FACTOR}x)")
    print(f"  • Artifact removal: {ENABLE_ARTIFACT_REMOVAL}")
    print(f"  • Colorization: {ENABLE_COLORIZATION}")
    print("\n🔄 NoGAN Strategy:")
    print(f"  • Pretrain: {PRETRAIN_EPOCHS} epochs")
    print(f"  • Critic: {CRITIC_EPOCHS} epochs per cycle")
    print(f"  • GAN: {GAN_EPOCHS} epochs per cycle")
    print(f"  • Cycles: {NUM_CYCLES}")
    print("=" * 80)

    # Create individual degradation functions
    noise_fn = GaussianNoise(sigma=NOISE_SIGMA)
    blur_fn = GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA)
    compression_fn = JPEGCompression(quality=JPEG_QUALITY)
    grayscale_fn = Grayscale()
    downscale_fn = SuperResolutionDegradation(
        scale_factor=UPSCALE_FACTOR,
        mode="bilinear"
    )

    # Create multi-task degradation pipeline
    degradation = MultiTaskDegradation(
        downscale_fn=downscale_fn if ENABLE_SUPER_RESOLUTION else None,
        noise_fn=noise_fn if ENABLE_ARTIFACT_REMOVAL else None,
        blur_fn=blur_fn if ENABLE_ARTIFACT_REMOVAL else None,
        compression_fn=compression_fn if ENABLE_ARTIFACT_REMOVAL else None,
        grayscale_fn=grayscale_fn if ENABLE_COLORIZATION else None,
        downscale_prob=DOWNSCALE_PROB,
        noise_prob=NOISE_PROB,
        blur_prob=BLUR_PROB,
        compression_prob=COMPRESSION_PROB,
        grayscale_prob=GRAYSCALE_PROB,
    )

    # Create dataset
    print(f"\nLoading dataset from: {DATASET_PATH}")
    dataset = ImageRestorationDataset(
        root_dir=DATASET_PATH,
        target_size=IMAGE_SIZE,
        degradation_fn=degradation,
        upscale_factor=UPSCALE_FACTOR,
        limit=LIMIT
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

    # Create generator (HAT model)
    print(f"\nInitializing generator: {MODEL_TYPE}")
    if MODEL_TYPE == "hat_simple_s":
        generator = hat_simple_s(
            in_channels=3,
            out_channels=3,
            upscale=UPSCALE_FACTOR,
            num_frames=NUM_FRAMES,
        )
    elif MODEL_TYPE == "hat_simple_m":
        generator = hat_simple_m(
            in_channels=3,
            out_channels=3,
            upscale=UPSCALE_FACTOR,
            num_frames=NUM_FRAMES,
        )
    elif MODEL_TYPE == "hat_simple_l":
        generator = hat_simple_l(
            in_channels=3,
            out_channels=3,
            upscale=UPSCALE_FACTOR,
            num_frames=NUM_FRAMES,
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    gen_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {gen_params:,}")

    # Create discriminator (PatchGAN)
    print(f"\nInitializing discriminator: PatchGAN")
    discriminator = PatchDiscriminator(
        input_channels=6,  # Input (degraded) + output (restored)
        num_filters=DISC_NUM_FILTERS,
        num_layers=DISC_NUM_LAYERS,
        use_spectral_norm=USE_SPECTRAL_NORM,
    )
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator parameters: {disc_params:,}")

    # Create content loss (multi-task)
    content_loss = MultiTaskLoss(
        rgb_weight=RGB_WEIGHT,
        perceptual_weight=PERCEPTUAL_WEIGHT,
        ab_weight=AB_WEIGHT,
        perceptual_net=PERCEPTUAL_NET,
        device=DEVICE,
        enable_colorization=ENABLE_COLORIZATION,
        enable_super_resolution=ENABLE_SUPER_RESOLUTION or ENABLE_ARTIFACT_REMOVAL,
    )

    # Create optimizers with BALANCED learning rates
    gen_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=GEN_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=DISC_LEARNING_RATE,  # 5x slower! Critical for balance
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

    # Create BALANCED multi-task GAN trainer
    trainer = BalancedMultiTaskGANTrainer(
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
        # Balancing parameters
        disc_update_frequency=DISC_UPDATE_FREQUENCY,
        use_adaptive_disc=USE_ADAPTIVE_DISC,
        disc_advantage_threshold=DISC_ADVANTAGE_THRESHOLD,
        # Training settings
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
        # Task configuration
        "SuperResolution": ENABLE_SUPER_RESOLUTION,
        "ArtifactRemoval": ENABLE_ARTIFACT_REMOVAL,
        "Colorization": ENABLE_COLORIZATION,
        # Degradation probabilities
        "DownscaleProb": DOWNSCALE_PROB,
        "NoiseProb": NOISE_PROB,
        "BlurProb": BLUR_PROB,
        "CompressionProb": COMPRESSION_PROB,
        "GrayscaleProb": GRAYSCALE_PROB,
    }

    trainer.log_experiment_info(**experiment_info)

    print("\nStarting NoGAN training with BALANCED discriminator...")
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
    print("Multi-task GAN training completed!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Logs saved to: {trainer.log_dir}")
    print(f"Exports saved to: {trainer.export_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
