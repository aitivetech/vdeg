"""
Training script for image restoration.

This script demonstrates how to train a model for image restoration tasks
like denoising, super-resolution, artifact removal, etc.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import ImageRestorationDataset
from src.degradations import (
    DegradationPipeline,
    GaussianNoise,
    JPEGCompression,
    Downscale,
    GaussianBlur, Grayscale,
)
from src.losses import MixedLoss, ColorizationLoss
from src.models import SimpleUNet
from src.training import Trainer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment settings
EXPERIMENT_ID = "image_restoration_001"
EXPERIMENT_DIR = "./experiments"

# Dataset settings
DATASET_PATH = "/media/bglueck/Data/datasets/laion_1024x1024_plus/images_512"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_WORKERS = 4

# Model settings
MODEL_CHANNELS = 64  # Base number of channels in U-Net
NUM_FRAMES = 1  # For images, always 1

# Training settings
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Degradation settings (for creating training data)
# Combine multiple degradations for realistic scenarios
NOISE_SIGMA = 0.05  # Gaussian noise standard deviation
JPEG_QUALITY = 70  # JPEG compression quality
BLUR_KERNEL_SIZE = 3  # Gaussian blur kernel size
BLUR_SIGMA = 0.5  # Gaussian blur sigma
DOWNSCALE_FACTOR = 4  # Super-resolution scale factor (1 = no downscaling)

# Loss settings
MSE_WEIGHT = 1.0  # Weight for MSE loss
PERCEPTUAL_WEIGHT = 0.1  # Weight for perceptual loss
PERCEPTUAL_NET = "alex"  # Network for perceptual loss ('alex', 'vgg', 'squeeze')
LAB_WEIGHT = 1.0

# Trainer settings
USE_AMP = True  # Use automatic mixed precision
GRADIENT_CLIP = 1.0  # Gradient clipping value (None to disable)
GRADIENT_ACCUMULATION_STEPS = 1  # Gradient accumulation steps
USE_EMA = True  # Use exponential moving average
EMA_DECAY = 0.999  # EMA decay rate
USE_COMPILE = False  # Use torch.compile (requires PyTorch 2.0+)
MULTI_GPU = False  # Use DataParallel for multi-GPU training

# Logging and checkpointing
LOG_INTERVAL = 1000  # Steps between metric logging (more frequent for monitoring)
CHECKPOINT_INTERVAL = 5000  # Steps between checkpoints
IMAGE_LOG_INTERVAL = 1000  # Steps between image logging (IMPORTANT: check images frequently!)

# Device
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SETUP
# =============================================================================

def main() -> None:
    """Main training function."""
    print("=" * 80)
    print("Image Restoration Training")
    print("=" * 80)

    # Create degradation pipeline for super-resolution + denoising + artifact removal
    degradation = DegradationPipeline(
        GaussianNoise(sigma=NOISE_SIGMA),
        GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA),
        Downscale(scale_factor=DOWNSCALE_FACTOR, mode="bilinear"),
        JPEGCompression(quality=JPEG_QUALITY),
    )

    # Create dataset
    print(f"\nLoading dataset from: {DATASET_PATH}")
    dataset = ImageRestorationDataset(
        root_dir=DATASET_PATH,
        target_size=IMAGE_SIZE,
        degradation_fn=degradation,
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

    # Create model
    print(f"\nInitializing model...")
    model = SimpleUNet(
        in_channels=3,
        out_channels=3,
        base_channels=MODEL_CHANNELS,
        num_frames=NUM_FRAMES,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    # Create loss function for super-resolution + denoising + artifact removal
    loss_fn = MixedLoss(
        mse_weight=MSE_WEIGHT,
        perceptual_weight=PERCEPTUAL_WEIGHT,
        perceptual_net=PERCEPTUAL_NET,
        device=DEVICE,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS * len(dataloader),
        eta_min=LEARNING_RATE * 0.01,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE,
        experiment_dir=EXPERIMENT_DIR,
        experiment_id=EXPERIMENT_ID,
        use_amp=USE_AMP,
        gradient_clip=GRADIENT_CLIP,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        use_ema=USE_EMA,
        ema_decay=EMA_DECAY,
        use_compile=USE_COMPILE,
        multi_gpu=MULTI_GPU,
        log_interval=LOG_INTERVAL,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        image_log_interval=IMAGE_LOG_INTERVAL,
    )

    # Set scheduler
    trainer.scheduler = scheduler

    # Log experiment info
    trainer.log_experiment_info(
        Dataset=DATASET_PATH,
        ImageSize=IMAGE_SIZE,
        BatchSize=BATCH_SIZE,
        ModelChannels=MODEL_CHANNELS,
        NumEpochs=NUM_EPOCHS,
        LearningRate=LEARNING_RATE,
        ModelParams=f"{num_params:,}",
        TotalSteps=NUM_EPOCHS * len(dataloader),
    )

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    for epoch in range(NUM_EPOCHS):
        epoch_metrics = trainer.train_epoch(dataloader, epoch)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {epoch_metrics['loss']:.6f}")
        print(f"  PSNR: {epoch_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {epoch_metrics['ssim']:.4f}")
        print("-" * 80)

    print("\nTraining completed!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Logs saved to: {trainer.log_dir}")
    print(f"Exports saved to: {trainer.export_dir}")


if __name__ == "__main__":
    main()
