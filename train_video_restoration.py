"""
Training script for video restoration.

This script demonstrates how to train a model for video restoration tasks
using temporal context from multiple frames.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import VideoRestorationDataset
from src.degradations import (
    DegradationPipeline,
    GaussianNoise,
    JPEGCompression,
    Downscale,
    GaussianBlur,
)
from src.losses import MixedLoss
from src.models import SimpleUNet
from src.training import Trainer


# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment settings
EXPERIMENT_ID = "video_restoration_001"
EXPERIMENT_DIR = "./experiments"

# Dataset settings
DATASET_PATH = "/path/to/video/dataset/"  # UPDATE THIS PATH
VIDEO_SIZE = (128, 128)  # (height, width)
TARGET_FPS = 24.0  # Target frame rate
NUM_FRAMES = 5  # Number of frames per clip (must be odd)
CLIPS_PER_VIDEO = 10  # Number of clips to sample from each video
BATCH_SIZE = 8
NUM_WORKERS = 4

# Model settings
MODEL_CHANNELS = 64  # Base number of channels in U-Net

# Training settings
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Degradation settings
NOISE_SIGMA = 0.05
JPEG_QUALITY = 70
BLUR_KERNEL_SIZE = 3
BLUR_SIGMA = 0.5
DOWNSCALE_FACTOR = 2

# Loss settings
MSE_WEIGHT = 1.0
PERCEPTUAL_WEIGHT = 0.1
PERCEPTUAL_NET = "alex"

# Trainer settings
USE_AMP = True
GRADIENT_CLIP = 1.0
GRADIENT_ACCUMULATION_STEPS = 1
USE_EMA = True
EMA_DECAY = 0.999
USE_COMPILE = False
MULTI_GPU = False

# Logging and checkpointing
LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 500
IMAGE_LOG_INTERVAL = 100

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SETUP
# =============================================================================

def main() -> None:
    """Main training function."""
    print("=" * 80)
    print("Video Restoration Training")
    print("=" * 80)

    # Validate num_frames
    if NUM_FRAMES % 2 == 0:
        raise ValueError(f"NUM_FRAMES must be odd, got {NUM_FRAMES}")

    # Create degradation pipeline
    degradation = DegradationPipeline(
        GaussianNoise(sigma=NOISE_SIGMA),
        GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA),
        Downscale(scale_factor=DOWNSCALE_FACTOR, mode="bilinear"),
        JPEGCompression(quality=JPEG_QUALITY),
    )

    # Create dataset
    print(f"\nLoading dataset from: {DATASET_PATH}")
    dataset = VideoRestorationDataset(
        root_dir=DATASET_PATH,
        target_size=VIDEO_SIZE,
        target_fps=TARGET_FPS,
        num_frames=NUM_FRAMES,
        clips_per_video=CLIPS_PER_VIDEO,
        degradation_fn=degradation,
    )
    print(f"Dataset size: {len(dataset)} clips")

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

    # Create loss function
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
        VideoSize=VIDEO_SIZE,
        TargetFPS=TARGET_FPS,
        NumFrames=NUM_FRAMES,
        ClipsPerVideo=CLIPS_PER_VIDEO,
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
