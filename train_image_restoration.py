"""
Training script for image restoration.

This script demonstrates how to train a model for image restoration tasks
like denoising, super-resolution, artifact removal, etc.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import ImageRestorationDataset
from src.degradations import (
    DegradationPipeline,
    GaussianNoise,
    JPEGCompression,
    Downscale,
    GaussianBlur,
    Grayscale,
    SuperResolutionDegradation,
)
from src.losses import MixedLoss, ColorizationLoss
from src.models import SimpleUNet, HAT, hat_s, hat_m, hat_l, HATSimple, hat_simple_s, hat_simple_m, hat_simple_l
from src.training import Trainer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment settings
EXPERIMENT_ID = "image_restoration_001"
EXPERIMENT_DIR = "./experiments"

# Dataset settings
DATASET_PATH = "/media/bglueck/Data/datasets/laion_high/laion-output"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 32
NUM_WORKERS = 4

# Model settings
MODEL_TYPE = "hat_simple_l"  # "SimpleUNet", "hat_simple_s", "hat_simple_m", "hat_simple_l", or custom "HATSimple"
MODEL_CHANNELS = 64  # Base number of channels in U-Net (for SimpleUNet)
HAT_EMBED_DIM = 96  # Embedding dimension for HAT (for custom HATSimple)
HAT_DEPTHS = [6, 6, 6, 6]  # Depth of each group for HAT (for custom HATSimple)
NUM_FRAMES = 1  # For images, always 1

# Training settings
NUM_EPOCHS = 100
LEARNING_RATE = 5e-5  # Lower LR for more stable training (was 1e-4)
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 1000  # Warmup steps for learning rate
USE_GRADIENT_ACCUMULATION = True  # Enable gradient accumulation for stability
ACCUMULATION_STEPS = 4  # Accumulate gradients over 4 steps (effective batch size = 2*4 = 8)

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
GRADIENT_CLIP = 0.5  # Lower gradient clipping for more stability (was 1.0)
USE_EMA = True  # Use exponential moving average
EMA_DECAY = 0.999  # EMA decay rate
USE_COMPILE = False  # Use torch.compile (requires PyTorch 2.0+)
MULTI_GPU = False  # Use DataParallel for multi-GPU training

# Logging and checkpointing
LOG_INTERVAL = 1000  # Steps between metric logging (more frequent for monitoring)
CHECKPOINT_INTERVAL = 5000  # Steps between checkpoints
IMAGE_LOG_INTERVAL = 1000  # Steps between image logging (IMPORTANT: check images frequently!)

# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SETUP
# =============================================================================

def main() -> None:
    """Main training function."""
    print("=" * 80)
    print("Image Restoration Training")
    print("=" * 80)

    # Create degradation pipeline for super-resolution + denoising + artifact removal
    # For HAT models with upscaling, use SuperResolutionDegradation to actually reduce resolution
    # For other models, use Downscale which downscales and upscales back
    if "hat" in MODEL_TYPE.lower():
        # True super-resolution: input will be smaller than target
        degradation = DegradationPipeline(
            GaussianNoise(sigma=NOISE_SIGMA),
            GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA),
            SuperResolutionDegradation(scale_factor=DOWNSCALE_FACTOR, mode="bilinear"),
            JPEGCompression(quality=JPEG_QUALITY),
        )
    else:
        # Image restoration: input and target same size
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
        upscale_factor=DOWNSCALE_FACTOR,  # For super-resolution
    )
    print(f"Dataset size: {len(dataset)} images")
    print(f"Target (HR) size: {IMAGE_SIZE}")
    print(f"Input (LR) size: ({IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}) (degraded by pipeline)")
    print(f"Model upscale factor: {DOWNSCALE_FACTOR}x")

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
    print(f"\nInitializing model: {MODEL_TYPE}")

    if MODEL_TYPE == "SimpleUNet":
        model = SimpleUNet(
            in_channels=3,
            out_channels=3,
            base_channels=MODEL_CHANNELS,
            num_frames=NUM_FRAMES,
        )
    elif MODEL_TYPE == "hat_simple_s":
        model = hat_simple_s(
            in_channels=3,
            out_channels=3,
            upscale=DOWNSCALE_FACTOR,
            num_frames=NUM_FRAMES,
        )
    elif MODEL_TYPE == "hat_simple_m":
        model = hat_simple_m(
            in_channels=3,
            out_channels=3,
            upscale=DOWNSCALE_FACTOR,
            num_frames=NUM_FRAMES,
        )
    elif MODEL_TYPE == "hat_simple_l":
        model = hat_simple_l(
            in_channels=3,
            out_channels=3,
            upscale=DOWNSCALE_FACTOR,
            num_frames=NUM_FRAMES,
        )
    elif MODEL_TYPE == "HATSimple":
        model = HATSimple(
            in_channels=3,
            out_channels=3,
            embed_dim=HAT_EMBED_DIM,
            depths=HAT_DEPTHS,
            upscale=DOWNSCALE_FACTOR,
            num_frames=NUM_FRAMES,
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

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

    # Create optimizer with more stable settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),  # Slightly higher beta2 for more stability (was 0.999)
        eps=1e-8,
    )

    # Create learning rate scheduler with warmup for stability
    # Warmup helps prevent early training instability
    total_steps = NUM_EPOCHS * len(dataloader)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            # Linear warmup
            return step / WARMUP_STEPS
        else:
            # Cosine decay after warmup
            progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        gradient_accumulation_steps=ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1,
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
    experiment_info = {
        "ModelType": MODEL_TYPE,
        "Dataset": DATASET_PATH,
        "ImageSize": IMAGE_SIZE,
        "BatchSize": BATCH_SIZE,
        "EffectiveBatchSize": BATCH_SIZE * (ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1),
        "NumEpochs": NUM_EPOCHS,
        "LearningRate": LEARNING_RATE,
        "WarmupSteps": WARMUP_STEPS,
        "GradientClip": GRADIENT_CLIP,
        "GradientAccumulation": ACCUMULATION_STEPS if USE_GRADIENT_ACCUMULATION else 1,
        "ModelParams": f"{num_params:,}",
        "TotalSteps": NUM_EPOCHS * len(dataloader),
    }

    if MODEL_TYPE == "SimpleUNet":
        experiment_info["ModelChannels"] = MODEL_CHANNELS
    elif MODEL_TYPE in ["HATSimple", "hat_simple_s", "hat_simple_m", "hat_simple_l"]:
        experiment_info["HAT_EmbedDim"] = HAT_EMBED_DIM if MODEL_TYPE == "HATSimple" else model.embed_dim
        experiment_info["HAT_Depths"] = HAT_DEPTHS if MODEL_TYPE == "HATSimple" else model.num_layers

    trainer.log_experiment_info(**experiment_info)

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
