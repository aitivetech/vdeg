"""
Unified multi-task training script for HAT model.

Trains a single HAT model on multiple tasks simultaneously:
- Super-resolution (upscaling low-res images)
- Artifact removal (denoising, deblurring, decompression)
- Colorization (grayscale to color)

This unified approach:
1. Eliminates code duplication across training scripts
2. Enables multi-task learning (shared features across tasks)
3. Produces a single versatile model
4. Supports flexible task weighting and curriculum learning
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
from src.losses import MultiTaskLoss, AdaptiveMultiTaskLoss
from src.models import HATSimple, hat_simple_s, hat_simple_m, hat_simple_l
from src.training import Trainer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment settings
EXPERIMENT_ID = "multitask_hat_001"
EXPERIMENT_DIR = "./experiments"

# Dataset settings
DATASET_PATH = "/media/bglueck/Data/datasets/laion_high/laion-output"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_WORKERS = 4

# Model settings
MODEL_TYPE = "hat_simple_l"  # "hat_simple_s", "hat_simple_m", "hat_simple_l"
UPSCALE_FACTOR = 4  # Super-resolution scale factor
NUM_FRAMES = 1  # For images, always 1

# Training settings
NUM_EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 1000
USE_GRADIENT_ACCUMULATION = True
ACCUMULATION_STEPS = 4

# Task settings - Enable/disable specific tasks
ENABLE_SUPER_RESOLUTION = True
ENABLE_ARTIFACT_REMOVAL = True
ENABLE_COLORIZATION = True

# Degradation probabilities (for multi-task training)
# Set to 1.0 to always apply, 0.0 to disable
DOWNSCALE_PROB = 1.0  # Super-resolution
NOISE_PROB = 0.7      # Denoising
BLUR_PROB = 0.5       # Deblurring
COMPRESSION_PROB = 0.8  # Artifact removal
GRAYSCALE_PROB = 1.0  # Colorization

# Degradation parameters
NOISE_SIGMA = 0.05
JPEG_QUALITY = 70
BLUR_KERNEL_SIZE = 3
BLUR_SIGMA = 0.5

# Loss settings
USE_ADAPTIVE_LOSS = False  # Use AdaptiveMultiTaskLoss for automatic task balancing
RGB_WEIGHT = 1.0          # Weight for RGB MSE (super-resolution + artifacts)
PERCEPTUAL_WEIGHT = 0.5   # Weight for perceptual loss (semantic coherence)
AB_WEIGHT = 2.0           # Weight for AB chrominance (colorization)
PERCEPTUAL_NET = "vgg"    # Network for perceptual loss ('alex', 'vgg')

# Trainer settings
USE_AMP = True
GRADIENT_CLIP = 0.5
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
    """Main multi-task training function."""
    print("=" * 80)
    print("MULTI-TASK HAT TRAINING")
    print("=" * 80)
    print(f"\nEnabled tasks:")
    print(f"  • Super-resolution: {ENABLE_SUPER_RESOLUTION} ({UPSCALE_FACTOR}x)")
    print(f"  • Artifact removal: {ENABLE_ARTIFACT_REMOVAL}")
    print(f"  • Colorization: {ENABLE_COLORIZATION}")
    print(f"\nDegradation probabilities:")
    print(f"  • Downscale: {DOWNSCALE_PROB:.1%}")
    print(f"  • Noise: {NOISE_PROB:.1%}")
    print(f"  • Blur: {BLUR_PROB:.1%}")
    print(f"  • Compression: {COMPRESSION_PROB:.1%}")
    print(f"  • Grayscale: {GRAYSCALE_PROB:.1%}")
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
    )
    print(f"Dataset size: {len(dataset)} images")
    print(f"Target (HR) size: {IMAGE_SIZE}")
    print(f"Model upscale factor: {UPSCALE_FACTOR}x")

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

    if MODEL_TYPE == "hat_simple_s":
        model = hat_simple_s(
            in_channels=3,
            out_channels=3,
            upscale=UPSCALE_FACTOR,
            num_frames=NUM_FRAMES,
        )
    elif MODEL_TYPE == "hat_simple_m":
        model = hat_simple_m(
            in_channels=3,
            out_channels=3,
            upscale=UPSCALE_FACTOR,
            num_frames=NUM_FRAMES,
        )
    elif MODEL_TYPE == "hat_simple_l":
        model = hat_simple_l(
            in_channels=3,
            out_channels=3,
            upscale=UPSCALE_FACTOR,
            num_frames=NUM_FRAMES,
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    # Create multi-task loss function
    if USE_ADAPTIVE_LOSS:
        print(f"\nUsing AdaptiveMultiTaskLoss (automatic task balancing)")
        loss_fn = AdaptiveMultiTaskLoss(
            rgb_weight=RGB_WEIGHT,
            perceptual_weight=PERCEPTUAL_WEIGHT,
            ab_weight=AB_WEIGHT,
            perceptual_net=PERCEPTUAL_NET,
            device=DEVICE,
            enable_colorization=ENABLE_COLORIZATION,
            enable_super_resolution=ENABLE_SUPER_RESOLUTION or ENABLE_ARTIFACT_REMOVAL,
        )
    else:
        print(f"\nUsing MultiTaskLoss (fixed weights)")
        print(f"  RGB weight: {RGB_WEIGHT}")
        print(f"  Perceptual weight: {PERCEPTUAL_WEIGHT}")
        print(f"  AB weight: {AB_WEIGHT}")
        loss_fn = MultiTaskLoss(
            rgb_weight=RGB_WEIGHT,
            perceptual_weight=PERCEPTUAL_WEIGHT,
            ab_weight=AB_WEIGHT,
            perceptual_net=PERCEPTUAL_NET,
            device=DEVICE,
            enable_colorization=ENABLE_COLORIZATION,
            enable_super_resolution=ENABLE_SUPER_RESOLUTION or ENABLE_ARTIFACT_REMOVAL,
        )

    # Create optimizer
    # For adaptive loss, include loss parameters in optimizer
    if USE_ADAPTIVE_LOSS:
        optimizer_params = [
            {'params': model.parameters()},
            {'params': loss_fn.parameters(), 'lr': LEARNING_RATE * 0.1},  # Lower LR for loss weights
        ]
    else:
        optimizer_params = model.parameters()

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    # Create learning rate scheduler with warmup
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
        "UpscaleFactor": UPSCALE_FACTOR,
        "ModelParams": f"{num_params:,}",
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
        # Loss configuration
        "AdaptiveLoss": USE_ADAPTIVE_LOSS,
        "RGBWeight": RGB_WEIGHT,
        "PerceptualWeight": PERCEPTUAL_WEIGHT,
        "ABWeight": AB_WEIGHT,
    }

    trainer.log_experiment_info(**experiment_info)

    # Training loop
    print("\nStarting multi-task training...")
    print("=" * 80)

    for epoch in range(NUM_EPOCHS):
        epoch_metrics = trainer.train_epoch(dataloader, epoch)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {epoch_metrics['loss']:.6f}")
        print(f"  PSNR: {epoch_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {epoch_metrics['ssim']:.4f}")

        # Log adaptive loss weights if using adaptive loss
        if USE_ADAPTIVE_LOSS and 'weight_rgb' in epoch_metrics:
            print(f"  Adaptive weights:")
            print(f"    RGB: {epoch_metrics.get('weight_rgb', 0):.3f}")
            print(f"    Perceptual: {epoch_metrics.get('weight_perceptual', 0):.3f}")
            print(f"    AB: {epoch_metrics.get('weight_ab', 0):.3f}")

        print("-" * 80)

    print("\nMulti-task training completed!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Logs saved to: {trainer.log_dir}")
    print(f"Exports saved to: {trainer.export_dir}")


if __name__ == "__main__":
    main()
