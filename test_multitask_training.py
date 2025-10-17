"""
Quick test to verify multi-task training pipeline works.

This is a minimal test with tiny settings to ensure everything runs.
"""

import torch
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
from src.losses import MultiTaskLoss
from src.models import hat_simple_s
from src.training import Trainer

# Minimal test settings
DATASET_PATH = "/media/bglueck/Data/datasets/laion_high/laion-output"
IMAGE_SIZE = (512, 512)  # Small for fast testing
BATCH_SIZE = 2
NUM_WORKERS = 0  # Single-threaded for testing
NUM_EPOCHS = 100  # Just one epoch
LIMIT = 100000  # Only 10 images

UPSCALE_FACTOR = 4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("MULTI-TASK TRAINING TEST")
print("=" * 80)

# Create degradations
noise_fn = GaussianNoise(sigma=0.05)
blur_fn = GaussianBlur(kernel_size=3, sigma=0.5)
compression_fn = JPEGCompression(quality=70)
grayscale_fn = Grayscale()
downscale_fn = SuperResolutionDegradation(scale_factor=UPSCALE_FACTOR, mode="bilinear")

degradation = MultiTaskDegradation(
    downscale_fn=downscale_fn,
    noise_fn=noise_fn,
    blur_fn=blur_fn,
    compression_fn=compression_fn,
    grayscale_fn=grayscale_fn,
    downscale_prob=1.0,
    noise_prob=0.7,
    blur_prob=0.5,
    compression_prob=0.8,
    grayscale_prob=1.0,
)

print("\n✓ Degradations created")

# Create dataset
dataset = ImageRestorationDataset(
    root_dir=DATASET_PATH,
    target_size=IMAGE_SIZE,
    degradation_fn=degradation,
    upscale_factor=UPSCALE_FACTOR,
    limit=LIMIT,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

print(f"✓ Dataset created: {len(dataset)} images, {len(dataloader)} batches")

# Create model
model = hat_simple_s(
    in_channels=3,
    out_channels=3,
    upscale=UPSCALE_FACTOR,
    num_frames=1,
)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created: {num_params:,} parameters")

# Create loss
loss_fn = MultiTaskLoss(
    rgb_weight=1.0,
    perceptual_weight=0.5,
    ab_weight=2.0,
    perceptual_net="vgg",
    device=DEVICE,
    enable_colorization=True,
    enable_super_resolution=True,
)

print(f"✓ Loss function created")

# Create optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
)

print(f"✓ Optimizer created")

# Create trainer
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=DEVICE,
    experiment_dir="./experiments",
    experiment_id="multitask_test",
    use_amp=True,
    gradient_clip=0.5,
    use_ema=True,
    ema_decay=0.999,
    log_interval=5,
    checkpoint_interval=100,  # Don't checkpoint in test
    image_log_interval=5,
)

print(f"✓ Trainer created")

# Test one epoch
print("\nRunning test epoch...")
print("=" * 80)

try:
    epoch_metrics = trainer.train_epoch(dataloader, epoch=0)

    print("\n" + "=" * 80)
    print("✅ TEST PASSED!")
    print("=" * 80)
    print(f"\nEpoch metrics:")
    print(f"  Loss: {epoch_metrics['loss']:.6f}")
    print(f"  PSNR: {epoch_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {epoch_metrics['ssim']:.4f}")
    print(f"\nMulti-task training pipeline is working correctly!")
    print(f"\nTo start full training, run:")
    print(f"  uv run python train_multitask.py")

except Exception as e:
    print("\n" + "=" * 80)
    print("❌ TEST FAILED!")
    print("=" * 80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    raise
