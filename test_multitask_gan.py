"""
Quick test to verify multi-task GAN training pipeline works.

Tests the balanced NoGAN training with a tiny configuration.
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
from src.losses import MultiTaskLoss, GANLoss
from src.models import PatchDiscriminator, hat_simple_s
from src.training import BalancedMultiTaskGANTrainer

# Minimal test settings
DATASET_PATH = "/media/bglueck/Data/datasets/laion_high/laion-output"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 2
NUM_WORKERS = 0
LIMIT = 10

UPSCALE_FACTOR = 2
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("MULTI-TASK GAN TRAINING TEST")
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

print("✓ Degradations created")

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

# Create generator
generator = hat_simple_s(
    in_channels=3,
    out_channels=3,
    upscale=UPSCALE_FACTOR,
    num_frames=1,
)
print(f"✓ Generator created: {sum(p.numel() for p in generator.parameters()):,} parameters")

# Create discriminator
discriminator = PatchDiscriminator(
    input_channels=6,
    num_filters=64,
    num_layers=3,
    use_spectral_norm=True,
)
print(f"✓ Discriminator created: {sum(p.numel() for p in discriminator.parameters()):,} parameters")

# Create content loss
content_loss = MultiTaskLoss(
    rgb_weight=1.0,
    perceptual_weight=0.5,
    ab_weight=2.0,
    perceptual_net="vgg",
    device=DEVICE,
    enable_colorization=True,
    enable_super_resolution=True,
)
print("✓ Content loss created")

# Create optimizers
gen_optimizer = torch.optim.AdamW(
    generator.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
)

disc_optimizer = torch.optim.AdamW(
    discriminator.parameters(),
    lr=2e-5,  # 5x slower
    weight_decay=1e-5,
)
print("✓ Optimizers created (balanced LR)")

# Create trainer
trainer = BalancedMultiTaskGANTrainer(
    generator=generator,
    discriminator=discriminator,
    content_loss=content_loss,
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer,
    device=DEVICE,
    experiment_dir="./experiments",
    experiment_id="multitask_gan_test",
    gan_loss_type="lsgan",
    gan_weight=0.5,
    content_weight=1.0,
    disc_update_frequency=1,
    use_adaptive_disc=True,
    disc_advantage_threshold=0.7,
    use_amp=True,
    gradient_clip=1.0,
    use_ema=True,
    ema_decay=0.999,
    log_interval=5,
    checkpoint_interval=1000,
    image_log_interval=5,
)
print("✓ Balanced GAN trainer created")

# Test all three phases
print("\n" + "=" * 80)
print("Testing NoGAN phases...")
print("=" * 80)

try:
    # Phase 1: Pretrain
    print("\n[PHASE 1: PRETRAIN]")
    trainer.set_phase("pretrain")
    pretrain_metrics = trainer.train_epoch(dataloader, epoch=0)
    print(f"✓ Pretrain phase passed")
    print(f"  Loss: {pretrain_metrics['loss']:.4f}, PSNR: {pretrain_metrics['psnr']:.2f}")

    # Phase 2: Critic
    print("\n[PHASE 2: CRITIC]")
    trainer.set_phase("critic")
    critic_metrics = trainer.train_epoch(dataloader, epoch=0)
    print(f"✓ Critic phase passed")
    print(f"  Disc Loss: {critic_metrics['disc_loss']:.4f}")
    print(f"  Real: {critic_metrics['disc_real']:.4f}, Fake: {critic_metrics['disc_fake']:.4f}")

    # Phase 3: GAN
    print("\n[PHASE 3: BALANCED GAN]")
    trainer.set_phase("gan")
    gan_metrics = trainer.train_epoch(dataloader, epoch=0)
    print(f"✓ Balanced GAN phase passed")
    print(f"  Gen Loss: {gan_metrics['gen_loss']:.4f}, Disc Loss: {gan_metrics['disc_loss']:.4f}")
    print(f"  Advantage: {gan_metrics['disc_advantage']:.4f}")
    print(f"  Disc Updates: {gan_metrics['disc_updates']}, Skips: {gan_metrics['disc_skips']}")
    print(f"  PSNR: {gan_metrics['psnr']:.2f}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nBalanced multi-task GAN training pipeline is working correctly!")
    print("\n🚀 Ready for full training:")
    print("  • Standard multi-task: uv run python train_multitask.py")
    print("  • GAN multi-task: uv run python train_multitask_gan.py")

except Exception as e:
    print("\n" + "=" * 80)
    print("❌ TEST FAILED!")
    print("=" * 80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    raise
