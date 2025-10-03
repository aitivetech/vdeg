"""
Example script showing how to resume training from a checkpoint.

This demonstrates loading a checkpoint and continuing training.
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.datasets import ImageRestorationDataset
from src.degradations import DegradationPipeline, GaussianNoise, Downscale
from src.losses import MixedLoss
from src.models import SimpleUNet
from src.training import Trainer


# =============================================================================
# CONFIGURATION
# =============================================================================

# Checkpoint to resume from
CHECKPOINT_PATH = "./experiments/image_restoration_001/checkpoints/best_step_5000.pt"

# Continue with same or modified settings
EXPERIMENT_ID = "image_restoration_001_resumed"
DATASET_PATH = "/media/bglueck/Data/datasets/laion_1024x1024_plus/00/"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
NUM_WORKERS = 4

# Training settings
NUM_ADDITIONAL_EPOCHS = 50  # Number of additional epochs to train
LEARNING_RATE = 1e-5  # Could use lower LR for fine-tuning

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SETUP
# =============================================================================

def main() -> None:
    """Resume training from checkpoint."""
    print("=" * 80)
    print("Resume Training from Checkpoint")
    print("=" * 80)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please update CHECKPOINT_PATH in this script.")
        return

    # Create dataset (same or different than original)
    degradation = DegradationPipeline(
        GaussianNoise(sigma=0.05),
        Downscale(scale_factor=2, mode="bilinear"),
    )

    dataset = ImageRestorationDataset(
        root_dir=DATASET_PATH,
        target_size=IMAGE_SIZE,
        degradation_fn=degradation,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Create model (must match checkpoint architecture)
    model = SimpleUNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        num_frames=1,
    )

    # Create loss
    loss_fn = MixedLoss(
        mse_weight=1.0,
        perceptual_weight=0.1,
        perceptual_net="alex",
        device=DEVICE,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_ADDITIONAL_EPOCHS * len(dataloader),
        eta_min=LEARNING_RATE * 0.01,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE,
        experiment_dir="./experiments",
        experiment_id=EXPERIMENT_ID,
        use_amp=True,
        gradient_clip=1.0,
        use_ema=True,
        use_compile=False,
        multi_gpu=False,
    )

    trainer.scheduler = scheduler

    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)
    print(f"Resumed from epoch {trainer.current_epoch}, step {trainer.global_step}")

    # Log info
    trainer.log_experiment_info(
        ResumedFrom=str(checkpoint_path),
        Dataset=DATASET_PATH,
        ImageSize=IMAGE_SIZE,
        BatchSize=BATCH_SIZE,
        AdditionalEpochs=NUM_ADDITIONAL_EPOCHS,
        LearningRate=LEARNING_RATE,
    )

    # Continue training
    print("\nResuming training...")
    print("=" * 80)

    start_epoch = trainer.current_epoch + 1
    for epoch in range(start_epoch, start_epoch + NUM_ADDITIONAL_EPOCHS):
        epoch_metrics = trainer.train_epoch(dataloader, epoch)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {epoch_metrics['loss']:.6f}")
        print(f"  PSNR: {epoch_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {epoch_metrics['ssim']:.4f}")
        print("-" * 80)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
