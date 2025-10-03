"""
Quick test script to verify the training pipeline works.

This runs a few training steps to validate the implementation.
"""

import torch
from torch.utils.data import DataLoader

from src.datasets import ImageRestorationDataset
from src.degradations import DegradationPipeline, GaussianNoise, Downscale
from src.losses import MixedLoss
from src.models import SimpleUNet
from src.training import Trainer


def main() -> None:
    """Test the training pipeline."""
    print("Testing training pipeline...")

    # Configuration
    DATASET_PATH = "/media/bglueck/Data/datasets/laion_1024x1024_plus/00/"
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")

    # Simple degradation for testing
    degradation = DegradationPipeline(
        GaussianNoise(sigma=0.05),
        Downscale(scale_factor=2, mode="bilinear"),
    )

    # Create dataset
    print("Loading dataset...")
    dataset = ImageRestorationDataset(
        root_dir=DATASET_PATH,
        target_size=IMAGE_SIZE,
        degradation_fn=degradation,
    )
    print(f"Found {len(dataset)} images")

    # Create dataloader with subset for testing
    from torch.utils.data import Subset
    test_dataset = Subset(dataset, range(min(20, len(dataset))))
    dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # No multiprocessing for testing
        drop_last=True,
    )

    # Create model
    print("Creating model...")
    model = SimpleUNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,  # Smaller for testing
        num_frames=1,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # Create loss
    print("Creating loss function...")
    loss_fn = MixedLoss(
        mse_weight=1.0,
        perceptual_weight=0.1,
        perceptual_net="alex",
        device=DEVICE,
    )

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE,
        experiment_dir="./test_experiments",
        experiment_id="test_run",
        use_amp=True,
        gradient_clip=1.0,
        use_ema=True,
        use_compile=False,
        multi_gpu=False,
        log_interval=1,
        checkpoint_interval=10,
        image_log_interval=5,
    )

    # Test single batch
    print("\nTesting single batch...")
    model.to(DEVICE)
    model.train()

    inputs, targets = next(iter(dataloader))
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")

    inputs = inputs.to(DEVICE)
    targets = targets.to(DEVICE)

    # Forward pass
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(inputs)
        print(f"Output shape: {outputs.shape}")

        loss_output = loss_fn(outputs, targets)
        if isinstance(loss_output, dict):
            print(f"Loss components: {loss_output}")
            loss = loss_output["total"]
        else:
            loss = loss_output
        print(f"Loss value: {loss.item():.6f}")

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Backward pass successful!")

    # Test training loop (just 1 epoch)
    print("\nRunning mini training loop (1 epoch)...")
    trainer.log_experiment_info(
        Dataset=DATASET_PATH,
        ImageSize=IMAGE_SIZE,
        BatchSize=BATCH_SIZE,
        TestMode=True,
    )

    metrics = trainer.train_epoch(dataloader, epoch=0)
    print(f"\nEpoch metrics:")
    print(f"  Loss: {metrics['loss']:.6f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")

    print("\n✓ All tests passed!")
    print(f"Check outputs in: ./test_experiments/test_run/")


if __name__ == "__main__":
    main()
