"""Quick test script for HAT model."""

import torch
from src.models import HAT, hat_s, hat_m, hat_l

# Debug mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU for debugging

def test_model(model, name, input_size=(1, 1, 3, 64, 64)):
    """Test model forward pass."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(input_size)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        y = model(x)

    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print("✓ Forward pass successful!")

if __name__ == "__main__":
    print("Testing HAT model variants...")

    # Test HAT-S (small)
    model_s = hat_s(
        in_channels=3,
        out_channels=3,
        upscale=4,
        img_size=64,
        num_frames=1
    )
    test_model(model_s, "HAT-S (Small)")

    # Test HAT-M (medium)
    model_m = hat_m(
        in_channels=3,
        out_channels=3,
        upscale=4,
        img_size=64,
        num_frames=1
    )
    test_model(model_m, "HAT-M (Medium)")

    # Test custom HAT
    model_custom = HAT(
        in_channels=3,
        out_channels=3,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=7,
        upscale=4,
        img_size=64,
        num_frames=1
    )
    test_model(model_custom, "HAT (Custom)")

    print(f"\n{'='*60}")
    print("All tests passed! ✓")
    print(f"{'='*60}\n")
