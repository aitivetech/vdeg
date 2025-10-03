"""Detailed debugging to see exactly what's happening."""

import torch
import torch.nn.functional as F
from src.datasets import ImageRestorationDataset
from src.degradations import Grayscale
from src.losses import ColorizationLossSimple
from src.models import SimpleUNet
# Skip matplotlib for now

print('=== Detailed Colorization Debug ===\n')

# Setup
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}\n')

# Load dataset
print('1. Loading dataset...')
degradation = Grayscale()
dataset = ImageRestorationDataset(
    root_dir='/media/bglueck/Data/datasets/laion_1024x1024_plus/00/',
    target_size=(256, 256),
    degradation_fn=degradation,
)
print(f'   Dataset size: {len(dataset)}')

# Get one sample
input_img, target_img = dataset[0]
print(f'   Input shape: {input_img.shape}')  # [1, 3, 256, 256]
print(f'   Target shape: {target_img.shape}')  # [3, 256, 256]

# Check grayscale
print(f'   Input R channel mean: {input_img[0, 0].mean():.4f}')
print(f'   Input G channel mean: {input_img[0, 1].mean():.4f}')
print(f'   Input B channel mean: {input_img[0, 2].mean():.4f}')
print(f'   Input is grayscale: {torch.allclose(input_img[0, 0], input_img[0, 1]) and torch.allclose(input_img[0, 1], input_img[0, 2])}')

print(f'   Target R channel mean: {target_img[0].mean():.4f}')
print(f'   Target G channel mean: {target_img[1].mean():.4f}')
print(f'   Target B channel mean: {target_img[2].mean():.4f}')

# Skip image saving

# Create model
print('\n2. Creating model...')
model = SimpleUNet(in_channels=3, out_channels=3, base_channels=64, num_frames=1)
model = model.to(device)
model.train()

num_params = sum(p.numel() for p in model.parameters())
print(f'   Parameters: {num_params:,}')

# Create loss
print('\n3. Creating loss function...')
loss_fn = ColorizationLossSimple(
    perceptual_weight=1.0,
    lab_weight=1.0,
    ab_weight=2.0,
    perceptual_net='alex',
    device=device,
)

# Create optimizer
print('\n4. Creating optimizer...')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Prepare batch
input_batch = input_img.unsqueeze(0).to(device)
target_batch = target_img.unsqueeze(0).to(device)

print(f'   Input batch shape: {input_batch.shape}')
print(f'   Target batch shape: {target_batch.shape}')

# Training steps
print('\n5. Running training steps...')
print('   (Watching for: decreasing loss, output diversity, gradient flow)\n')

for step in range(10):
    model.zero_grad()

    # Forward pass
    output = model(input_batch)

    # Check output stats
    output_mean = output.mean()
    output_std = output.std()
    output_min = output.min()
    output_max = output.max()

    # Check if output channels are different (not stuck)
    r_mean = output[0, 0].mean()
    g_mean = output[0, 1].mean()
    b_mean = output[0, 2].mean()

    # Clamp to [0, 1] for loss
    output_clamped = torch.clamp(output, 0, 1)

    # Compute loss
    loss_dict = loss_fn(output_clamped, target_batch)
    loss = loss_dict['total']

    # Backward
    loss.backward()

    # Check gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    max_grad = max(grad_norms)
    mean_grad = sum(grad_norms) / len(grad_norms)

    # Optimizer step
    optimizer.step()

    print(f'   Step {step}:')
    print(f'     Loss: {loss.item():.6f} (perceptual: {loss_dict["perceptual"].item():.4f}, lab: {loss_dict["lab"].item():.4f}, ab: {loss_dict["ab"].item():.4f})')
    print(f'     Output: mean={output_mean:.4f}, std={output_std:.4f}, range=[{output_min:.4f}, {output_max:.4f}]')
    print(f'     RGB means: R={r_mean:.4f}, G={g_mean:.4f}, B={b_mean:.4f}')
    print(f'     Gradients: max={max_grad:.4f}, mean={mean_grad:.4f}')

    # Check for issues
    if torch.isnan(loss):
        print('     ⚠️  Loss is NaN!')
        break
    if output_std < 0.01:
        print('     ⚠️  Output has very low variance!')
    if abs(r_mean - g_mean) < 0.01 and abs(g_mean - b_mean) < 0.01:
        print('     ⚠️  Output channels are too similar!')

    # Skip image saving

print('\n=== Debug Complete ===')
print('Check debug_*.png files to see what the model is producing')
