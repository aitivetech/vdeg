"""Debug script to test colorization setup thoroughly."""

import torch
from src.datasets import ImageRestorationDataset
from src.degradations import Grayscale
from src.losses import ColorizationLossSimple
from src.models import SimpleUNet

print('=== Testing Colorization Setup ===\n')

# Test 1: Dataset
print('1. Testing dataset...')
degradation = Grayscale()
dataset = ImageRestorationDataset(
    root_dir='/media/bglueck/Data/datasets/laion_1024x1024_plus/00/',
    target_size=(256, 256),
    degradation_fn=degradation,
)
print(f'   Dataset size: {len(dataset)}')

# Get a sample
input_img, target_img = dataset[0]
print(f'   Input shape: {input_img.shape}')  # Should be [1, 3, 256, 256]
print(f'   Target shape: {target_img.shape}')  # Should be [3, 256, 256]
print(f'   Input range: [{input_img.min():.3f}, {input_img.max():.3f}]')
print(f'   Target range: [{target_img.min():.3f}, {target_img.max():.3f}]')

# Check if grayscale actually makes it gray
r_eq_g = torch.allclose(input_img[0, 0], input_img[0, 1], atol=1e-6)
g_eq_b = torch.allclose(input_img[0, 1], input_img[0, 2], atol=1e-6)
print(f'   Input is grayscale (R==G==B): {r_eq_g and g_eq_b}')

r_eq_g_target = torch.allclose(target_img[0], target_img[1], atol=1e-2)
print(f'   Target is color (NOT grayscale): {not r_eq_g_target}')

# Test 2: Model
print('\n2. Testing model...')
model = SimpleUNet(in_channels=3, out_channels=3, base_channels=64, num_frames=1)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.train()  # Make sure model is in training mode

# Forward pass
input_batch = input_img.unsqueeze(0).to(device)  # [1, 1, 3, 256, 256]
target_batch = target_img.unsqueeze(0).to(device)  # [1, 3, 256, 256]

print(f'   Input batch shape: {input_batch.shape}')
print(f'   Target batch shape: {target_batch.shape}')

output = model(input_batch)
print(f'   Output shape: {output.shape}')
print(f'   Output range: [{output.min():.3f}, {output.max():.3f}]')
print(f'   Output std dev: {output.std():.4f}')

# Check if output has variance
output_r = output[0, 0].flatten()
output_g = output[0, 1].flatten()
output_b = output[0, 2].flatten()
print(f'   Output R channel std: {output_r.std():.4f}')
print(f'   Output G channel std: {output_g.std():.4f}')
print(f'   Output B channel std: {output_b.std():.4f}')

# Test 3: Loss
print('\n3. Testing loss...')
loss_fn = ColorizationLossSimple(
    perceptual_weight=1.0,
    lab_weight=1.0,
    ab_weight=2.0,
    perceptual_net='alex',
    device=device,
)

# Clamp output to [0, 1] as expected
output_clamped = torch.clamp(output, 0, 1)
loss_dict = loss_fn(output_clamped, target_batch)
print(f'   Total loss: {loss_dict["total"].item():.4f}')
for k, v in loss_dict.items():
    if k != 'total':
        print(f'     - {k}: {v.item():.4f}')

# Test 4: Check for NaN/Inf
print('\n4. Checking for numerical issues...')
print(f'   Output has NaN: {torch.isnan(output).any()}')
print(f'   Output has Inf: {torch.isinf(output).any()}')
print(f'   Loss has NaN: {torch.isnan(loss_dict["total"])}')
print(f'   Loss has Inf: {torch.isinf(loss_dict["total"])}')

# Test 5: Backward pass
print('\n5. Testing backward pass...')
model.zero_grad()
loss_dict['total'].backward()
has_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
total_params = sum(1 for p in model.parameters())
print(f'   Parameters with gradients: {has_grads}/{total_params}')
max_grad = max(p.grad.abs().max() for p in model.parameters() if p.grad is not None)
print(f'   Max gradient magnitude: {max_grad:.6f}')
print(f'   Gradients have NaN: {any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)}')

# Test 6: Optimizer step
print('\n6. Testing optimizer step...')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
old_param = next(model.parameters()).clone()
optimizer.step()
new_param = next(model.parameters())
param_changed = not torch.allclose(old_param, new_param)
print(f'   Parameters changed after step: {param_changed}')
print(f'   Parameter change magnitude: {(new_param - old_param).abs().max():.8f}')

# Test 7: Multiple training steps
print('\n7. Testing multiple training steps...')
losses = []
for i in range(5):
    model.zero_grad()
    output = model(input_batch)
    output_clamped = torch.clamp(output, 0, 1)
    loss_dict = loss_fn(output_clamped, target_batch)
    loss_dict['total'].backward()
    optimizer.step()
    losses.append(loss_dict['total'].item())
    print(f'   Step {i}: loss = {losses[-1]:.4f}')

print(f'\n   Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}')
if losses[-1] < losses[0]:
    print('   ✓ Loss is decreasing!')
else:
    print('   ⚠ Loss is NOT decreasing - potential issue!')

print('\n=== Testing Complete ===')
