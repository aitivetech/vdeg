# The REAL Problem and Fix for Colorization

## Root Cause Found

**The LAB color space values were NOT normalized**, causing massive loss imbalance:

- Perceptual loss: ~0.9 (reasonable)
- LAB loss: ~680.0 (100-1000x too large!)
- AB loss: ~791.0 (100-1000x too large!)

**Total loss: ~2263** (should be ~1-5)

This caused:
1. LAB loss completely dominates the gradient
2. Model ignores perceptual guidance (semantic understanding)
3. Training becomes unstable
4. Model gets stuck in poor local minima (green/black output)

## Why LAB Values Are Huge

LAB color space has different ranges than RGB:
- **L (Lightness)**: [0, 100] (not [0, 1])
- **A (green-red)**: [-128, 128] (not [0, 1])
- **B (blue-yellow)**: [-128, 128] (not [0, 1])

When you compute MSE on these values directly:
```python
# WRONG - unnormalized
lab_loss = F.mse_loss(pred_lab, target_lab)  # Can be 100-10000!
```

The loss is 100-10000x larger than RGB MSE or perceptual loss!

## The Fix

**Normalize LAB values to [0, 1] range before computing MSE:**

```python
# Normalize LAB to [0, 1]
pred_lab_norm = pred_lab.clone()
pred_lab_norm[:, 0:1] = pred_lab[:, 0:1] / 100.0          # L: [0,100] -> [0,1]
pred_lab_norm[:, 1:3] = (pred_lab[:, 1:3] + 128.0) / 256.0  # AB: [-128,128] -> [0,1]

# Same for target
target_lab_norm = target_lab.clone()
target_lab_norm[:, 0:1] = target_lab[:, 0:1] / 100.0
target_lab_norm[:, 1:3] = (target_lab[:, 1:3] + 128.0) / 256.0

# Now compute loss on normalized values
lab_loss = F.mse_loss(pred_lab_norm, target_lab_norm)  # Now ~0.1-1.0 range!
```

## Results After Fix

```
BEFORE (unnormalized):
Total loss: 2263.0
  - perceptual: 0.97
  - lab: 679.88
  - ab: 791.08

AFTER (normalized):
Total loss: 1.10
  - perceptual: 0.87
  - lab: 0.12
  - ab: 0.11
```

**Loss is now balanced!** All components are in the same order of magnitude.

## Training Results

With normalized LAB loss, the model trains stably:

```
Step 0: loss = 0.8785
Step 1: loss = 0.7528
Step 2: loss = 0.6765
Step 3: loss = 0.6212
Step 4: loss = 0.5785

✓ Loss is decreasing!
```

## What This Means For You

**The fix is already applied** to `ColorizationLossSimple` in `src/losses/colorization_v2.py`.

Your `train_image_restoration.py` is already configured to use it.

**Just run:**
```bash
uv run python train_image_restoration.py
```

**You should now see:**
- Loss starting around 1-5 (not 1000+)
- Loss steadily decreasing
- PSNR gradually increasing
- Varied colors in comparison images (not green/black)

## Why This Was Hard to Find

1. **No error messages** - mathematically, the code was correct
2. **Loss was decreasing** - just very slowly because gradients were imbalanced
3. **Different color spaces** - LAB is not as familiar as RGB
4. **Implicit assumption** - most people assume color space conversions normalize

## Lessons Learned

When working with different color spaces:
1. **Always normalize** to comparable ranges
2. **Check loss magnitudes** - all components should be similar order of magnitude
3. **Test with simple cases** first (as we did with debug_colorization.py)

## Bottom Line

**Unnormalized LAB values caused 100-1000x loss imbalance.**
**Now fixed. Training should work correctly.**

Run `uv run python train_image_restoration.py` and you should see proper convergence.
