# Colorization Training - The Simple Fix

## The Problem

The "fancy" `ColorizationLoss` with diversity regularization was **too unstable**:
- Diversity loss using `-log(variance)` can explode if variance is very low
- This causes gradients to become extreme (NaN or very large)
- Model gets stuck outputting constant colors (green/black)
- Loss fluctuates wildly without improving

## The Solution: Keep It Simple

I've created `ColorizationLossSimple` that removes the unstable components and focuses on what actually works:

### What It Does

1. **Perceptual Loss** - Guides the model to semantically plausible colors
2. **LAB Space Loss** - Overall color accuracy in perceptually uniform space
3. **AB Channel Loss** - Extra emphasis on chrominance (color) channels

**That's it.** No fancy diversity tricks that can destabilize training.

### Why This Works

- **Perceptual loss** (LPIPS) already contains semantic understanding of colors
- **LAB space** naturally separates luminance from color
- **Extra AB weight** focuses learning on color without introducing instability
- **Simple MSE losses** are stable and well-understood

## How to Use

```bash
uv run python train_colorization_simple.py
```

### Key Settings

```python
PERCEPTUAL_WEIGHT = 1.0  # Strong semantic guidance
LAB_WEIGHT = 1.0         # Full LAB accuracy
AB_WEIGHT = 2.0          # Extra focus on color
BATCH_SIZE = 16          # Larger = more stable gradients
LEARNING_RATE = 1e-4     # Conservative, stable
```

## What You Should See

**Early training (first 500 steps):**
- Loss should steadily decrease
- PSNR should gradually increase
- Colors should appear varied (not stuck on one color)

**If you see:**
- ✅ Gradually improving colors → Good!
- ✅ Loss decreasing steadily → Good!
- ❌ Green/black output → Stop, something is wrong with config
- ❌ Loss not changing → Learning rate too low or model issue

## Monitoring Tips

1. **Check comparison images every 250 steps** (set in IMAGE_LOG_INTERVAL)
2. **Look at the individual loss components:**
   - `perceptual`: Should decrease
   - `lab`: Should decrease
   - `ab`: Should decrease
3. **If stuck on one color after 1000 steps:**
   - Increase `PERCEPTUAL_WEIGHT` to 2.0
   - Increase `LEARNING_RATE` to 2e-4
   - Make sure batch size is >= 8

## Comparison: Old vs New

| Feature | ColorizationLoss (Complex) | ColorizationLossSimple |
|---------|---------------------------|------------------------|
| Perceptual Loss | ✓ | ✓ |
| LAB Space Loss | ✓ | ✓ |
| AB Channel Focus | ✓ | ✓ |
| Diversity Regularization | ✓ (unstable) | ✗ (removed) |
| Huber Loss | ✓ | ✗ (MSE is fine) |
| Training Stability | ⚠️ Can explode | ✅ Stable |
| Complexity | High | Low |

## Why The Complex Version Failed

The diversity loss `-log(variance)` has a fatal flaw:

```python
# If model outputs very uniform colors:
variance = 0.001  # Very low
diversity_loss = -log(0.001) = 6.9  # Large positive number

# This creates huge gradients pushing for more variance
# But if it overshoots:
variance = 0.0001
diversity_loss = -log(0.0001) = 9.2  # Even larger!

# This can spiral into instability
```

Even with clamping, it's finicky. **Simple MSE losses are more stable.**

## Advanced: If You Still Want Diversity

If simple approach works but colors are too conservative, you can add a **mild** diversity term:

```python
# In ColorizationLossSimple, add to forward():
# Variance encouragement (optional)
variance = torch.var(pred_ab, dim=[2, 3]).mean()
diversity_loss = 1.0 / (variance + 1.0)  # Smoother than log

total_loss = (
    perceptual_weight * perceptual_loss
    + lab_weight * lab_loss
    + ab_weight * ab_loss
    + 0.01 * diversity_loss  # Very small weight
)
```

This is **much more stable** than `-log(variance)`.

## Bottom Line

**For colorization, simpler is better:**
- Strong perceptual loss does most of the work
- LAB space keeps colors perceptually accurate
- Extra AB weight focuses on chrominance
- No need for complex regularization that can destabilize training

Try `train_colorization_simple.py` and you should see steady improvement from the start.
