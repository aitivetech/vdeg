# Colorization Loss Tuning Guide

## The Purple/Sepia Problem

When colorization works but produces purple/sepia tones instead of vibrant colors, it means:
- Model is being too conservative with colors
- Perceptual loss is dominating too much
- Chrominance (AB) channels need more emphasis

## Solution: ColorizationLossEnhanced

The enhanced loss addresses this with:

1. **Reduced perceptual weight** (1.0 → 0.5)
   - Less semantic constraints = more color freedom
   - Model can explore a wider color range

2. **Separated L and AB weights**
   - L (luminance): 0.3 (lower)
   - AB (chrominance): 3.0 (MUCH higher)
   - This heavily emphasizes learning correct colors over brightness

3. **Saturation loss** (new!)
   - Penalizes desaturated (gray/sepia) predictions
   - Encourages vibrant, saturated colors
   - Computed as MSE between color saturation in LAB space

## Default Settings (Good Starting Point)

```python
ColorizationLossEnhanced(
    perceptual_weight=0.5,   # Reduced from 1.0
    l_weight=0.3,            # Lower weight on brightness
    ab_weight=3.0,           # HIGHEST - focus on color!
    saturation_weight=0.5,   # Encourage vibrant colors
    perceptual_net='alex',
    device=DEVICE,
)
```

## Tuning for Different Issues

### If colors are still too purple/sepia:

```python
# Option 1: Reduce perceptual even more
perceptual_weight=0.3  # Was 0.5

# Option 2: Increase AB weight even more
ab_weight=4.0  # Was 3.0

# Option 3: Increase saturation weight
saturation_weight=1.0  # Was 0.5

# Or combine all three!
```

### If colors are too wild/unrealistic:

```python
# Increase perceptual weight for more semantic guidance
perceptual_weight=0.8  # Was 0.5

# Or reduce AB weight
ab_weight=2.0  # Was 3.0
```

### If brightness is wrong:

```python
# Increase L weight
l_weight=0.5  # Was 0.3
```

### If colors are washed out:

```python
# Increase saturation weight significantly
saturation_weight=1.0  # Was 0.5

# And increase AB weight
ab_weight=4.0  # Was 3.0
```

## Understanding the Loss Components

When training, watch these metrics in TensorBoard:

1. **perceptual** (~0.2-0.5)
   - Semantic guidance
   - If too high: increase perceptual_weight
   - If too low: decrease perceptual_weight

2. **l_luminance** (~0.05-0.15)
   - Brightness accuracy
   - Usually lowest component

3. **ab_chrominance** (~0.1-0.3)
   - Color accuracy
   - Should be HIGHER than l_luminance
   - If too high relative to others: decrease ab_weight

4. **saturation** (~0.01-0.1)
   - Color vibrancy
   - Low values = good color saturation
   - If stuck high: model is producing gray/sepia

## Weight Balance Guidelines

**Golden rule:** AB should be 2-4x higher than L or perceptual

Good balances:
- Conservative: `perceptual=0.5, l=0.3, ab=2.0, sat=0.3`
- Balanced (default): `perceptual=0.5, l=0.3, ab=3.0, sat=0.5`
- Vibrant: `perceptual=0.3, l=0.2, ab=4.0, sat=1.0`
- Maximum color: `perceptual=0.2, l=0.2, ab=5.0, sat=1.5`

## Training Tips

1. **Start with default settings**
   - Train for 5k-10k steps
   - Check comparison images in TensorBoard

2. **Identify the problem:**
   - Purple/sepia → Too conservative
   - Wild colors → Too much freedom
   - Wrong objects → Need more perceptual
   - Dull colors → Need more saturation

3. **Adjust ONE parameter at a time:**
   - Change by 0.2-0.5 increments
   - Retrain from checkpoint or scratch
   - Compare results

4. **Watch the loss ratios:**
   - If AB loss is 10x smaller than perceptual → increase ab_weight
   - If saturation loss is stuck → increase saturation_weight

## Example Tuning Session

```
Initial (purple/sepia tones):
  perceptual_weight=1.0, l_weight=1.0, ab_weight=2.0, saturation_weight=0.1

Step 1 - Reduce perceptual dominance:
  perceptual_weight=0.5  ← Changed
  → Colors slightly better but still sepia

Step 2 - Emphasize chrominance:
  ab_weight=3.0  ← Changed
  → Better! More color variety

Step 3 - Add saturation push:
  saturation_weight=0.5  ← Changed
  → Much better! Vibrant colors

Step 4 - Fine-tune luminance:
  l_weight=0.3  ← Changed
  → Perfect! Good colors and brightness
```

## Advanced: Per-Channel AB Weights

If you want even more control, you can weight A and B channels separately in the loss function. This is useful if you notice:
- Too much red/green but not enough blue/yellow → increase B channel weight
- Too much blue/yellow but not enough red/green → increase A channel weight

(This would require modifying the loss function - let me know if needed)

## Quick Reference

| Issue | Solution |
|-------|----------|
| Purple/sepia tones | ↓ perceptual_weight, ↑ ab_weight, ↑ saturation_weight |
| Washed out colors | ↑ saturation_weight, ↑ ab_weight |
| Wrong object colors | ↑ perceptual_weight |
| Too bright/dark | ↑ l_weight |
| Too vibrant (unrealistic) | ↓ saturation_weight, ↑ perceptual_weight |
| Gray output | ↑↑ saturation_weight, check if model has sigmoid |

## Current Setup

Your `train_image_restoration.py` is now using:

```python
ColorizationLossEnhanced(
    perceptual_weight=0.5,
    l_weight=0.3,
    ab_weight=3.0,
    saturation_weight=0.5,
)
```

This should significantly improve color quality compared to the simple version!
