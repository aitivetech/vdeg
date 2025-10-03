# Anti-Average Colorization Loss

## The Problem with Averaging

When models produce sepia/purple tones, it's usually because they're **averaging multiple valid color choices**:

```
Sky could be: light blue, dark blue, pale blue, azure, cyan
Model averages: → grayish purple/blue (safe but wrong)

Grass could be: bright green, dark green, yellow-green, olive
Model averages: → brownish sepia (safe but wrong)
```

This happens because **MSE loss rewards averaging**. The model learns that a neutral middle color has lower error than committing to one specific color.

## Why Classification Would Help

Papers like Zhang et al. treat color as classification:
- Discrete bins force the model to pick ONE specific color
- No averaging allowed - must commit to a bin
- Works great but complex to implement correctly

## Our Solution: Anti-Average Loss

Instead of full classification, we use **multiple losses that specifically penalize averaging behavior**:

### 1. Standard LAB Loss (baseline)
```python
lab_loss = MSE(pred_LAB, target_LAB)
```
Basic color accuracy.

### 2. Saturation Loss (anti-gray)
```python
pred_saturation = sqrt(a² + b²)  # Distance from gray in AB space
target_saturation = sqrt(a² + b²)
saturation_loss = MSE(pred_saturation, target_saturation)
```
**Why it helps:** Gray/sepia has low saturation (near 0,0 in AB space). This heavily penalizes desaturated predictions.

### 3. Hue Diversity Loss (anti-monotone)
```python
pred_hue = atan2(b, a)  # Hue angle for each pixel
pred_hue_variance = var(pred_hue)  # How varied are the hues?
target_hue_variance = var(target_hue)
diversity_loss = MSE(pred_hue_var, target_hue_var)
```
**Why it helps:** If model outputs only one hue (all sepia), variance is low. This penalizes mode collapse to single colors.

### 4. Color Confidence Loss (anti-neutral)
```python
pred_distance_from_gray = sqrt(a² + b²)  # How far from gray?
target_distance_from_gray = sqrt(a² + b²)
confidence_loss = relu(target_distance - pred_distance).mean()
```
**Why it helps:** Penalizes if prediction is closer to gray than target. Forces model to be "confident" with colors.

## Settings

```python
ColorizationAntiAverageLoss(
    lab_weight=1.0,          # Standard color accuracy
    saturation_weight=0.01,  # Scaled down (saturation values are large)
    diversity_weight=0.5,    # Moderate - encourage varied hues
    confidence_weight=1.0,   # Strong - penalize neutral predictions
)
```

## Intuition

Each loss component attacks the averaging problem from a different angle:

| Component | Attacks | How |
|-----------|---------|-----|
| LAB | General inaccuracy | Basic color matching |
| Saturation | Gray/desaturated outputs | Pushes away from (0,0) in AB space |
| Diversity | Single-hue mode collapse | Requires varied hues within image |
| Confidence | Neutral/"safe" predictions | Penalizes if closer to gray than target |

Together, these make it **harder for the model to average** and **easier to commit to specific colors**.

## Why Not Full Classification?

Full classification (Zhang et al. style) requires:
1. Discretizing AB space into bins
2. Creating probability distributions over bins
3. Using cross-entropy loss
4. Decoding from bins back to continuous
5. Handling out-of-gamut colors
6. Complex gradient flow

Our approach is **simpler** but attacks the same core problem: **preventing averaging**.

## Tuning

If still getting sepia/purple:

1. **Increase confidence_weight** (e.g., 2.0)
   - Stronger penalty for neutral colors

2. **Increase saturation_weight** (e.g., 0.02 or 0.05)
   - But be careful - values are large, so small increments

3. **Increase diversity_weight** (e.g., 1.0)
   - Forces more varied hues

If colors become too wild:

1. **Decrease confidence_weight** (e.g., 0.5)
2. **Add perceptual loss** for semantic guidance

## Expected Behavior

With anti-average loss:
- ✅ Model commits to specific colors (not averages)
- ✅ Higher saturation (vibrant, not dull)
- ✅ Diverse hues within images (not monotone)
- ✅ Confident predictions (far from gray)

The confidence loss is especially powerful - it directly penalizes the "play it safe with neutral colors" strategy.

## Current Setup

Your `train_image_restoration.py` is now using:

```python
ColorizationAntiAverageLoss(
    lab_weight=1.0,
    saturation_weight=0.01,   # Scaled for large saturation values
    diversity_weight=0.5,
    confidence_weight=1.0,
)
```

This should prevent sepia/purple mode collapse!

## Monitoring

Watch these metrics in TensorBoard:

- **lab**: Should decrease (color accuracy improving)
- **saturation**: Should stay moderate (not explode)
- **diversity**: Should decrease (hue variance matching)
- **confidence**: Should decrease (getting bolder with colors)

If **confidence** stays high → model is still being too neutral → increase `confidence_weight`

## Alternative: Try Both

Since the enhanced loss helped with skin tones, you could also try **combining** approaches:

```python
# Use anti-average for main loss, add a bit of perceptual for semantics
from src.losses import ColorizationAntiAverageLoss
from src.losses.perceptual import PerceptualLoss

anti_avg = ColorizationAntiAverageLoss(...)
perceptual = PerceptualLoss(net='alex', device=DEVICE)

# In training loop:
loss_dict = anti_avg(pred, target)
perc_loss = perceptual(pred, target)
total_loss = loss_dict['total'] + 0.3 * perc_loss
```

This gives you the best of both worlds: anti-averaging + semantic guidance.
