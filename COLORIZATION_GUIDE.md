# Colorization Training Guide

## The Problem

Colorization models often get stuck in local minima, producing only red/green colors or sepia tones. This is called "mode collapse" - the model finds a safe middle ground instead of learning the full color spectrum.

## The Solution

The improved `ColorizationLoss` addresses these issues with several key strategies:

### 1. LAB Color Space with Channel Separation

**Why it helps:**
- LAB separates luminance (L) from color (AB channels)
- A = green-red axis, B = blue-yellow axis
- The model can focus on learning colors without affecting brightness
- Prevents the model from "hiding" poor color choices behind brightness changes

**Implementation:**
```python
pred_lab = rgb_to_lab(pred)
pred_ab = pred_lab[:, 1:3, :, :]  # Just the color channels
ab_loss = F.smooth_l1_loss(pred_ab_norm, target_ab_norm)
```

### 2. Huber Loss (Smooth L1) Instead of MSE

**Why it helps:**
- MSE heavily penalizes outliers, making the model conservative
- Huber loss is more forgiving of small errors, encouraging exploration
- Prevents the model from averaging colors (which creates brown/gray)

**Implementation:**
```python
ab_loss = F.smooth_l1_loss(pred_ab_norm, target_ab_norm, beta=0.1)
```

### 3. Color Diversity Regularization

**Why it helps:**
- Penalizes low color variance in predictions
- Encourages the model to use a wider range of colors
- Prevents mode collapse to safe middle colors

**Implementation:**
```python
variance = torch.var(ab_flat, dim=2).mean()
diversity_loss = -torch.log(variance + 1e-6)
```

### 4. Higher Perceptual Loss Weight

**Why it helps:**
- VGG perceptual loss understands semantics (sky=blue, grass=green)
- Provides strong guidance for plausible colors
- Prevents physically implausible color combinations

**Implementation:**
```python
perceptual_loss = PerceptualLoss(net="vgg")  # VGG > AlexNet for colorization
```

### 5. Weight Priorities

**Recommended weights:**
```python
RGB_WEIGHT = 0.5          # Lowest - let model learn structure
PERCEPTUAL_WEIGHT = 1.0   # High - semantic guidance
AB_WEIGHT = 2.0           # Highest - focus on chrominance
DIVERSITY_WEIGHT = 0.1    # Regularization - prevent collapse
```

## Usage

### Quick Start

```bash
uv run python train_colorization.py
```

The script `train_colorization.py` has optimized defaults for colorization.

### Monitoring Training

**Watch these metrics in TensorBoard:**

1. **AB Chrominance Loss**: Should steadily decrease
2. **Diversity Loss**: Should stabilize (not go to -inf)
3. **Comparison Images**: Look for varied, realistic colors

**Red flags:**
- All images turning red/green → Increase `diversity_weight`
- Sepia/brown tones → Increase `ab_weight`, decrease `rgb_weight`
- Washed out colors → Increase `perceptual_weight`

### Training Tips

1. **Start with smaller images (256x256)**
   - Faster iteration for testing hyperparameters
   - Scale up once colors look good

2. **Use larger batch sizes (8-16)**
   - Helps diversity loss work better
   - More stable gradient estimates for color learning

3. **Learning rate warmup**
   - Start slow to avoid early mode collapse
   - The script includes 5 epochs of warmup

4. **Monitor early (first 1000 steps)**
   - If colors are bad early, they won't improve
   - Stop and adjust hyperparameters

5. **Use EMA model for inference**
   - EMA produces more stable, realistic colors
   - Already enabled in the trainer

## Advanced Tuning

### If you get mostly red/green:

```python
# Increase these:
DIVERSITY_WEIGHT = 0.2  # Double the diversity penalty
AB_WEIGHT = 3.0         # Even more focus on chrominance
PERCEPTUAL_WEIGHT = 1.5 # More semantic guidance

# Decrease this:
RGB_WEIGHT = 0.3        # Less direct RGB matching
```

### If colors are too conservative/dull:

```python
# Increase these:
HUBER_DELTA = 0.2       # More tolerance for errors
DIVERSITY_WEIGHT = 0.15 # Encourage more variation
LEARNING_RATE = 3e-4    # Faster learning

# Decrease this:
AB_WEIGHT = 1.5         # Less strict color matching
```

### If colors are implausible (blue grass, green sky):

```python
# Increase this:
PERCEPTUAL_WEIGHT = 2.0  # Much stronger semantic guidance

# Consider switching to VGG if using AlexNet:
PERCEPTUAL_NET = "vgg"   # Better semantic understanding
```

## Architecture Considerations

The `SimpleUNet` works well for colorization, but you might want to:

1. **Add skip connections with attention**
   - Helps preserve spatial structure while learning colors

2. **Increase model capacity**
   - Try `MODEL_CHANNELS = 96` or `128`
   - Colorization needs to learn complex color distributions

3. **Use deeper network**
   - More layers = better semantic understanding
   - Important for context-dependent colors

## Dataset Considerations

**Quality matters more than quantity for colorization:**

1. **Diverse color content**
   - Avoid datasets with mostly similar colors
   - Include variety: nature, urban, people, objects

2. **High quality originals**
   - No existing color casts or bad white balance
   - Natural, saturated colors for the model to learn

3. **Varied lighting**
   - Different times of day
   - Indoor/outdoor scenes
   - Helps model learn context-dependent colors

## Evaluation

**Don't just look at loss numbers:**

1. **Visual inspection is critical**
   - Loss can be low but colors still wrong
   - Look at comparison images in TensorBoard

2. **Test on diverse images**
   - Portraits, landscapes, objects, indoor/outdoor
   - A good model works across categories

3. **Compare to ground truth**
   - Are blues actually blue?
   - Are skin tones natural?
   - Is vegetation green (not yellow or brown)?

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Red/green only | Mode collapse | Increase diversity_weight |
| Sepia tones | RGB loss dominating | Increase ab_weight, decrease rgb_weight |
| Washed out | Weak perceptual guidance | Increase perceptual_weight, use VGG |
| Implausible colors | Weak semantic understanding | Increase perceptual_weight significantly |
| Training unstable | Learning rate too high | Add warmup, decrease LR |
| No improvement | Learning rate too low | Increase LR to 3e-4 or 5e-4 |

## Example Configuration That Works Well

```python
# Loss weights
RGB_WEIGHT = 0.5
PERCEPTUAL_WEIGHT = 1.0
AB_WEIGHT = 2.0
DIVERSITY_WEIGHT = 0.1
PERCEPTUAL_NET = "vgg"

# Training
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)
USE_EMA = True

# Start with these, then tune based on results
```

## References

The improved loss function is based on research in:
- Zhang et al. "Colorful Image Colorization" (ECCV 2016)
- Iizuka et al. "Let there be Color!" (SIGGRAPH 2016)
- LAB color space properties for perceptual uniformity
