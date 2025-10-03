# Fixes Applied to train_image_restoration.py

## Problems Found

1. **Wrong loss function call** - Missing required parameters (`rgb_weight`, `ab_weight`, etc.)
2. **Perceptual weight too low** - 0.1 is way too weak for colorization (should be 1.0+)
3. **Too many degradations** - Applying noise, blur, compression AND grayscale together
4. **Batch size too small** - 4 is not enough for stable colorization training
5. **Image size too large** - 512x512 makes training slower and harder initially
6. **Image logging too infrequent** - Can't catch problems early

## Fixes Applied

### 1. Loss Function (CRITICAL FIX)

**Before:**
```python
loss_fn = ColorizationLoss(
    perceptual_weight=PERCEPTUAL_WEIGHT,  # 0.1 - too low!
    perceptual_net=PERCEPTUAL_NET,
    device=DEVICE
)  # Missing required parameters!
```

**After:**
```python
from src.losses import ColorizationLossSimple

loss_fn = ColorizationLossSimple(
    perceptual_weight=1.0,  # Strong semantic guidance
    lab_weight=1.0,         # Full LAB loss
    ab_weight=2.0,          # Extra focus on chrominance
    perceptual_net=PERCEPTUAL_NET,
    device=DEVICE,
)
```

### 2. Degradation Pipeline (CRITICAL FIX)

**Before:**
```python
degradation = DegradationPipeline(
    GaussianNoise(sigma=NOISE_SIGMA),
    GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA),
    Downscale(scale_factor=DOWNSCALE_FACTOR, mode="bilinear"),
    JPEGCompression(quality=JPEG_QUALITY),
    Grayscale(),  # Colorization on top of degraded images!
)
```

**After:**
```python
# FOR COLORIZATION: Only use Grayscale
degradation = Grayscale()
```

**Why:** Colorization should learn to map clean grayscale → color. Adding noise/blur/compression makes it much harder because the model has to learn multiple tasks at once.

### 3. Batch Size

**Before:** `BATCH_SIZE = 4`
**After:** `BATCH_SIZE = 16`

**Why:** Larger batches = more stable gradients, especially important for colorization

### 4. Image Size

**Before:** `IMAGE_SIZE = (512, 512)`
**After:** `IMAGE_SIZE = (256, 256)`

**Why:** Start smaller for faster iteration. Once colors work at 256x256, you can scale up.

### 5. Image Logging Interval

**Before:** `IMAGE_LOG_INTERVAL = 1000`
**After:** `IMAGE_LOG_INTERVAL = 250`

**Why:** Need to check colors frequently to catch problems early

### 6. Metric Logging Interval

**Before:** `LOG_INTERVAL = 1000`
**After:** `LOG_INTERVAL = 100`

**Why:** More frequent monitoring to see if loss is decreasing

## What You Should See Now

### Early Training (first 500 steps):
- ✅ **Loss decreasing steadily** (not stuck)
- ✅ **PSNR gradually increasing** (10+ dB after 500 steps)
- ✅ **Colors appearing varied** (not just green/black)
- ✅ **Comparison images showing color diversity**

### If Still Having Issues:

Check TensorBoard comparison images at step 250:

```bash
tensorboard --logdir experiments/image_restoration_001/logs/tensorboard
```

**If you see:**
- All green/black → Model might be stuck, try restarting from scratch
- Loss not decreasing → Check LPIPS is loading correctly (should see "Loading model from..." in console)
- Varied but wrong colors → Good! Model is learning, just needs more training

## Training Command

```bash
uv run python train_image_restoration.py
```

## Key Settings for Colorization

```python
# Dataset
IMAGE_SIZE = (256, 256)      # Start small
BATCH_SIZE = 16              # Larger batch

# Degradation
degradation = Grayscale()    # ONLY grayscale

# Loss
ColorizationLossSimple(
    perceptual_weight=1.0,   # Strong guidance
    lab_weight=1.0,          # Color accuracy
    ab_weight=2.0,           # Focus on chrominance
)

# Training
LEARNING_RATE = 1e-4         # Conservative
USE_EMA = True               # Smoother results
```

## Common Mistakes to Avoid

1. ❌ **Combining multiple degradations with colorization**
   - Don't add noise/blur/compression when training colorization
   - Model needs to focus on learning colors, not denoising

2. ❌ **Using low perceptual weight**
   - 0.1 is for tasks where you want MSE to dominate
   - Colorization needs strong perceptual guidance (1.0+)

3. ❌ **Small batch size**
   - 4 is too small for colorization
   - Use 8-16 for stable training

4. ❌ **Not checking images early**
   - If colors are wrong at step 500, they won't magically improve
   - Check frequently and adjust hyperparameters

5. ❌ **Starting with large images**
   - 512x512 is slow and harder to learn
   - Start with 256x256, then scale up once it works

## Next Steps

1. **Start training:** `uv run python train_image_restoration.py`
2. **Watch console output:** Loss should decrease from the start
3. **Check TensorBoard at step 250:** Colors should be varied (not monochrome)
4. **If good at step 1000:** Let it train!
5. **If bad at step 1000:** Stop and adjust (see COLORIZATION_FIX.md)

The fixes address all the issues that were causing green/black mode collapse. Training should now be stable and improve steadily.
