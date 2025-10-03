# Colorization: The Simple Truth

## We've Been Overcomplicating It

After all the complex losses (perceptual, diversity, saturation, confidence, etc.), the real issues are:

1. **Model architecture** - needs to be powerful enough
2. **Learning rate** - too high causes oscillation
3. **Batch size** - too large can destabilize
4. **Data quality** - needs diverse, saturated colors

## The Simplest Loss That Works

`ColorizationWeightedLoss` - just weighted MSE:
- Saturated colors get HIGHER weight (1.5x)
- Gray/sepia colors get LOWER weight (0.5x)
- This naturally discourages averaging to neutral colors

**That's it.** No perceptual, no diversity, no confidence tricks.

## Critical Training Settings

### 1. Learning Rate (MOST IMPORTANT)

```python
LEARNING_RATE = 5e-5  # LOWER than default 1e-4
```

**Why:** Colorization is sensitive. Too high LR → oscillation.

### 2. Batch Size

```python
BATCH_SIZE = 16  # NOT 64!
```

**Why:** Batch size 64 with batch norm can cause instability in colorization.

### 3. Model Size

```python
MODEL_CHANNELS = 96  # Bigger than 64
```

**Why:** Colorization needs capacity to learn color distributions.

### 4. Image Size

```python
IMAGE_SIZE = (256, 256)  # Good starting point
```

**Why:** 512 is harder to learn initially.

### 5. Optimizer

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,  # LOWER!
    weight_decay=1e-5,
    betas=(0.9, 0.999),  # Default is fine
)
```

### 6. Learning Rate Schedule

```python
# Warmup then decay
warmup_steps = 1000
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-5,
    total_steps=NUM_EPOCHS * len(dataloader),
    pct_start=0.1,  # 10% warmup
)
```

## Why Loss Oscillates

### Common Causes:

1. **Learning rate too high**
   - Solution: Reduce to 5e-5 or even 2e-5

2. **Batch size too large**
   - Large batches → noisy gradients in batch norm
   - Solution: Use 8-16, not 64

3. **No LR warmup**
   - Model starts with random weights
   - High LR at start → unstable
   - Solution: Use OneCycleLR with warmup

4. **Model too small**
   - Can't learn color distributions
   - Solution: Increase MODEL_CHANNELS to 96 or 128

5. **Bad data**
   - Dataset has mostly gray/sepia images
   - Model learns to predict gray
   - Solution: Filter dataset for colorful images

## Recommended Full Config

```python
# Dataset
DATASET_PATH = "your/path"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16  # NOT 64

# Model
MODEL_CHANNELS = 96  # Bigger than default

# Training
NUM_EPOCHS = 100
LEARNING_RATE = 5e-5  # LOWER than default
WEIGHT_DECAY = 1e-5

# Loss - SIMPLE
loss_fn = ColorizationWeightedLoss(device=DEVICE)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Scheduler - WITH WARMUP
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    total_steps=NUM_EPOCHS * len(dataloader),
    pct_start=0.1,  # 10% warmup
)
```

## What You Should See

**Good training:**
```
Step 0:    loss=0.12
Step 100:  loss=0.10
Step 500:  loss=0.08
Step 1000: loss=0.06
Step 5000: loss=0.04
```
Steady decrease, no oscillation.

**Bad training (oscillating):**
```
Step 0:    loss=0.12
Step 100:  loss=0.15  ← went UP
Step 500:  loss=0.09
Step 1000: loss=0.13  ← went UP again
```
Bouncing around → LR too high or batch size too large.

## Emergency Fixes

### If loss oscillates:
1. STOP training
2. Reduce LR to 2e-5
3. Reduce batch size to 8
4. Add LR warmup (OneCycleLR)
5. Restart training

### If loss plateaus at high value (>0.10):
1. Increase model size (MODEL_CHANNELS = 128)
2. Check your data - is it actually colorful?
3. Train longer - colorization takes time

### If colors are still sepia/gray at loss=0.04:
1. Model might be too small → increase capacity
2. Data might be bad → check your dataset
3. Try adding tiny bit of perceptual: `0.01 * perceptual_loss`

## The Bottom Line

**Colorization is sensitive to hyperparameters.**

The loss function matters LESS than:
- Learning rate (too high → oscillation)
- Batch size (too large → instability)
- Model capacity (too small → can't learn)
- Data quality (gray images → gray outputs)

Start with:
- Simple weighted MSE loss
- LR = 5e-5
- Batch = 16
- Model channels = 96
- OneCycleLR with warmup

If that doesn't work, the problem is likely your data or model architecture, not the loss function.
