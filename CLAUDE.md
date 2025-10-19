# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`vdeg` is a unified framework for multi-task image restoration using PyTorch with GAN training. It supports colorization, super-resolution, and artifact removal in a single training pipeline with balanced discriminator updates and NoGAN strategy.

## Development Setup

This project uses `uv` for dependency management.

**Install dependencies:**
```bash
uv sync
```

**Run training:**
```bash
uv run python train.py
```

Edit `train.py` to configure experiments via the `create_config()` function.

## Key Dependencies

- **Core**: torch, torchvision, torchcodec
- **Metrics**: torchmetrics, lpips
- **Vision**: kornia
- **Logging**: tensorboardx, tqdm
- **Export**: onnx, onnxscript

## Project Architecture

### Configuration System

All training is configured via `src/config.py` dataclasses:
- `ExperimentConfig`: Top-level config containing all sub-configs
- `DatasetConfig`: Dataset paths, batch size, workers
- `ModelConfig`: Architecture selection, channels, upscale
- `TaskConfig`: Enable/disable tasks, degradation probabilities
- `LossConfig`: Loss weights, GAN settings
- `DiscriminatorConfig`: Balanced discriminator parameters
- `TrainingConfig`: Learning rates, epochs, AMP, EMA

### Data Format

All models use a unified data format:
- **Input**: `BxTxCxHxW` where T is the temporal dimension (T=1 for images)
- **Output**: `BxCxHxW` (single frame output)

### Directory Structure

```
src/
├── config.py           # Unified configuration system
├── datasets/           # Image and video dataset loaders
├── degradations/       # Composable degradation pipelines
├── losses/            # Multi-task loss functions (core.py, gan.py)
├── models/            # HAT and SimpleUNet architectures
├── training/          # Multi-task GAN trainer (NoGAN strategy)
└── utils/             # Logger, EMA, smart checkpointing, ONNX export
train.py               # Unified training script
```

### Core Components

**Datasets** (`src/datasets/`):
- `ImageRestorationDataset`: Loads images recursively, resizes with aspect-ratio-preserving cropping
- `VideoRestorationDataset`: Samples clips from videos with frame rate normalization
- Both apply degradation functions to create training pairs

**Degradations** (`src/degradations/`):
- Composable pipeline using `DegradationPipeline`
- Available: `GaussianNoise`, `PoissonNoise`, `JPEGCompression`, `GaussianBlur`, `MotionBlur`, `Downscale`, `Grayscale`, `ReduceDynamicRange`

**Trainer** (`src/training/multitask_gan_trainer.py`):
- **BalancedMultiTaskGANTrainer**: Single trainer for all tasks
- NoGAN strategy with 3 phases: pretrain, critic, gan
- Balanced discriminator with adaptive updates
- Mixed precision (AMP), gradient clipping/accumulation
- EMA for stable generator weights
- Multi-GPU support via DataParallel

**Checkpointing**:
- Experiments organized in `experiments/{experiment_id}/`
- Subdirectories: `checkpoints/`, `logs/`, `exports/`
- **Only EMA generator weights are exported** (better quality)
- `best_model_info.json` contains metadata linking PyTorch and ONNX exports
- Automatic ONNX export for deployment

**Losses** (`src/losses/core.py`):
- `MultiTaskLoss`: Unified loss for all tasks (RGB + perceptual + LAB AB)
- `PerceptualLoss`: LPIPS-based perceptual loss
- Automatic LAB color space handling for colorization
- All losses return dict with components for detailed logging

## Training Workflow

1. **Configure** via `create_config()` in `train.py`
   - Set dataset paths, task enables, model architecture
   - All settings are typed dataclasses in `src/config.py`

2. **Run training**
   ```bash
   uv run python train.py
   ```

3. **NoGAN training phases** (automatic):
   - Phase 1: Pretrain generator with content loss
   - Phase 2: Pretrain discriminator (critic)
   - Phase 3: Balanced GAN training
   - Cycles repeat for progressive improvement

4. **Monitor progress**
   ```bash
   tensorboard --logdir experiments/{experiment_id}/logs/tensorboard
   ```

5. **Deploy best model**
   - Check `checkpoints/best_model_info.json` for metadata
   - Use EMA weights: `generator_ema_step_XXXX.pt`
   - Or ONNX export: `generator_ema_step_XXXX.onnx`

## Creating New Models

Models should:
- Accept input of shape `(B, T, C, H, W)`
- Output shape `(B, C, H, W)`
- Have an `input_shape` attribute: `(T, C, H, W)` for ONNX export
- Example: See `src/models/hat_simple.py` line 235

## Key Refactoring Changes

This codebase has been refactored to eliminate duplication:
- ✅ Single trainer (`BalancedMultiTaskGANTrainer`) replaces 3 trainers
- ✅ Unified loss system (`MultiTaskLoss`) replaces 8+ colorization losses
- ✅ Configuration-based (`src/config.py`) replaces hardcoded params
- ✅ Smart checkpointing (EMA only + metadata) for easy deployment
- ✅ Standardized model interface (all have `input_shape` property)
