# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`vdeg` is a comprehensive framework for training video and image restoration models using PyTorch. It supports various restoration tasks including denoising, super-resolution, artifact removal, colorization, and SDR to HDR enhancement.

## Development Setup

This project uses `uv` for dependency management.

**Install dependencies:**
```bash
uv sync
```

**Run training scripts:**
```bash
uv run python train_image_restoration.py
uv run python train_video_restoration.py
```

**Run tests:**
```bash
uv run python test_training.py
```

## Key Dependencies

- **Core**: torch, torchvision, torchcodec
- **Metrics**: torchmetrics, lpips
- **Vision**: kornia
- **Logging**: tensorboardx, tqdm
- **Export**: onnx, onnxscript

## Project Architecture

### Data Format

All models use a unified data format:
- **Input**: `BxTxCxHxW` where T is the temporal dimension (T=1 for images)
- **Output**: `BxCxHxW` (single frame output)
- For videos: output is the middle frame, other frames provide temporal context

### Directory Structure

```
src/
├── datasets/           # Image and video dataset loaders with normalization
├── degradations/       # Degradation pipelines (noise, blur, compression, etc.)
├── losses/            # Loss functions (perceptual, mixed, colorization)
├── models/            # Model architectures (SimpleUNet example)
├── training/          # Trainer class with AMP, EMA, checkpointing
└── utils/             # Logger, EMA, checkpointing, ONNX export
```

### Core Components

**Datasets** (`src/datasets/`):
- `ImageRestorationDataset`: Loads images recursively, resizes with aspect-ratio-preserving cropping
- `VideoRestorationDataset`: Samples clips from videos with frame rate normalization
- Both apply degradation functions to create training pairs

**Degradations** (`src/degradations/`):
- Composable pipeline using `DegradationPipeline`
- Available: `GaussianNoise`, `PoissonNoise`, `JPEGCompression`, `GaussianBlur`, `MotionBlur`, `Downscale`, `Grayscale`, `ReduceDynamicRange`

**Trainer** (`src/training/trainer.py`):
- Mixed precision (AMP) support
- Gradient clipping and accumulation
- EMA for stable model weights
- Automatic checkpointing with best model tracking
- TensorBoard logging with comparison images
- Multi-GPU support via DataParallel
- torch.compile support

**Checkpointing**:
- Experiments organized in `experiments/{experiment_id}/`
- Subdirectories: `checkpoints/`, `logs/`, `exports/`
- Saves both regular model and EMA version
- Automatic ONNX export alongside checkpoints

**Losses** (`src/losses/`):
- `MixedLoss`: Combines MSE + LPIPS perceptual loss
- `ColorizationLoss`: LAB color space + perceptual + MSE
- All losses return dict with components for detailed logging

## Training Workflow

1. **Configure** the training script parameters (all at the top of the file)
2. **Create degradation pipeline** to generate training data from high-quality images/videos
3. **Initialize model, loss, optimizer**
4. **Create Trainer** with desired settings (AMP, EMA, gradient clipping, etc.)
5. **Train** with automatic logging, checkpointing, and ONNX export

## Creating New Models

Models should:
- Accept input of shape `(B, T, C, H, W)`
- Output shape `(B, C, H, W)`
- Have an `input_shape` attribute for ONNX export: `(T, C, H, W)`
- Handle T=1 for images, T>1 for videos

## Dataset Switching During Training

To change datasets/degradations during training:
- Create new dataloader with different settings
- Call `trainer.train_epoch(new_dataloader, epoch)` with the new loader
- Optimizer, scheduler, and EMA state persist across dataloader changes
