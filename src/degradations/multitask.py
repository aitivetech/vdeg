"""
Multi-task degradation pipeline for combined training.

Applies multiple degradations with configurable probabilities to support
simultaneous training on super-resolution, artifact removal, and colorization.
"""

import random
from typing import Callable

import torch


class MultiTaskDegradation:
    """
    Multi-task degradation that randomly applies different combinations of degradations.

    This enables training a single model on multiple tasks:
    - Super-resolution (downscale)
    - Denoising (noise)
    - Artifact removal (compression, blur)
    - Colorization (grayscale)

    Each degradation can be applied with a configurable probability.
    """

    def __init__(
        self,
        downscale_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        noise_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        blur_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        compression_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        grayscale_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        downscale_prob: float = 1.0,
        noise_prob: float = 0.7,
        blur_prob: float = 0.5,
        compression_prob: float = 0.8,
        grayscale_prob: float = 1.0,
    ) -> None:
        """
        Initialize multi-task degradation.

        Args:
            downscale_fn: Super-resolution degradation (reduces resolution)
            noise_fn: Noise degradation
            blur_fn: Blur degradation
            compression_fn: Compression artifact degradation (JPEG, etc.)
            grayscale_fn: Grayscale/colorization degradation
            downscale_prob: Probability of applying downscale (0-1)
            noise_prob: Probability of applying noise (0-1)
            blur_prob: Probability of applying blur (0-1)
            compression_prob: Probability of applying compression (0-1)
            grayscale_prob: Probability of applying grayscale (0-1)
        """
        self.degradations = []

        # Store degradations with their probabilities
        # Order matters: apply in realistic sequence
        if blur_fn is not None:
            self.degradations.append(('blur', blur_fn, blur_prob))

        if downscale_fn is not None:
            self.degradations.append(('downscale', downscale_fn, downscale_prob))

        if noise_fn is not None:
            self.degradations.append(('noise', noise_fn, noise_prob))

        if compression_fn is not None:
            self.degradations.append(('compression', compression_fn, compression_prob))

        if grayscale_fn is not None:
            self.degradations.append(('grayscale', grayscale_fn, grayscale_prob))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply degradations with their respective probabilities.

        Args:
            x: Input tensor of shape (T, C, H, W)

        Returns:
            Degraded tensor (shape depends on degradations applied)
        """
        for name, degradation_fn, prob in self.degradations:
            if random.random() < prob:
                x = degradation_fn(x)

        return x

    def set_probabilities(
        self,
        downscale_prob: float | None = None,
        noise_prob: float | None = None,
        blur_prob: float | None = None,
        compression_prob: float | None = None,
        grayscale_prob: float | None = None,
    ) -> None:
        """
        Dynamically adjust degradation probabilities during training.

        Useful for curriculum learning (e.g., start with easy tasks, add harder ones).

        Args:
            downscale_prob: New probability for downscale (None to keep current)
            noise_prob: New probability for noise (None to keep current)
            blur_prob: New probability for blur (None to keep current)
            compression_prob: New probability for compression (None to keep current)
            grayscale_prob: New probability for grayscale (None to keep current)
        """
        prob_map = {
            'downscale': downscale_prob,
            'noise': noise_prob,
            'blur': blur_prob,
            'compression': compression_prob,
            'grayscale': grayscale_prob,
        }

        for i, (name, degradation_fn, old_prob) in enumerate(self.degradations):
            new_prob = prob_map.get(name)
            if new_prob is not None:
                self.degradations[i] = (name, degradation_fn, new_prob)


class TaskSpecificDegradation:
    """
    Task-specific degradation that ensures specific degradations for each task.

    Unlike MultiTaskDegradation which randomly applies degradations,
    this class ensures specific combinations for targeted training:
    - Super-resolution: downscale + optional artifacts
    - Denoising: noise only
    - Artifact removal: compression + blur only
    - Colorization: grayscale only
    - Multi-task: all degradations
    """

    def __init__(
        self,
        task: str,
        downscale_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        noise_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        blur_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        compression_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        grayscale_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Initialize task-specific degradation.

        Args:
            task: Task type ('super_resolution', 'denoising', 'artifact_removal',
                  'colorization', 'multi_task')
            downscale_fn: Super-resolution degradation
            noise_fn: Noise degradation
            blur_fn: Blur degradation
            compression_fn: Compression degradation
            grayscale_fn: Grayscale degradation
        """
        self.task = task
        self.degradations = []

        # Define task-specific degradation sequences
        if task == 'super_resolution':
            # SR: downscale is mandatory
            if downscale_fn is not None:
                self.degradations.append(downscale_fn)
            # Optional: add compression for realistic SR
            if compression_fn is not None and random.random() < 0.5:
                self.degradations.append(compression_fn)

        elif task == 'denoising':
            # Denoising: only noise
            if noise_fn is not None:
                self.degradations.append(noise_fn)

        elif task == 'artifact_removal':
            # Artifact removal: blur + compression
            if blur_fn is not None:
                self.degradations.append(blur_fn)
            if compression_fn is not None:
                self.degradations.append(compression_fn)

        elif task == 'colorization':
            # Colorization: only grayscale
            if grayscale_fn is not None:
                self.degradations.append(grayscale_fn)

        elif task == 'multi_task':
            # Multi-task: all degradations in realistic order
            if blur_fn is not None:
                self.degradations.append(blur_fn)
            if downscale_fn is not None:
                self.degradations.append(downscale_fn)
            if noise_fn is not None:
                self.degradations.append(noise_fn)
            if compression_fn is not None:
                self.degradations.append(compression_fn)
            if grayscale_fn is not None:
                self.degradations.append(grayscale_fn)

        else:
            raise ValueError(
                f"Unknown task: {task}. Must be one of: "
                f"'super_resolution', 'denoising', 'artifact_removal', "
                f"'colorization', 'multi_task'"
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply task-specific degradations.

        Args:
            x: Input tensor of shape (T, C, H, W)

        Returns:
            Degraded tensor
        """
        for degradation_fn in self.degradations:
            x = degradation_fn(x)
        return x
