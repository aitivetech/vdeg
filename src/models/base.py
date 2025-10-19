"""Base model interface for restoration models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class RestorationModel(nn.Module, ABC):
    """
    Base class for all restoration models.

    All models should inherit from this class and implement the forward method.
    This ensures consistent interface for training and export.
    """

    def __init__(
        self,
        num_frames: int = 1,
        in_channels: int = 3,
        out_channels: int = 3,
        image_size: int = 256,
    ):
        """
        Initialize base restoration model.

        Args:
            num_frames: Number of temporal frames (T dimension)
            in_channels: Number of input channels
            out_channels: Number of output channels
            image_size: Default image size for ONNX export
        """
        super().__init__()
        self._input_shape = (num_frames, in_channels, image_size, image_size)
        self._num_frames = num_frames
        self._in_channels = in_channels
        self._out_channels = out_channels

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        """
        Get input shape for ONNX export.

        Returns:
            Tuple of (T, C, H, W)
        """
        return self._input_shape

    @input_shape.setter
    def input_shape(self, shape: tuple[int, int, int, int]) -> None:
        """
        Set input shape for ONNX export.

        Args:
            shape: Tuple of (T, C, H, W)
        """
        self._input_shape = shape

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        pass

    def get_model_info(self) -> dict[str, any]:
        """
        Get model information for logging and metadata.

        Returns:
            Dictionary with model information
        """
        return {
            "model_class": self.__class__.__name__,
            "input_shape": self._input_shape,
            "num_frames": self._num_frames,
            "in_channels": self._in_channels,
            "out_channels": self._out_channels,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
