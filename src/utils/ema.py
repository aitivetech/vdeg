"""Exponential Moving Average for model parameters."""

from typing import Any

import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that are updated using EMA.
    Useful for more stable predictions during training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        """
        Initialize EMA.

        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower updates)
        """
        self.model = model
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()

        # Freeze shadow model
        for param in self.shadow.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self) -> None:
        """Update EMA parameters."""
        model_params = dict(self.model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())

        for name, param in model_params.items():
            if param.requires_grad:
                shadow_params[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self) -> None:
        """Apply shadow parameters to the model (for evaluation)."""
        self._swap_parameters()

    def restore(self) -> None:
        """Restore original model parameters (after evaluation)."""
        self._swap_parameters()

    def _swap_parameters(self) -> None:
        """Swap parameters between model and shadow."""
        model_params = dict(self.model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())

        for name in model_params.keys():
            if model_params[name].requires_grad:
                tmp = model_params[name].data.clone()
                model_params[name].data.copy_(shadow_params[name].data)
                shadow_params[name].data.copy_(tmp)

    def state_dict(self) -> dict[str, Any]:
        """Get EMA state dict."""
        return {
            "decay": self.decay,
            "shadow": self.shadow.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load EMA state dict."""
        self.decay = state_dict["decay"]
        self.shadow.load_state_dict(state_dict["shadow"])
