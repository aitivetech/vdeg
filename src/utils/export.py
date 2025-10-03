"""ONNX export functionality."""

from pathlib import Path

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int],
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> None:
    """
    Export model to ONNX format.

    Args:
        model: Model to export
        output_path: Path to save ONNX file
        input_shape: Input shape (T, C, H, W)
        opset_version: ONNX opset version
        dynamic_batch: Whether to support dynamic batch size
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy input
    T, C, H, W = input_shape
    dummy_input = torch.randn(1, T, C, H, W)

    # Move to same device as model
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

    # Set model to eval mode
    model.eval()

    # Define dynamic axes for batch dimension
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Export
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
