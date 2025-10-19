"""
Real-time Video Processing GUI with Multi-Model Pipeline.

Features:
- Multi-stage processing pipeline (chain multiple models)
- Real-time preview of each processing stage
- Comprehensive throughput and performance metrics
- Optimized for inference speed with torch.compile and AMP
- Generic - works with any restoration models
"""

import sys
import time
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.amp import autocast
import numpy as np
from PIL import Image
import cv2

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QProgressBar, QGroupBox,
        QGridLayout, QListWidget, QTextEdit, QSpinBox, QCheckBox
    )
    from PyQt6.QtCore import QThread, pyqtSignal, Qt
    from PyQt6.QtGui import QImage, QPixmap
except ImportError:
    print("PyQt6 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt6"])
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QProgressBar, QGroupBox,
        QGridLayout, QListWidget, QTextEdit, QSpinBox, QCheckBox
    )
    from PyQt6.QtCore import QThread, pyqtSignal, Qt
    from PyQt6.QtGui import QImage, QPixmap

from src.models import hat_simple_s, hat_simple_m, hat_simple_l


@dataclass
class ModelStage:
    """Configuration for one processing stage."""
    name: str
    model_path: str
    model: Optional[nn.Module] = None


class VideoProcessor(QThread):
    """Background thread for video processing with multi-model pipeline."""

    progress_updated = pyqtSignal(int, int, dict)  # current, total, metrics
    frame_processed = pyqtSignal(list)  # list of frames (input + each stage output)
    finished = pyqtSignal(str, dict)  # output_path, final_metrics
    error = pyqtSignal(str)  # error_message

    def __init__(self):
        super().__init__()
        self.input_path: Optional[str] = None
        self.output_path: Optional[str] = None
        self.model_stages: List[ModelStage] = []
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp: bool = True
        self.use_compile: bool = False
        self.should_stop: bool = False

        # Performance tracking per stage
        self.stage_times: List[List[float]] = []

    def stop(self):
        """Stop processing."""
        self.should_stop = True

    def load_model(self, checkpoint_path: str, model_name: str) -> nn.Module:
        """
        Load a model from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            model_name: Descriptive name for the model

        Returns:
            Loaded model ready for inference
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

            # Infer model architecture from checkpoint
            # Try to create appropriate model based on checkpoint structure
            state_dict = checkpoint if isinstance(checkpoint, dict) and 'model' not in checkpoint else checkpoint.get('model', checkpoint)

            # Check embed_dim from checkpoint to infer model size
            if 'conv_first.weight' in state_dict:
                embed_dim = state_dict['conv_first.weight'].shape[0]
                if embed_dim == 64:
                    model = hat_simple_s()
                elif embed_dim == 96:
                    model = hat_simple_m()
                elif embed_dim == 128:
                    model = hat_simple_l()
                else:
                    # Default to small
                    model = hat_simple_s()
            else:
                # Default if we can't determine
                model = hat_simple_s()

            # Load state dict
            model.load_state_dict(state_dict)
            model = model.to(self.device).eval()

            # Compile if requested
            if self.use_compile:
                model = torch.compile(model, mode="max-autotune")

            print(f"✓ Loaded {model_name}: {checkpoint_path}")
            print(f"  Model: {model.__class__.__name__}, Device: {self.device}")

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name} from {checkpoint_path}: {e}")

    def run(self):
        """Process video through multi-model pipeline."""
        try:
            self.should_stop = False

            # Load all models
            for stage in self.model_stages:
                if stage.model is None:
                    stage.model = self.load_model(stage.model_path, stage.name)

            # Initialize stage timing lists
            self.stage_times = [[] for _ in range(len(self.model_stages))]

            # Process video
            self._process_video()

        except Exception as e:
            import traceback
            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

    def _process_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Process a single frame through all model stages.

        Args:
            frame: Input frame (H, W, C) in BGR format

        Returns:
            List of frames: [input, stage1_output, stage2_output, ...]
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        frame_tensor = frame_tensor.to(self.device)

        outputs = [frame_rgb]  # Store original input
        current = frame_tensor

        # Process through each stage
        for i, stage in enumerate(self.model_stages):
            start_time = time.time()

            with torch.no_grad():
                if self.use_amp:
                    with autocast('cuda' if self.device == 'cuda' else 'cpu'):
                        output = stage.model(current)
                else:
                    output = stage.model(current)

            # Update timing
            self.stage_times[i].append(time.time() - start_time)

            # Convert to numpy for visualization
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
            outputs.append(output_np)

            # Use output as input for next stage
            current = output

        return outputs

    def _process_video(self):
        """Main video processing loop."""
        # Open input video
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_idx = 0
        start_time = time.time()

        try:
            while not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame through pipeline
                outputs = self._process_frame(frame)

                # Write final output to file (convert RGB back to BGR)
                final_output = outputs[-1]
                final_output_bgr = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)
                out.write(final_output_bgr)

                # Emit progress
                frame_idx += 1
                elapsed = time.time() - start_time
                fps_current = frame_idx / elapsed if elapsed > 0 else 0

                # Calculate average stage times
                stage_avg_times = [
                    np.mean(times[-30:]) if times else 0
                    for times in self.stage_times
                ]

                metrics = {
                    'fps': fps_current,
                    'frame': frame_idx,
                    'total': total_frames,
                    'elapsed': elapsed,
                    'stage_times': stage_avg_times,
                }

                self.progress_updated.emit(frame_idx, total_frames, metrics)

                # Emit frames for preview (downscale for GUI if needed)
                preview_outputs = []
                for output in outputs:
                    if output.shape[0] > 480:  # Downscale large frames for GUI
                        scale = 480 / output.shape[0]
                        new_w = int(output.shape[1] * scale)
                        output_small = cv2.resize(output, (new_w, 480))
                        preview_outputs.append(output_small)
                    else:
                        preview_outputs.append(output)

                self.frame_processed.emit(preview_outputs)

        finally:
            cap.release()
            out.release()

            # Emit final metrics
            total_time = time.time() - start_time
            final_metrics = {
                'total_frames': frame_idx,
                'total_time': total_time,
                'avg_fps': frame_idx / total_time if total_time > 0 else 0,
                'stage_avg_times': [
                    np.mean(times) if times else 0
                    for times in self.stage_times
                ],
            }

            self.finished.emit(self.output_path, final_metrics)


class VideoProcessingGUI(QMainWindow):
    """Main GUI window for video processing."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Model Video Processing")
        self.setGeometry(100, 100, 1400, 900)

        self.processor = VideoProcessor()
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.frame_processed.connect(self.update_preview)
        self.processor.finished.connect(self.processing_finished)
        self.processor.error.connect(self.processing_error)

        self.model_stages: List[ModelStage] = []
        self.preview_labels: List[QLabel] = []

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Model Configuration Section
        model_group = QGroupBox("Model Pipeline Configuration")
        model_layout = QVBoxLayout()

        # Model list
        self.model_list = QListWidget()
        model_layout.addWidget(QLabel("Processing Stages (in order):"))
        model_layout.addWidget(self.model_list)

        # Model buttons
        btn_layout = QHBoxLayout()
        self.add_model_btn = QPushButton("Add Model Stage")
        self.add_model_btn.clicked.connect(self.add_model_stage)
        self.remove_model_btn = QPushButton("Remove Selected")
        self.remove_model_btn.clicked.connect(self.remove_model_stage)
        self.clear_models_btn = QPushButton("Clear All")
        self.clear_models_btn.clicked.connect(self.clear_models)

        btn_layout.addWidget(self.add_model_btn)
        btn_layout.addWidget(self.remove_model_btn)
        btn_layout.addWidget(self.clear_models_btn)
        model_layout.addLayout(btn_layout)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # Video Selection Section
        video_group = QGroupBox("Video Files")
        video_layout = QGridLayout()

        self.input_label = QLabel("No input selected")
        self.output_label = QLabel("No output selected")
        self.select_input_btn = QPushButton("Select Input Video")
        self.select_output_btn = QPushButton("Select Output Location")

        self.select_input_btn.clicked.connect(self.select_input)
        self.select_output_btn.clicked.connect(self.select_output)

        video_layout.addWidget(QLabel("Input:"), 0, 0)
        video_layout.addWidget(self.input_label, 0, 1)
        video_layout.addWidget(self.select_input_btn, 0, 2)
        video_layout.addWidget(QLabel("Output:"), 1, 0)
        video_layout.addWidget(self.output_label, 1, 1)
        video_layout.addWidget(self.select_output_btn, 1, 2)

        video_group.setLayout(video_layout)
        main_layout.addWidget(video_group)

        # Processing Controls
        control_group = QGroupBox("Processing Controls")
        control_layout = QHBoxLayout()

        self.process_btn = QPushButton("Start Processing")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)

        self.use_compile_check = QCheckBox("Use torch.compile (faster, but slow first run)")
        self.use_compile_check.setChecked(False)

        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.use_compile_check)
        control_layout.addStretch()

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # Progress Section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)

        # Preview Section (will be populated dynamically based on stages)
        self.preview_group = QGroupBox("Preview")
        self.preview_layout = QHBoxLayout()
        self.preview_group.setLayout(self.preview_layout)
        main_layout.addWidget(self.preview_group)

        # Metrics Section
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(100)

        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

    def add_model_stage(self):
        """Add a new model stage to the pipeline."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Checkpoint",
            "",
            "PyTorch Models (*.pt *.pth);;All Files (*)"
        )

        if file_path:
            name = Path(file_path).stem
            self.model_stages.append(ModelStage(name=name, model_path=file_path))
            self.model_list.addItem(f"{len(self.model_stages)}. {name}")
            self.update_preview_layout()
            self.update_process_button()

    def remove_model_stage(self):
        """Remove selected model stage."""
        current_row = self.model_list.currentRow()
        if current_row >= 0:
            del self.model_stages[current_row]
            self.model_list.takeItem(current_row)
            self.update_preview_layout()
            self.update_process_button()

    def clear_models(self):
        """Clear all model stages."""
        self.model_stages.clear()
        self.model_list.clear()
        self.update_preview_layout()
        self.update_process_button()

    def update_preview_layout(self):
        """Update preview section based on number of stages."""
        # Clear existing labels
        for label in self.preview_labels:
            label.deleteLater()
        self.preview_labels.clear()

        # Create new labels: input + one per stage
        num_previews = len(self.model_stages) + 1
        for i in range(num_previews):
            if i == 0:
                name = "Input"
            else:
                name = self.model_stages[i-1].name

            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(240, 180)
            label.setStyleSheet("border: 1px solid gray;")
            self.preview_labels.append(label)
            self.preview_layout.addWidget(label)

    def select_input(self):
        """Select input video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Video",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )

        if file_path:
            self.processor.input_path = file_path
            self.input_label.setText(Path(file_path).name)
            self.update_process_button()

    def select_output(self):
        """Select output video file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output Video",
            "",
            "Video Files (*.mp4 *.avi);;All Files (*)"
        )

        if file_path:
            self.processor.output_path = file_path
            self.output_label.setText(Path(file_path).name)
            self.update_process_button()

    def update_process_button(self):
        """Enable/disable process button based on configuration."""
        can_process = (
            self.processor.input_path is not None and
            self.processor.output_path is not None and
            len(self.model_stages) > 0
        )
        self.process_btn.setEnabled(can_process)

    def start_processing(self):
        """Start video processing."""
        self.processor.model_stages = self.model_stages
        self.processor.use_compile = self.use_compile_check.isChecked()

        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.metrics_text.setText("Processing started...")

        self.processor.start()

    def stop_processing(self):
        """Stop video processing."""
        self.processor.stop()
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopping...")

    def update_progress(self, current: int, total: int, metrics: dict):
        """Update progress bar and metrics."""
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)

        # Update status
        fps = metrics.get('fps', 0)
        elapsed = metrics.get('elapsed', 0)
        self.status_label.setText(
            f"Frame {current}/{total} | {fps:.1f} FPS | "
            f"Elapsed: {elapsed:.1f}s"
        )

        # Update metrics text
        stage_times = metrics.get('stage_times', [])
        metrics_text = f"Performance Metrics:\n"
        metrics_text += f"Current FPS: {fps:.2f}\n"
        for i, stage_time in enumerate(stage_times):
            stage_name = self.model_stages[i].name if i < len(self.model_stages) else f"Stage {i+1}"
            metrics_text += f"{stage_name}: {stage_time*1000:.1f}ms\n"

        self.metrics_text.setText(metrics_text)

    def update_preview(self, frames: List[np.ndarray]):
        """Update preview images."""
        for i, (frame, label) in enumerate(zip(frames, self.preview_labels)):
            if frame is not None:
                # Convert numpy array to QImage
                h, w, c = frame.shape
                bytes_per_line = 3 * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

                # Scale to fit label
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                label.setPixmap(scaled_pixmap)

    def processing_finished(self, output_path: str, metrics: dict):
        """Handle processing completion."""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        total_time = metrics.get('total_time', 0)
        avg_fps = metrics.get('avg_fps', 0)
        total_frames = metrics.get('total_frames', 0)

        result_text = f"✓ Processing Complete!\n\n"
        result_text += f"Output: {Path(output_path).name}\n"
        result_text += f"Total Frames: {total_frames}\n"
        result_text += f"Total Time: {total_time:.1f}s\n"
        result_text += f"Average FPS: {avg_fps:.2f}\n\n"
        result_text += "Stage Average Times:\n"

        stage_avg_times = metrics.get('stage_avg_times', [])
        for i, stage_time in enumerate(stage_avg_times):
            stage_name = self.model_stages[i].name if i < len(self.model_stages) else f"Stage {i+1}"
            result_text += f"  {stage_name}: {stage_time*1000:.1f}ms\n"

        self.metrics_text.setText(result_text)
        self.status_label.setText("Processing complete!")

    def processing_error(self, error_msg: str):
        """Handle processing errors."""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.metrics_text.setText(f"ERROR:\n{error_msg}")
        self.status_label.setText("Processing failed")


def main():
    """Run the application."""
    app = QApplication(sys.argv)
    window = VideoProcessingGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
