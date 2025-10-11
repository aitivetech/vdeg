"""
Real-time Video Restoration GUI Application with Two-Stage Pipeline.

Features:
- Two-stage processing pipeline (both stages optional):
  1. Colorization (optional)
  2. Enhancement: super-resolution, denoising, artifact removal (optional)
- Real-time preview of input and output frames
- Comprehensive throughput and performance metrics
- Optimized for inference speed with:
  - torch.compile
  - AMP (Automatic Mixed Precision)
  - Efficient video I/O with OpenCV
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum

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
        QGridLayout, QComboBox, QCheckBox, QSpinBox, QTextEdit
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
        QGridLayout, QComboBox, QCheckBox, QSpinBox, QTextEdit
    )
    from PyQt6.QtCore import QThread, pyqtSignal, Qt
    from PyQt6.QtGui import QImage, QPixmap

from src.models import SimpleUNetColorization, SimpleUNet


class ProcessingStage(Enum):
    """Processing stage types."""
    COLORIZATION = "colorization"
    ENHANCEMENT = "enhancement"


class VideoProcessor(QThread):
    """Background thread for video processing."""

    progress_updated = pyqtSignal(int, int, dict)  # current, total, metrics
    frame_processed = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # input, intermediate, output
    finished = pyqtSignal(str, dict)  # output_path, final_metrics
    error = pyqtSignal(str)  # error_message

    def __init__(self):
        super().__init__()
        self.input_path: Optional[str] = None
        self.output_path: Optional[str] = None
        self.colorization_model: Optional[nn.Module] = None
        self.enhancement_model: Optional[nn.Module] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp: bool = True
        self.should_stop: bool = False

        # Performance tracking
        self.stage1_times = []
        self.stage2_times = []
        self.total_times = []

    def stop(self):
        """Stop processing."""
        self.should_stop = True

    def run(self):
        """Process video."""
        try:
            self.should_stop = False
            self._process_video()
        except Exception as e:
            import traceback
            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

    def _process_video(self):
        """Main video processing loop."""
        # Open input video
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine output resolution (might be upscaled if enhancement model has upscale)
        output_width, output_height = self._get_output_resolution(input_width, input_height)

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (output_width, output_height))

        # Processing loop
        frame_idx = 0
        start_time = time.time()

        # Reset performance tracking
        self.stage1_times = []
        self.stage2_times = []
        self.total_times = []

        while cap.isOpened() and not self.should_stop:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.perf_counter()

            # Process frame through pipeline
            intermediate_frame, output_frame, stage1_time, stage2_time = self._process_frame(frame)

            # Write output
            out.write(output_frame)

            # Track performance
            frame_time = time.perf_counter() - frame_start
            self.total_times.append(frame_time)
            if stage1_time > 0:
                self.stage1_times.append(stage1_time)
            if stage2_time > 0:
                self.stage2_times.append(stage2_time)

            # Keep only recent samples for moving average
            if len(self.total_times) > 30:
                self.total_times.pop(0)
            if len(self.stage1_times) > 30:
                self.stage1_times.pop(0)
            if len(self.stage2_times) > 30:
                self.stage2_times.pop(0)

            # Calculate metrics
            metrics = self._calculate_metrics(frame_idx + 1, total_frames, start_time)

            # Emit signals
            self.frame_processed.emit(frame, intermediate_frame, output_frame)
            self.progress_updated.emit(frame_idx + 1, total_frames, metrics)

            frame_idx += 1

        # Cleanup
        cap.release()
        out.release()

        if not self.should_stop:
            final_metrics = self._calculate_final_metrics(total_frames, start_time)
            self.finished.emit(self.output_path, final_metrics)

    def _get_output_resolution(self, input_width: int, input_height: int) -> Tuple[int, int]:
        """Determine output resolution based on models."""
        width, height = input_width, input_height

        # Check if enhancement model has upscaling
        if self.enhancement_model is not None:
            if hasattr(self.enhancement_model, 'upscale_factor'):
                scale = self.enhancement_model.upscale_factor
                width *= scale
                height *= scale

        return width, height

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Process a single frame through the pipeline.

        Returns:
            intermediate_frame: Frame after stage 1 (or input if stage 1 disabled)
            output_frame: Final output frame
            stage1_time: Time spent in stage 1 (seconds)
            stage2_time: Time spent in stage 2 (seconds)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare tensor
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        frame_tensor = frame_tensor.to(self.device)

        stage1_time = 0.0
        stage2_time = 0.0
        intermediate_tensor = frame_tensor

        # Stage 1: Colorization (if enabled)
        if self.colorization_model is not None:
            stage1_start = time.perf_counter()
            with torch.no_grad():
                with autocast("cuda", enabled=self.use_amp and self.device == "cuda"):
                    intermediate_tensor = self.colorization_model(frame_tensor)
                    intermediate_tensor = intermediate_tensor.unsqueeze(1)  # Add T dimension back
            stage1_time = time.perf_counter() - stage1_start

        # Stage 2: Enhancement (if enabled)
        output_tensor = intermediate_tensor
        if self.enhancement_model is not None:
            stage2_start = time.perf_counter()
            with torch.no_grad():
                with autocast("cuda", enabled=self.use_amp and self.device == "cuda"):
                    output_tensor = self.enhancement_model(intermediate_tensor)
            stage2_time = time.perf_counter() - stage2_start

        # Convert intermediate to numpy (for display)
        intermediate_np = intermediate_tensor.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        intermediate_np = np.clip(intermediate_np * 255, 0, 255).astype(np.uint8)
        intermediate_bgr = cv2.cvtColor(intermediate_np, cv2.COLOR_RGB2BGR)

        # Convert output to numpy
        output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        return intermediate_bgr, output_bgr, stage1_time, stage2_time

    def _calculate_metrics(self, current_frame: int, total_frames: int, start_time: float) -> dict:
        """Calculate current processing metrics."""
        elapsed = time.time() - start_time

        # Overall FPS
        overall_fps = current_frame / elapsed if elapsed > 0 else 0

        # Instantaneous FPS (based on recent frames)
        avg_frame_time = np.mean(self.total_times) if self.total_times else 0
        instant_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        # Stage-specific FPS
        stage1_fps = 1.0 / np.mean(self.stage1_times) if self.stage1_times else 0
        stage2_fps = 1.0 / np.mean(self.stage2_times) if self.stage2_times else 0

        # ETA
        frames_remaining = total_frames - current_frame
        eta_seconds = frames_remaining * avg_frame_time

        # Throughput (frames per second)
        throughput = instant_fps

        return {
            'elapsed': elapsed,
            'overall_fps': overall_fps,
            'instant_fps': instant_fps,
            'stage1_fps': stage1_fps,
            'stage2_fps': stage2_fps,
            'eta_seconds': eta_seconds,
            'throughput': throughput,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'stage1_time_ms': np.mean(self.stage1_times) * 1000 if self.stage1_times else 0,
            'stage2_time_ms': np.mean(self.stage2_times) * 1000 if self.stage2_times else 0,
        }

    def _calculate_final_metrics(self, total_frames: int, start_time: float) -> dict:
        """Calculate final processing metrics."""
        total_time = time.time() - start_time

        return {
            'total_frames': total_frames,
            'total_time': total_time,
            'average_fps': total_frames / total_time if total_time > 0 else 0,
            'total_stage1_time': sum(self.stage1_times) if self.stage1_times else 0,
            'total_stage2_time': sum(self.stage2_times) if self.stage2_times else 0,
            'avg_frame_time_ms': np.mean(self.total_times) * 1000 if self.total_times else 0,
            'avg_stage1_time_ms': np.mean(self.stage1_times) * 1000 if self.stage1_times else 0,
            'avg_stage2_time_ms': np.mean(self.stage2_times) * 1000 if self.stage2_times else 0,
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class VideoRestorationGUI(QMainWindow):
    """Main GUI window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Restoration - Two-Stage Pipeline")
        self.setGeometry(100, 100, 1600, 900)

        # State
        self.input_video_path: Optional[str] = None
        self.output_video_path: Optional[str] = None
        self.colorization_model: Optional[nn.Module] = None
        self.enhancement_model: Optional[nn.Module] = None
        self.colorization_model_path: Optional[str] = None
        self.enhancement_model_path: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = VideoProcessor()

        # Setup UI
        self._setup_ui()

        # Connect signals
        self.processor.progress_updated.connect(self._update_progress)
        self.processor.frame_processed.connect(self._display_frames)
        self.processor.finished.connect(self._processing_finished)
        self.processor.error.connect(self._processing_error)

    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Pipeline configuration
        pipeline_group = QGroupBox("Two-Stage Pipeline Configuration")
        pipeline_layout = QGridLayout()

        # Stage 1: Colorization
        pipeline_layout.addWidget(QLabel("Stage 1 - Colorization:"), 0, 0)
        self.colorization_enabled = QCheckBox("Enable")
        self.colorization_enabled.setChecked(False)
        pipeline_layout.addWidget(self.colorization_enabled, 0, 1)

        self.colorization_path_label = QLabel("No model loaded")
        pipeline_layout.addWidget(self.colorization_path_label, 0, 2)

        load_colorization_btn = QPushButton("Load Model")
        load_colorization_btn.clicked.connect(lambda: self._load_model(ProcessingStage.COLORIZATION))
        pipeline_layout.addWidget(load_colorization_btn, 0, 3)

        # Stage 2: Enhancement
        pipeline_layout.addWidget(QLabel("Stage 2 - Enhancement:"), 1, 0)
        self.enhancement_enabled = QCheckBox("Enable")
        self.enhancement_enabled.setChecked(False)
        pipeline_layout.addWidget(self.enhancement_enabled, 1, 1)

        self.enhancement_path_label = QLabel("No model loaded")
        pipeline_layout.addWidget(self.enhancement_path_label, 1, 2)

        load_enhancement_btn = QPushButton("Load Model")
        load_enhancement_btn.clicked.connect(lambda: self._load_model(ProcessingStage.ENHANCEMENT))
        pipeline_layout.addWidget(load_enhancement_btn, 1, 3)

        # Optimization options
        pipeline_layout.addWidget(QLabel("Use AMP:"), 2, 0)
        self.amp_checkbox = QCheckBox()
        self.amp_checkbox.setChecked(True)
        pipeline_layout.addWidget(self.amp_checkbox, 2, 1)

        pipeline_layout.addWidget(QLabel("Use Compile:"), 2, 2)
        self.compile_checkbox = QCheckBox()
        self.compile_checkbox.setChecked(False)
        pipeline_layout.addWidget(self.compile_checkbox, 2, 3)

        pipeline_group.setLayout(pipeline_layout)
        layout.addWidget(pipeline_group)

        # Video selection group
        video_group = QGroupBox("Video Files")
        video_layout = QGridLayout()

        self.input_video_label = QLabel("No input video selected")
        video_layout.addWidget(QLabel("Input:"), 0, 0)
        video_layout.addWidget(self.input_video_label, 0, 1)

        select_input_btn = QPushButton("Select Input Video")
        select_input_btn.clicked.connect(self._select_input_video)
        video_layout.addWidget(select_input_btn, 0, 2)

        self.output_video_label = QLabel("No output path set")
        video_layout.addWidget(QLabel("Output:"), 1, 0)
        video_layout.addWidget(self.output_video_label, 1, 1)

        select_output_btn = QPushButton("Select Output Path")
        select_output_btn.clicked.connect(self._select_output_path)
        video_layout.addWidget(select_output_btn, 1, 2)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # Preview group
        preview_group = QGroupBox("Real-time Preview")
        preview_layout = QHBoxLayout()

        # Input frame
        input_frame_layout = QVBoxLayout()
        input_frame_layout.addWidget(QLabel("Input Frame"))
        self.input_frame_label = QLabel()
        self.input_frame_label.setMinimumSize(480, 270)
        self.input_frame_label.setStyleSheet("background-color: black;")
        self.input_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        input_frame_layout.addWidget(self.input_frame_label)
        preview_layout.addLayout(input_frame_layout)

        # Intermediate frame (after stage 1)
        intermediate_frame_layout = QVBoxLayout()
        intermediate_frame_layout.addWidget(QLabel("After Stage 1"))
        self.intermediate_frame_label = QLabel()
        self.intermediate_frame_label.setMinimumSize(480, 270)
        self.intermediate_frame_label.setStyleSheet("background-color: black;")
        self.intermediate_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intermediate_frame_layout.addWidget(self.intermediate_frame_label)
        preview_layout.addLayout(intermediate_frame_layout)

        # Output frame
        output_frame_layout = QVBoxLayout()
        output_frame_layout.addWidget(QLabel("Final Output"))
        self.output_frame_label = QLabel()
        self.output_frame_label.setMinimumSize(480, 270)
        self.output_frame_label.setStyleSheet("background-color: black;")
        self.output_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        output_frame_layout.addWidget(self.output_frame_label)
        preview_layout.addLayout(output_frame_layout)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Progress group
        progress_group = QGroupBox("Processing Progress & Performance Metrics")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        # Performance metrics
        metrics_layout = QGridLayout()

        # Row 1: Basic stats
        self.frame_count_label = QLabel("Frames: 0 / 0")
        self.elapsed_label = QLabel("Elapsed: 00:00:00")
        self.eta_label = QLabel("ETA: 00:00:00")
        metrics_layout.addWidget(self.frame_count_label, 0, 0)
        metrics_layout.addWidget(self.elapsed_label, 0, 1)
        metrics_layout.addWidget(self.eta_label, 0, 2)

        # Row 2: FPS metrics
        self.overall_fps_label = QLabel("Overall FPS: 0.0")
        self.instant_fps_label = QLabel("Current FPS: 0.0")
        self.throughput_label = QLabel("Throughput: 0.0 fps")
        metrics_layout.addWidget(self.overall_fps_label, 1, 0)
        metrics_layout.addWidget(self.instant_fps_label, 1, 1)
        metrics_layout.addWidget(self.throughput_label, 1, 2)

        # Row 3: Timing breakdown
        self.frame_time_label = QLabel("Frame: 0.0 ms")
        self.stage1_time_label = QLabel("Stage 1: 0.0 ms")
        self.stage2_time_label = QLabel("Stage 2: 0.0 ms")
        metrics_layout.addWidget(self.frame_time_label, 2, 0)
        metrics_layout.addWidget(self.stage1_time_label, 2, 1)
        metrics_layout.addWidget(self.stage2_time_label, 2, 2)

        # Row 4: Stage FPS
        self.stage1_fps_label = QLabel("Stage 1 FPS: 0.0")
        self.stage2_fps_label = QLabel("Stage 2 FPS: 0.0")
        metrics_layout.addWidget(self.stage1_fps_label, 3, 0)
        metrics_layout.addWidget(self.stage2_fps_label, 3, 1)

        progress_layout.addLayout(metrics_layout)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Control buttons
        control_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self._start_processing)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        layout.addLayout(control_layout)

        # Status
        self.status_label = QLabel(f"Ready. Device: {self.device}")
        layout.addWidget(self.status_label)

    def _load_model(self, stage: ProcessingStage):
        """Load a model for the specified stage."""
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {stage.value.title()} Model Checkpoint",
            "./experiments",
            "Model Files (*.pt *.pth);;All Files (*)"
        )

        if not model_path:
            return

        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Create model based on stage
            if stage == ProcessingStage.COLORIZATION:
                model = SimpleUNetColorization(
                    in_channels=3,
                    out_channels=3,
                    base_channels=96,
                    num_frames=1,
                )
            else:  # ENHANCEMENT
                # Try to infer model type from checkpoint
                # Default to SimpleUNet for now
                model = SimpleUNet(
                    in_channels=3,
                    out_channels=3,
                    base_channels=64,
                    num_frames=1,
                )

            # Load weights
            if "generator_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["generator_state_dict"])
            elif "ema_state_dict" in checkpoint:
                # Try EMA weights
                from src.utils import EMA
                ema = EMA(model, decay=0.999)
                ema.load_state_dict(checkpoint["ema_state_dict"])
                ema.apply_shadow()
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(self.device)
            model.eval()

            # Apply optimizations
            if self.compile_checkbox.isChecked() and hasattr(torch, 'compile'):
                self.status_label.setText(f"Compiling {stage.value} model... (this may take a minute)")
                QApplication.processEvents()
                model = torch.compile(model)

            # Store model
            if stage == ProcessingStage.COLORIZATION:
                self.colorization_model = model
                self.colorization_model_path = model_path
                self.colorization_path_label.setText(Path(model_path).name)
                self.colorization_enabled.setChecked(True)
            else:
                self.enhancement_model = model
                self.enhancement_model_path = model_path
                self.enhancement_path_label.setText(Path(model_path).name)
                self.enhancement_enabled.setChecked(True)

            self.status_label.setText(f"{stage.value.title()} model loaded: {Path(model_path).name}")
            self._update_start_button()

        except Exception as e:
            import traceback
            self.status_label.setText(f"Error loading {stage.value} model: {str(e)}")
            print(traceback.format_exc())

    def _select_input_video(self):
        """Select input video file."""
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )

        if video_path:
            self.input_video_path = video_path
            self.input_video_label.setText(Path(video_path).name)

            # Auto-suggest output path
            input_path = Path(video_path)
            output_path = input_path.parent / f"{input_path.stem}_restored{input_path.suffix}"
            self.output_video_path = str(output_path)
            self.output_video_label.setText(output_path.name)

            self._update_start_button()

    def _select_output_path(self):
        """Select output video path."""
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output Video Path",
            self.output_video_path or "",
            "Video Files (*.mp4 *.avi);;All Files (*)"
        )

        if output_path:
            self.output_video_path = output_path
            self.output_video_label.setText(Path(output_path).name)
            self._update_start_button()

    def _update_start_button(self):
        """Update start button state."""
        # Need at least one model enabled and video paths set
        has_model = (
            (self.colorization_enabled.isChecked() and self.colorization_model is not None) or
            (self.enhancement_enabled.isChecked() and self.enhancement_model is not None)
        )
        can_start = (
            has_model and
            self.input_video_path is not None and
            self.output_video_path is not None
        )
        self.start_btn.setEnabled(can_start)

    def _start_processing(self):
        """Start video processing."""
        # Setup processor with enabled models only
        self.processor.input_path = self.input_video_path
        self.processor.output_path = self.output_video_path
        self.processor.colorization_model = self.colorization_model if self.colorization_enabled.isChecked() else None
        self.processor.enhancement_model = self.enhancement_model if self.enhancement_enabled.isChecked() else None
        self.processor.device = self.device
        self.processor.use_amp = self.amp_checkbox.isChecked()

        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Show pipeline info
        pipeline_info = []
        if self.processor.colorization_model:
            pipeline_info.append("Colorization")
        if self.processor.enhancement_model:
            pipeline_info.append("Enhancement")
        self.status_label.setText(f"Processing with: {' → '.join(pipeline_info)}")

        # Start processing
        self.processor.start()

    def _stop_processing(self):
        """Stop video processing."""
        self.processor.stop()
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopping...")

    def _update_progress(self, current: int, total: int, metrics: dict):
        """Update progress display."""
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)

        # Update labels
        self.frame_count_label.setText(f"Frames: {current} / {total}")
        self.elapsed_label.setText(f"Elapsed: {self._format_time(metrics['elapsed'])}")
        self.eta_label.setText(f"ETA: {self._format_time(metrics['eta_seconds'])}")

        self.overall_fps_label.setText(f"Overall FPS: {metrics['overall_fps']:.1f}")
        self.instant_fps_label.setText(f"Current FPS: {metrics['instant_fps']:.1f}")
        self.throughput_label.setText(f"Throughput: {metrics['throughput']:.1f} fps")

        self.frame_time_label.setText(f"Frame: {metrics['avg_frame_time_ms']:.1f} ms")
        self.stage1_time_label.setText(f"Stage 1: {metrics['stage1_time_ms']:.1f} ms")
        self.stage2_time_label.setText(f"Stage 2: {metrics['stage2_time_ms']:.1f} ms")

        self.stage1_fps_label.setText(f"Stage 1 FPS: {metrics['stage1_fps']:.1f}")
        self.stage2_fps_label.setText(f"Stage 2 FPS: {metrics['stage2_fps']:.1f}")

    def _display_frames(self, input_frame: np.ndarray, intermediate_frame: np.ndarray, output_frame: np.ndarray):
        """Display input, intermediate, and output frames."""
        # Resize for display
        display_width = 480
        display_height = 270

        input_resized = cv2.resize(input_frame, (display_width, display_height))
        intermediate_resized = cv2.resize(intermediate_frame, (display_width, display_height))
        output_resized = cv2.resize(output_frame, (display_width, display_height))

        # Convert to QPixmap
        input_pixmap = self._numpy_to_pixmap(input_resized)
        intermediate_pixmap = self._numpy_to_pixmap(intermediate_resized)
        output_pixmap = self._numpy_to_pixmap(output_resized)

        # Display
        self.input_frame_label.setPixmap(input_pixmap)
        self.intermediate_frame_label.setPixmap(intermediate_pixmap)
        self.output_frame_label.setPixmap(output_pixmap)

    def _numpy_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap."""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        return QPixmap.fromImage(q_image)

    def _processing_finished(self, output_path: str, metrics: dict):
        """Handle processing completion."""
        # Format summary
        summary = f"""✓ Processing Complete!

Output: {Path(output_path).name}
Total Frames: {metrics['total_frames']}
Total Time: {self._format_time(metrics['total_time'])}
Average FPS: {metrics['average_fps']:.2f}

Performance Breakdown:
• Average frame time: {metrics['avg_frame_time_ms']:.1f} ms
• Stage 1 (Colorization): {metrics['avg_stage1_time_ms']:.1f} ms
• Stage 2 (Enhancement): {metrics['avg_stage2_time_ms']:.1f} ms

Throughput: {metrics['average_fps']:.2f} frames/second"""

        self.status_label.setText(summary.replace('\n', ' | '))
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        print(summary)

    def _processing_error(self, error_message: str):
        """Handle processing error."""
        self.status_label.setText(f"Error: {error_message}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    """Run the application."""
    app = QApplication(sys.argv)
    window = VideoRestorationGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
