"""
Unified configuration system for multi-task training.

Provides dataclass-based configuration with validation and defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    root_dir: str | Path
    """Root directory containing images or videos"""

    image_size: tuple[int, int] = (512, 512)
    """Target image size (height, width)"""

    batch_size: int = 4
    """Batch size for training"""

    num_workers: int = 4
    """Number of data loading workers"""

    limit: int | None = None
    """Limit number of images/videos (None = no limit)"""


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    model_type: Literal["hat_simple_s", "hat_simple_m", "hat_simple_l"] = "hat_simple_s"
    """Model architecture variant"""

    in_channels: int = 3
    """Number of input channels"""

    out_channels: int = 3
    """Number of output channels"""

    num_frames: int = 1
    """Number of temporal frames (1 for images, >1 for videos)"""


@dataclass
class TaskConfig:
    """Multi-task configuration."""

    enable_super_resolution: bool = False
    """Enable super-resolution task"""

    enable_artifact_removal: bool = False
    """Enable artifact removal task"""

    enable_colorization: bool = True
    """Enable colorization task"""

    # Degradation probabilities
    downscale_prob: float = 1.0
    """Probability of applying downscaling"""

    noise_prob: float = 0.7
    """Probability of adding noise"""

    blur_prob: float = 0.5
    """Probability of adding blur"""

    compression_prob: float = 0.8
    """Probability of JPEG compression"""

    grayscale_prob: float = 1.0
    """Probability of grayscale conversion"""

    # Degradation parameters
    sr_scale_factor: int = 4
    """Super-resolution scale factor (downscale then upscale)"""

    noise_sigma: float = 0.05
    """Gaussian noise sigma"""

    jpeg_quality: int = 70
    """JPEG compression quality (1-100)"""

    blur_kernel_size: int = 3
    """Gaussian blur kernel size"""

    blur_sigma: float = 0.5
    """Gaussian blur sigma"""


@dataclass
class LossConfig:
    """Loss function configuration."""

    # Loss weights
    rgb_weight: float = 1.0
    """Weight for RGB reconstruction loss"""

    perceptual_weight: float = 0.5
    """Weight for perceptual loss"""

    ab_weight: float = 2.0
    """Weight for AB chrominance loss (colorization)"""

    # GAN settings
    gan_weight: float = 0.5
    """Weight for adversarial loss"""

    content_weight: float = 1.0
    """Weight for content loss in GAN training"""

    gan_loss_type: Literal["vanilla", "lsgan", "hinge"] = "lsgan"
    """Type of GAN loss"""

    # Perceptual network
    perceptual_net: Literal["alex", "vgg", "squeeze"] = "alex"
    """Network for perceptual loss"""


@dataclass
class DiscriminatorConfig:
    """Discriminator configuration."""

    num_filters: int = 64
    """Base number of filters"""

    num_layers: int = 3
    """Number of discriminator layers"""

    use_spectral_norm: bool = True
    """Use spectral normalization"""

    # Balancing parameters
    learning_rate_ratio: float = 5.0
    """Generator LR / Discriminator LR ratio (e.g., 5.0 = disc is 5x slower)"""

    update_frequency: int = 1
    """Train discriminator every N generator updates"""

    use_adaptive: bool = True
    """Enable adaptive discriminator updates"""

    advantage_threshold: float = 0.7
    """Skip disc update if Real-Fake gap > this value"""


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimizer settings
    learning_rate: float = 1e-4
    """Generator learning rate"""

    weight_decay: float = 1e-5
    """Weight decay for optimizer"""

    warmup_steps: int = 1000
    """Number of warmup steps for learning rate"""

    # Training phases (NoGAN strategy)
    pretrain_epochs: int = 10
    """Number of pretrain epochs (generator only, content loss)"""

    critic_epochs: int = 2
    """Number of critic epochs per cycle (discriminator only)"""

    gan_epochs: int = 10
    """Number of GAN epochs per cycle (both networks)"""

    num_cycles: int = 5
    """Number of NoGAN training cycles"""

    # Training settings
    use_amp: bool = True
    """Use automatic mixed precision"""

    gradient_clip: float = 1.0
    """Gradient clipping value"""

    gradient_accumulation_steps: int = 2
    """Number of gradient accumulation steps"""

    use_ema: bool = True
    """Use exponential moving average"""

    ema_decay: float = 0.999
    """EMA decay rate"""

    use_compile: bool = False
    """Use torch.compile for optimization"""

    multi_gpu: bool = False
    """Use DataParallel for multi-GPU"""

    # Logging and checkpointing
    log_interval: int = 100
    """Steps between metric logging"""

    checkpoint_interval: int = 5000
    """Steps between checkpoints"""

    image_log_interval: int = 100
    """Steps between image logging"""


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    experiment_id: str = "multitask_gan_001"
    """Unique experiment identifier"""

    experiment_dir: str | Path = "./experiments"
    """Root directory for experiments"""

    device: str = "cuda:0"
    """Device to train on"""

    # Sub-configurations
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert paths
        self.experiment_dir = Path(self.experiment_dir)
        self.dataset.root_dir = Path(self.dataset.root_dir)

        # Validate at least one task is enabled
        if not any([
            self.tasks.enable_super_resolution,
            self.tasks.enable_artifact_removal,
            self.tasks.enable_colorization,
        ]):
            raise ValueError("At least one task must be enabled")

    def get_discriminator_lr(self) -> float:
        """Calculate discriminator learning rate from ratio."""
        return self.training.learning_rate / self.discriminator.learning_rate_ratio

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        from dataclasses import asdict
        return asdict(self)
