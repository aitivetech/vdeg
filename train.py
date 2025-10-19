"""
Unified training script for multi-task GAN restoration.

Supports colorization, super-resolution, and artifact removal with
balanced GAN training and NoGAN strategy.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    ExperimentConfig,
    DatasetConfig,
    ModelConfig,
    TaskConfig,
    LossConfig,
    DiscriminatorConfig,
    TrainingConfig,
)
from src.datasets import ImageRestorationDataset
from src.degradations import (
    GaussianNoise,
    JPEGCompression,
    GaussianBlur,
    Grayscale,
    SuperResolutionDegradation,
    MultiTaskDegradation,
)
from src.losses import MultiTaskLoss, GANLoss
from src.models import PatchDiscriminator, hat_simple_s, hat_simple_m, hat_simple_l
from src.training import BalancedMultiTaskGANTrainer


def create_config() -> ExperimentConfig:
    """
    Create experiment configuration.

    Modify this function to customize your training setup.
    """
    config = ExperimentConfig(
        experiment_id="multitask_colorization_001",
        experiment_dir="./experiments",
        device="cuda:0" if torch.cuda.is_available() else "cpu",

        dataset=DatasetConfig(
            root_dir="/media/bglueck/Data/datasets/laion_high/laion-output",
            image_size=(256, 256),
            batch_size=16,
            num_workers=4,
            limit=25000,  # Set to None for full dataset
        ),

        model=ModelConfig(
            model_type="hat_simple_m",
            in_channels=3,
            out_channels=3,
            num_frames=1,
        ),

        tasks=TaskConfig(
            # Enable/disable tasks
            enable_colorization=True,
            enable_super_resolution=False,
            enable_artifact_removal=False,

            # Degradation probabilities
            grayscale_prob=1.0,
            downscale_prob=1.0,
            noise_prob=0.7,
            blur_prob=0.5,
            compression_prob=0.8,

            # Degradation parameters
            noise_sigma=0.05,
            jpeg_quality=70,
            blur_kernel_size=3,
            blur_sigma=0.5,
            sr_scale_factor=4
        ),

        loss=LossConfig(
            rgb_weight=1.0,
            perceptual_weight=0.5,
            ab_weight=2.0,
            gan_weight=0.5,
            content_weight=1.0,
            gan_loss_type="lsgan",
            perceptual_net="alex",
        ),

        discriminator=DiscriminatorConfig(
            num_filters=64,
            num_layers=3,
            use_spectral_norm=True,
            learning_rate_ratio=5.0,  # Disc LR is 5x slower than gen
            update_frequency=1,
            use_adaptive=True,
            advantage_threshold=0.7,
        ),

        training=TrainingConfig(
            learning_rate=1e-4,
            weight_decay=1e-5,
            warmup_steps=1000,

            # NoGAN training phases
            #pretrain_epochs=10,
            #critic_epochs=2,
            #gan_epochs=10,
            #num_cycles=5,

            pretrain_epochs=5,
            critic_epochs=1,
            gan_epochs=5,
            num_cycles=3,

            # Training settings
            use_amp=True,
            gradient_clip=1.0,
            gradient_accumulation_steps=2,
            use_ema=True,
            ema_decay=0.999,
            use_compile=False,
            multi_gpu=False,

            # Logging
            log_interval=100,
            checkpoint_interval=5000,
            image_log_interval=100,
        ),
    )

    return config


def create_degradation_pipeline(config: ExperimentConfig) -> MultiTaskDegradation:
    """Create degradation pipeline from config."""
    tasks = config.tasks

    # Create individual degradations
    noise_fn = GaussianNoise(sigma=tasks.noise_sigma)
    blur_fn = GaussianBlur(kernel_size=tasks.blur_kernel_size, sigma=tasks.blur_sigma)
    compression_fn = JPEGCompression(quality=tasks.jpeg_quality)
    grayscale_fn = Grayscale()
    downscale_fn = SuperResolutionDegradation(
        scale_factor=config.tasks.sr_scale_factor,
        mode="bicubic",
    )

    # Combine into multi-task pipeline
    degradation = MultiTaskDegradation(
        downscale_fn=downscale_fn if tasks.enable_super_resolution else None,
        noise_fn=noise_fn if tasks.enable_artifact_removal else None,
        blur_fn=blur_fn if tasks.enable_artifact_removal else None,
        compression_fn=compression_fn if tasks.enable_artifact_removal else None,
        grayscale_fn=grayscale_fn if tasks.enable_colorization else None,
        downscale_prob=tasks.downscale_prob,
        noise_prob=tasks.noise_prob,
        blur_prob=tasks.blur_prob,
        compression_prob=tasks.compression_prob,
        grayscale_prob=tasks.grayscale_prob,
    )

    return degradation


def create_model(config: ExperimentConfig) -> nn.Module:
    """Create generator model from config."""
    model_kwargs = {
        'in_channels': config.model.in_channels,
        'out_channels': config.model.out_channels,
        'num_frames': config.model.num_frames,
    }

    if config.model.model_type == "hat_simple_s":
        return hat_simple_s(**model_kwargs)
    elif config.model.model_type == "hat_simple_m":
        return hat_simple_m(**model_kwargs)
    elif config.model.model_type == "hat_simple_l":
        return hat_simple_l(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")


def main():
    """Main training function."""
    # Create configuration
    config = create_config()

    print("=" * 80)
    print("MULTI-TASK GAN TRAINING")
    print("=" * 80)
    print(f"\nExperiment ID: {config.experiment_id}")
    print(f"Device: {config.device}")
    print("\n📋 Enabled tasks:")
    print(f"  • Super-resolution: {config.tasks.enable_super_resolution} "
          f"({config.tasks.sr_scale_factor}x)" if config.tasks.enable_super_resolution else "  • Super-resolution: False")
    print(f"  • Artifact removal: {config.tasks.enable_artifact_removal}")
    print(f"  • Colorization: {config.tasks.enable_colorization}")
    print("\n🔄 NoGAN Strategy:")
    print(f"  • Pretrain: {config.training.pretrain_epochs} epochs")
    print(f"  • Critic: {config.training.critic_epochs} epochs per cycle")
    print(f"  • GAN: {config.training.gan_epochs} epochs per cycle")
    print(f"  • Cycles: {config.training.num_cycles}")
    print("\n🔧 Discriminator Balance:")
    print(f"  • LR ratio: 1:{config.discriminator.learning_rate_ratio:.1f} "
          f"(disc is slower)")
    print(f"  • Adaptive updates: {config.discriminator.use_adaptive}")
    print("=" * 80)

    # Create degradation pipeline
    degradation = create_degradation_pipeline(config)

    # Create dataset
    print(f"\n📂 Loading dataset from: {config.dataset.root_dir}")
    dataset = ImageRestorationDataset(
        root_dir=config.dataset.root_dir,
        target_size=config.dataset.image_size,
        degradation_fn=degradation,
        limit=config.dataset.limit,
    )
    print(f"Dataset size: {len(dataset)} images")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Batches per epoch: {len(dataloader)}")

    # Create generator
    print(f"\n🤖 Initializing generator: {config.model.model_type}")
    generator = create_model(config)
    gen_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {gen_params:,}")

    # Create discriminator
    print(f"\n🛡️  Initializing discriminator: PatchGAN")
    discriminator = PatchDiscriminator(
        input_channels=6,  # Input (degraded) + output (restored)
        num_filters=config.discriminator.num_filters,
        num_layers=config.discriminator.num_layers,
        use_spectral_norm=config.discriminator.use_spectral_norm,
    )
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator parameters: {disc_params:,}")

    # Create content loss
    content_loss = MultiTaskLoss(
        rgb_weight=config.loss.rgb_weight,
        perceptual_weight=config.loss.perceptual_weight,
        ab_weight=config.loss.ab_weight,
        perceptual_net=config.loss.perceptual_net,
        device=config.device,
        enable_colorization=config.tasks.enable_colorization,
        enable_super_resolution=(
            config.tasks.enable_super_resolution or
            config.tasks.enable_artifact_removal
        ),
    )

    # Create optimizers
    gen_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999),
    )

    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.get_discriminator_lr(),
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999),
    )

    # Create learning rate schedulers
    total_pretrain_steps = config.training.pretrain_epochs * len(dataloader)
    total_gan_steps = (config.training.gan_epochs * len(dataloader) *
                      config.training.num_cycles)

    def gen_lr_lambda(step):
        if step < config.training.warmup_steps:
            return step / config.training.warmup_steps
        else:
            total_steps = total_pretrain_steps + total_gan_steps
            progress = (step - config.training.warmup_steps) / (total_steps - config.training.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    def disc_lr_lambda(step):
        if step < config.training.warmup_steps:
            return step / config.training.warmup_steps
        else:
            progress = (step - config.training.warmup_steps) / (total_gan_steps - config.training.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, gen_lr_lambda)
    disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, disc_lr_lambda)

    # Create trainer
    trainer = BalancedMultiTaskGANTrainer(
        generator=generator,
        discriminator=discriminator,
        content_loss=content_loss,
        generator_optimizer=gen_optimizer,
        discriminator_optimizer=disc_optimizer,
        device=config.device,
        experiment_dir=config.experiment_dir,
        experiment_id=config.experiment_id,
        gan_loss_type=config.loss.gan_loss_type,
        gan_weight=config.loss.gan_weight,
        content_weight=config.loss.content_weight,
        disc_update_frequency=config.discriminator.update_frequency,
        use_adaptive_disc=config.discriminator.use_adaptive,
        disc_advantage_threshold=config.discriminator.advantage_threshold,
        use_amp=config.training.use_amp,
        gradient_clip=config.training.gradient_clip,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        use_ema=config.training.use_ema,
        ema_decay=config.training.ema_decay,
        use_compile=config.training.use_compile,
        multi_gpu=config.training.multi_gpu,
        log_interval=config.training.log_interval,
        checkpoint_interval=config.training.checkpoint_interval,
        image_log_interval=config.training.image_log_interval,
    )

    trainer.gen_scheduler = gen_scheduler
    trainer.disc_scheduler = disc_scheduler

    # Log experiment configuration
    trainer.log_experiment_info(**config.to_dict())

    print("\n🚀 Starting training...")
    print("=" * 80)

    # Phase 1: PRETRAIN
    print("\n" + "=" * 80)
    print("PHASE 1: PRETRAIN GENERATOR (Content Loss Only)")
    print("=" * 80)
    trainer.set_phase("pretrain")

    for epoch in range(config.training.pretrain_epochs):
        epoch_metrics = trainer.train_epoch(dataloader, epoch)
        print(f"\n✓ Pretrain Epoch {epoch} Summary:")
        print(f"  Loss: {epoch_metrics['loss']:.6f}")
        print(f"  PSNR: {epoch_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {epoch_metrics['ssim']:.4f}")
        print("-" * 80)

    # NoGAN Cycles
    for cycle in range(config.training.num_cycles):
        print("\n" + "=" * 80)
        print(f"CYCLE {cycle + 1}/{config.training.num_cycles}")
        print("=" * 80)

        # Phase 2: CRITIC
        print("\n" + "=" * 80)
        print("PHASE 2: PRETRAIN CRITIC (Discriminator Only)")
        print("=" * 80)
        trainer.set_phase("critic")

        for epoch in range(config.training.critic_epochs):
            epoch_metrics = trainer.train_epoch(dataloader, epoch)
            print(f"\n✓ Critic Epoch {epoch} Summary:")
            print(f"  Disc Loss: {epoch_metrics['disc_loss']:.6f}")
            print(f"  Disc Real: {epoch_metrics['disc_real']:.4f}")
            print(f"  Disc Fake: {epoch_metrics['disc_fake']:.4f}")
            print("-" * 80)

        # Phase 3: BALANCED GAN
        print("\n" + "=" * 80)
        print("PHASE 3: BALANCED ADVERSARIAL TRAINING")
        print("=" * 80)
        trainer.set_phase("gan")

        for epoch in range(config.training.gan_epochs):
            epoch_metrics = trainer.train_epoch(dataloader, epoch)
            advantage = epoch_metrics.get('disc_advantage', 0)
            disc_updates = epoch_metrics.get('disc_updates', 0)
            disc_skips = epoch_metrics.get('disc_skips', 0)

            print(f"\n✓ GAN Epoch {epoch} Summary:")
            print(f"  Gen Loss: {epoch_metrics['gen_loss']:.6f}")
            print(f"  Disc Loss: {epoch_metrics['disc_loss']:.6f}")
            print(f"  Disc Advantage: {advantage:.4f} (Real-Fake gap)")
            print(f"  Disc Updates: {disc_updates}, Skips: {disc_skips}")
            print(f"  PSNR: {epoch_metrics['psnr']:.2f} dB")
            print(f"  SSIM: {epoch_metrics['ssim']:.4f}")
            print("-" * 80)

    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print(f"📁 Checkpoints: {trainer.checkpoint_dir}")
    print(f"📊 Logs: {trainer.log_dir}")
    print(f"📦 ONNX exports: {trainer.export_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
