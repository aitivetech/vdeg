"""Model architectures."""

from .simple_unet import SimpleUNet
from .simple_unet_colorization import SimpleUNetColorization
from .hat import HAT, hat_s, hat_m, hat_l
from .hat_simple import HATSimple, hat_simple_s, hat_simple_m, hat_simple_l
from .discriminator import PatchDiscriminator, MultiScaleDiscriminator

__all__ = [
    "SimpleUNet",
    "SimpleUNetColorization",
    "HAT", "hat_s", "hat_m", "hat_l",
    "HATSimple", "hat_simple_s", "hat_simple_m", "hat_simple_l",
    "PatchDiscriminator", "MultiScaleDiscriminator"
]
