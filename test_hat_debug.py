"""Debug HAT model."""

import torch
from src.models.hat import PatchEmbed

# Test patch embed
patch_embed = PatchEmbed(
    img_size=64,
    patch_size=1,
    in_chans=3,
    embed_dim=64
)

x = torch.randn(1, 3, 64, 64)
print(f"Input shape: {x.shape}")

out = patch_embed(x)
print(f"PatchEmbed output shape: {out.shape}")
print(f"Expected: (1, 4096, 64) = (B, H*W, embed_dim)")
