"""
Single source of truth for LeWM dimensional constants.

These values are LeWM-specific. The future imagination engine uses ModelSpec
fields instead (see harness/specs/lewm_spec.py); this module is the bridge
that lets pre-fork code stay tidy without spreading magic numbers.
"""

# DINOv2 ViT-S/14 CLS token width — what LeWM trains and plans in.
LEWM_EMBED_DIM: int = 192

# Common project default for action dim. Concrete value comes from the
# model's action_encoder.patch_embed.in_channels at runtime; this default
# is only used by tests/benchmarks that don't load a real model.
LEWM_DEFAULT_ACTION_DIM: int = 2

__all__ = ["LEWM_EMBED_DIM", "LEWM_DEFAULT_ACTION_DIM"]
