"""Declarative spec for LeWM (the existing JEPA + CEM stack).

This is the first declarative spec — it describes today's LeWM exactly as
the pipeline runs it. Pre-fork it serves as documentation and a single
source of truth; post-fork the imagination engine consumes it as its only
model-specific entry point.
"""

from __future__ import annotations

from harness.dims import LEWM_DEFAULT_ACTION_DIM, LEWM_EMBED_DIM
from harness.specs.base import (
    CacheSpec,
    EncoderSpec,
    ModelSpec,
    OutputKind,
    RunnerKind,
)


def build_lewm_spec(
    *,
    embed_dim: int = LEWM_EMBED_DIM,
    action_dim: int = LEWM_DEFAULT_ACTION_DIM,
) -> ModelSpec:
    """Build a LeWM ModelSpec.

    Defaults match the production checkpoints (DINOv2-S/14 + 2-D action).
    Override action_dim per task (PushT=2, TwoRoom=2, Cube=4, Reacher=2).
    """
    return ModelSpec(
        name="lewm",
        runner=RunnerKind.DETERMINISTIC,
        output=OutputKind.EMBEDDING,
        embed_dim=embed_dim,
        action_dim=action_dim,
        encoders={
            # Image goal: encoded by the same ViT used at planning time.
            "image_goal": EncoderSpec(
                name="image_goal",
                out_dim=embed_dim,
                # factory deliberately None pre-fork — pipeline constructs it.
                factory=None,
            ),
            # VLM-projected goal: SigLIP / T5 / Eagle / PaliGemma → embed_dim.
            "vlm_goal": EncoderSpec(
                name="vlm_goal",
                out_dim=embed_dim,
                factory=None,
            ),
            # Text goal: coord parsing or CLIP → embed_dim.
            "text_goal": EncoderSpec(
                name="text_goal",
                out_dim=embed_dim,
                factory=None,
            ),
        },
        # JEPA uses a fixed history buffer, not a KV cache.
        cache=CacheSpec(layout="none"),
        # No diffusion solver — this is a deterministic predictor.
        scheduler=None,
        # LeWM doesn't currently quantize; placeholder for future use.
        quant_blacklist=[],
        # Inference-time patches applied today.
        inference_patches=[
            "compile_predictor",
            "compile_encoder",
            "buffer_rollout",
        ],
        metadata={
            "history_size": 3,
            "default_horizon": 5,
            "default_num_samples": 128,
            "default_n_steps": 15,
            "default_topk": 25,
        },
        builder=None,  # post-fork: returns a JEPA module from cfg.
    )


# Convenience module-level handle.
LEWM_SPEC = build_lewm_spec()
