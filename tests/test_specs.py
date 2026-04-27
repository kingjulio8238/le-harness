"""Tests for the declarative ModelSpec sketch (S1)."""

from __future__ import annotations

import pytest

from harness.dims import LEWM_DEFAULT_ACTION_DIM, LEWM_EMBED_DIM
from harness.specs import (
    LEWM_SPEC,
    ModelSpec,
    OutputKind,
    RunnerKind,
    build_lewm_spec,
)


class TestLeWMSpec:
    def test_module_handle_is_modelspec(self):
        assert isinstance(LEWM_SPEC, ModelSpec)

    def test_basic_fields(self):
        assert LEWM_SPEC.name == "lewm"
        assert LEWM_SPEC.runner is RunnerKind.DETERMINISTIC
        assert LEWM_SPEC.output is OutputKind.EMBEDDING
        assert LEWM_SPEC.embed_dim == LEWM_EMBED_DIM
        assert LEWM_SPEC.action_dim == LEWM_DEFAULT_ACTION_DIM

    def test_no_cache(self):
        assert LEWM_SPEC.cache is not None
        assert LEWM_SPEC.cache.layout == "none"

    def test_no_scheduler(self):
        # LeWM is deterministic — no diffusion scheduler.
        assert LEWM_SPEC.scheduler is None

    def test_encoders(self):
        assert "image_goal" in LEWM_SPEC.encoders
        assert "vlm_goal" in LEWM_SPEC.encoders
        assert "text_goal" in LEWM_SPEC.encoders
        for enc in LEWM_SPEC.encoders.values():
            assert enc.out_dim == LEWM_EMBED_DIM

    def test_inference_patches_present(self):
        assert "compile_predictor" in LEWM_SPEC.inference_patches
        assert "compile_encoder" in LEWM_SPEC.inference_patches
        assert "buffer_rollout" in LEWM_SPEC.inference_patches

    def test_build_with_overrides(self):
        spec = build_lewm_spec(embed_dim=384, action_dim=4)
        assert spec.embed_dim == 384
        assert spec.action_dim == 4
        for enc in spec.encoders.values():
            assert enc.out_dim == 384
