"""Tests for harness/compiled_inference.py — locks in the behavior of
optimize_model and the buffer-pre-allocated rollout patch.

Key parity check: the patched optimized_rollout must produce numerically
identical trajectories to JEPA.rollout for the same inputs. M3 (rollout
consolidation) merges these, and these tests guard the merge.
"""

from __future__ import annotations

import copy

import pytest
import torch

from harness.compiled_inference import optimize_model, _patch_rollout_with_buffers
from harness.rollout import JepaAdapter, ModelAdapter, rollout_buffered
from tests._engine_fixtures import make_tiny_jepa


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return make_tiny_jepa(embed_dim=16, action_dim=2, action_emb_dim=4)


# ==================== _patch_rollout_with_buffers ====================


class TestRolloutPatch:
    def test_patch_replaces_rollout(self, tiny_model):
        original = tiny_model.rollout
        _patch_rollout_with_buffers(tiny_model)
        assert tiny_model.rollout is not original

    def test_patched_rollout_output_shape(self, tiny_model):
        _patch_rollout_with_buffers(tiny_model)
        pixels = torch.rand(1, 4, 1, 3, 64, 64)
        actions = torch.randn(1, 4, 5, 2)
        info = {"pixels": pixels, "action": actions[:, :, :1]}
        info = tiny_model.rollout(info, actions, history_size=3)
        # Same shape contract as JEPA.rollout: B, S, T+1, D
        assert info["predicted_emb"].shape == (1, 4, 6, 16)

    def test_patched_rollout_matches_legacy(self, tiny_model):
        """The buffer-based optimized_rollout must produce numerically equal
        trajectories to JEPA.rollout. This is the parity guarantee that M3
        depends on.
        """
        torch.manual_seed(42)
        ref_model = copy.deepcopy(tiny_model)        # legacy rollout
        opt_model = copy.deepcopy(tiny_model)        # patched rollout
        _patch_rollout_with_buffers(opt_model)

        pixels = torch.rand(1, 4, 1, 3, 64, 64)
        actions = torch.randn(1, 4, 5, 2)

        ref_out = ref_model.rollout(
            {"pixels": pixels.clone(), "action": actions[:, :, :1].clone()},
            actions.clone(),
            history_size=3,
        )["predicted_emb"]
        opt_out = opt_model.rollout(
            {"pixels": pixels.clone(), "action": actions[:, :, :1].clone()},
            actions.clone(),
            history_size=3,
        )["predicted_emb"]

        # Allow minor float differences from buffer indexing vs cat ordering
        assert torch.allclose(ref_out, opt_out, atol=1e-5, rtol=1e-5), (
            f"max abs diff = {(ref_out - opt_out).abs().max():.6e}"
        )

    @pytest.mark.parametrize("history_size", [1, 2, 3, 5])
    def test_patched_rollout_history_size_parity(self, tiny_model, history_size):
        torch.manual_seed(0)
        ref_model = copy.deepcopy(tiny_model)
        opt_model = copy.deepcopy(tiny_model)
        _patch_rollout_with_buffers(opt_model)

        pixels = torch.rand(1, 2, 1, 3, 64, 64)
        actions = torch.randn(1, 2, 4, 2)

        ref = ref_model.rollout(
            {"pixels": pixels.clone(), "action": actions[:, :, :1].clone()},
            actions.clone(),
            history_size=history_size,
        )["predicted_emb"]
        opt = opt_model.rollout(
            {"pixels": pixels.clone(), "action": actions[:, :, :1].clone()},
            actions.clone(),
            history_size=history_size,
        )["predicted_emb"]

        assert torch.allclose(ref, opt, atol=1e-5)


# ==================== optimize_model ====================


class TestOptimizeModel:
    def test_optimize_model_returns_same_object(self, tiny_model):
        out = optimize_model(
            tiny_model,
            compile_predictor=False,
            compile_encoder=False,
        )
        assert out is tiny_model, "optimize_model should mutate in place"

    def test_optimize_model_patches_rollout(self, tiny_model):
        original_rollout = tiny_model.rollout
        optimize_model(
            tiny_model,
            compile_predictor=False,
            compile_encoder=False,
        )
        assert tiny_model.rollout is not original_rollout

    def test_optimize_model_compile_predictor_flag(self, tiny_model):
        original_predictor = tiny_model.predictor
        optimize_model(
            tiny_model,
            compile_predictor=True,
            compile_encoder=False,
            mode="default",
        )
        # torch.compile wraps the module; it should be a different object
        # (OptimizedModule wrapper).
        assert tiny_model.predictor is not original_predictor

    def test_optimize_model_compile_encoder_flag(self, tiny_model):
        original_encoder = tiny_model.encoder
        optimize_model(
            tiny_model,
            compile_predictor=False,
            compile_encoder=True,
            mode="default",
        )
        assert tiny_model.encoder is not original_encoder

    def test_optimize_model_eval_mode(self, tiny_model):
        tiny_model.train()  # put in train mode first
        optimize_model(
            tiny_model,
            compile_predictor=False,
            compile_encoder=False,
        )
        assert not tiny_model.training, "optimize_model must put model in eval"


# ==================== ModelAdapter abstraction ====================


class TestModelAdapter:
    def test_jepa_adapter_satisfies_protocol(self, tiny_model):
        adapter = JepaAdapter(tiny_model)
        assert isinstance(adapter, ModelAdapter)

    def test_jepa_adapter_methods(self, tiny_model):
        adapter = JepaAdapter(tiny_model)
        info = {"pixels": torch.rand(1, 1, 3, 64, 64)}
        info = adapter.encode(info)
        assert "emb" in info

        act = torch.randn(1, 1, 2)
        act_emb = adapter.action_encode(act)
        assert act_emb.shape == (1, 1, 4)

        emb = torch.randn(1, 1, 16)
        pred = adapter.predict(emb, act_emb)
        assert pred.shape == (1, 1, 16)

    def test_custom_adapter_supports_rollout(self, tiny_model):
        """Custom adapter that wraps the same model differently — proves the
        rollout loop is decoupled from JEPA-specific internals."""

        class WrappedAdapter:
            """Adapter that asserts every method is hit, then delegates to JEPA."""

            def __init__(self, model):
                self.model = model
                self.encode_calls = 0
                self.action_encode_calls = 0
                self.predict_calls = 0

            def encode(self, info):
                self.encode_calls += 1
                return self.model.encode(info)

            def action_encode(self, act):
                self.action_encode_calls += 1
                return self.model.action_encoder(act)

            def predict(self, emb, act_emb):
                self.predict_calls += 1
                return self.model.predict(emb, act_emb)

        adapter = WrappedAdapter(tiny_model)
        info = {"pixels": torch.rand(1, 4, 1, 3, 64, 64)}
        actions = torch.randn(1, 4, 5, 2)
        rollout_buffered(adapter, info, actions, history_size=3)

        assert adapter.encode_calls == 1
        # 4 inner-step predicts + 1 final = 5 predict calls
        assert adapter.predict_calls == 5
        assert adapter.action_encode_calls == 5

    def test_optimize_model_accepts_custom_adapter(self, tiny_model):
        """optimize_model must accept an adapter_cls kwarg so future imagination
        engines can plug in their own adapter without forking compiled_inference."""

        class TaggedAdapter(JepaAdapter):
            tag = "custom"

        optimize_model(
            tiny_model,
            compile_predictor=False,
            compile_encoder=False,
            adapter_cls=TaggedAdapter,
        )
        # Smoke check — the patched rollout still runs.
        info = {"pixels": torch.rand(1, 2, 1, 3, 64, 64)}
        actions = torch.randn(1, 2, 4, 2)
        out = tiny_model.rollout(info, actions, history_size=3)
        assert "predicted_emb" in out
