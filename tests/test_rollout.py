"""Tests for JEPA.rollout — locks in the legacy info_dict-based rollout
contract before M3 (rollout consolidation).

These tests exercise the full surface used by pipeline._evaluate_candidates
and harness/value_cost.py, so the consolidated rollout in M3 must preserve
the same shapes and numerical outputs.
"""

from __future__ import annotations

import pytest
import torch

from tests._engine_fixtures import make_tiny_jepa


# ==================== JEPA.rollout shape contracts ====================


class TestJEPARollout:
    @pytest.mark.parametrize("B,S,T,H", [(1, 4, 4, 1), (1, 8, 5, 1), (2, 3, 4, 1)])
    def test_rollout_output_shapes(self, B, S, T, H):
        model = make_tiny_jepa(embed_dim=16, action_dim=2)
        pixels = torch.rand(B, S, H, 3, 64, 64)
        actions = torch.randn(B, S, T, 2)

        info = {"pixels": pixels, "action": actions[:, :, :H]}
        info = model.rollout(info, actions, history_size=3)

        assert "predicted_emb" in info
        # Trajectory length = H + (T - H) future + 1 final = T + 1
        assert info["predicted_emb"].shape == (B, S, T + 1, 16)

    def test_rollout_writes_emb_and_predicted_emb(self):
        model = make_tiny_jepa()
        pixels = torch.rand(1, 4, 1, 3, 64, 64)
        actions = torch.randn(1, 4, 4, 2)

        info = {"pixels": pixels, "action": actions[:, :, :1]}
        info = model.rollout(info, actions)

        assert "emb" in info, "rollout must populate emb"
        assert "predicted_emb" in info, "rollout must populate predicted_emb"
        # emb is the broadcast init; predicted_emb is the full trajectory
        assert info["predicted_emb"].dim() == 4

    def test_rollout_deterministic_with_fixed_seed(self):
        torch.manual_seed(0)
        model = make_tiny_jepa()
        pixels = torch.rand(1, 4, 1, 3, 64, 64)
        actions = torch.randn(1, 4, 4, 2)

        info1 = {"pixels": pixels.clone(), "action": actions[:, :, :1].clone()}
        info2 = {"pixels": pixels.clone(), "action": actions[:, :, :1].clone()}

        out1 = model.rollout(info1, actions.clone())["predicted_emb"]
        out2 = model.rollout(info2, actions.clone())["predicted_emb"]

        assert torch.allclose(out1, out2), "rollout must be deterministic"

    def test_rollout_different_actions_different_outputs(self):
        model = make_tiny_jepa()
        pixels = torch.rand(1, 2, 1, 3, 64, 64)
        a1 = torch.zeros(1, 2, 4, 2)
        a2 = torch.ones(1, 2, 4, 2)

        out1 = model.rollout({"pixels": pixels.clone(), "action": a1[:, :, :1]}, a1.clone())[
            "predicted_emb"
        ]
        out2 = model.rollout({"pixels": pixels.clone(), "action": a2[:, :, :1]}, a2.clone())[
            "predicted_emb"
        ]

        assert not torch.allclose(out1, out2), "different actions must yield different rollouts"

    @pytest.mark.parametrize("history_size", [1, 2, 3, 5, 10])
    def test_history_size_accepted(self, history_size):
        """rollout must accept any history_size >= 1 without crashing.

        (For per-step predictors like TinyPredictor, the value has no numerical
        effect; the test is a smoke check that slicing logic is robust to the
        full range of values used by pipeline.history_size.)"""
        model = make_tiny_jepa()
        pixels = torch.rand(1, 2, 1, 3, 64, 64)
        actions = torch.randn(1, 2, 4, 2)

        info = {"pixels": pixels, "action": actions[:, :, :1]}
        info = model.rollout(info, actions, history_size=history_size)
        assert info["predicted_emb"].shape == (1, 2, 5, 32)


# ==================== JEPA.encode contract ====================


class TestJEPAEncode:
    def test_encode_writes_emb(self):
        model = make_tiny_jepa(embed_dim=16)
        info = {"pixels": torch.rand(1, 1, 3, 64, 64)}
        info = model.encode(info)
        assert "emb" in info
        assert info["emb"].shape == (1, 1, 16)

    def test_encode_writes_act_emb_when_action_present(self):
        model = make_tiny_jepa()
        info = {
            "pixels": torch.rand(1, 1, 3, 64, 64),
            "action": torch.randn(1, 1, 2),
        }
        info = model.encode(info)
        assert "act_emb" in info

    def test_encode_no_action_no_act_emb(self):
        model = make_tiny_jepa()
        info = {"pixels": torch.rand(1, 1, 3, 64, 64)}
        info = model.encode(info)
        assert "act_emb" not in info


# ==================== JEPA.predict contract ====================


class TestJEPAPredict:
    def test_predict_shape(self):
        model = make_tiny_jepa(embed_dim=16, action_emb_dim=8)
        emb = torch.randn(2, 3, 16)
        act_emb = torch.randn(2, 3, 8)
        out = model.predict(emb, act_emb)
        assert out.shape == (2, 3, 16)

    def test_predict_batch_independence(self):
        """Predicting two batches independently must equal predicting them stacked."""
        model = make_tiny_jepa()
        emb1 = torch.randn(1, 3, 32)
        emb2 = torch.randn(1, 3, 32)
        a1 = torch.randn(1, 3, 8)
        a2 = torch.randn(1, 3, 8)

        out_separate = torch.cat([model.predict(emb1, a1), model.predict(emb2, a2)], dim=0)
        out_stacked = model.predict(torch.cat([emb1, emb2], dim=0), torch.cat([a1, a2], dim=0))

        assert torch.allclose(out_separate, out_stacked, atol=1e-5)
