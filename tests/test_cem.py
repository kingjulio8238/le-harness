"""Tests for harness/cem.py::CEMSolver.

Locks in the behavior extracted from pipeline._cem_plan, _cem_plan_batched,
_score_state, _evaluate_candidates so the post-fork engine can rely on a
stable solver contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harness.cem import CEMSolver
from harness.contracts import TerminalMSECost
from tests._engine_fixtures import make_tiny_jepa


# Shared small CEM setup — keeps tests fast on CPU.
DEFAULT_KW = dict(
    horizon=3,
    history_size=3,
    num_samples=8,
    n_steps=2,
    topk=3,
    action_dim=2,
)


@pytest.fixture
def solver():
    torch.manual_seed(0)
    model = make_tiny_jepa(embed_dim=16, action_dim=2, action_emb_dim=4)
    return CEMSolver(model, **DEFAULT_KW)


# ==================== plan() ====================


class TestPlan:
    def test_plan_action_shape(self, solver):
        torch.manual_seed(0)
        obs = torch.randn(1, 1, 16)
        goal = torch.randn(1, 1, 16)
        action = solver.plan(obs, goal)
        assert isinstance(action, np.ndarray)
        assert action.shape == (DEFAULT_KW["action_dim"],)

    def test_plan_returns_terminal_when_requested(self, solver):
        torch.manual_seed(0)
        obs = torch.randn(1, 1, 16)
        goal = torch.randn(1, 1, 16)
        action, terminal, cost = solver.plan(
            obs, goal, return_terminal_emb=True, return_cost=True
        )
        assert action.shape == (2,)
        assert terminal.shape == (1, 1, 16)
        assert isinstance(cost, float)

    def test_plan_deterministic_with_fixed_seed(self, solver):
        obs = torch.randn(1, 1, 16)
        goal = torch.randn(1, 1, 16)

        torch.manual_seed(42)
        a1 = solver.plan(obs.clone(), goal.clone())
        torch.manual_seed(42)
        a2 = solver.plan(obs.clone(), goal.clone())

        assert np.allclose(a1, a2)

    def test_plan_different_seeds_different_actions(self, solver):
        obs = torch.randn(1, 1, 16)
        goal = torch.randn(1, 1, 16)

        torch.manual_seed(0)
        a1 = solver.plan(obs.clone(), goal.clone())
        torch.manual_seed(1)
        a2 = solver.plan(obs.clone(), goal.clone())

        # Stochastic CEM with different seeds → almost certainly different action.
        assert not np.allclose(a1, a2, atol=1e-6)

    def test_plan_finite_cost(self, solver):
        torch.manual_seed(0)
        obs = torch.randn(1, 1, 16)
        goal = torch.randn(1, 1, 16)
        _, _, cost = solver.plan(obs, goal, return_terminal_emb=True, return_cost=True)
        assert np.isfinite(cost), f"cost must be finite, got {cost}"


# ==================== plan_batched() ====================


class TestPlanBatched:
    @pytest.mark.parametrize("B", [1, 2, 4])
    def test_batched_action_shape(self, B):
        torch.manual_seed(0)
        model = make_tiny_jepa(embed_dim=16, action_dim=2, action_emb_dim=4)
        solver = CEMSolver(model, **DEFAULT_KW)

        obs = torch.randn(B, 1, 16)
        goal = torch.randn(B, 1, 16)
        actions, terminals = solver.plan_batched(obs, goal, return_terminal_emb=True)

        assert actions.shape == (B, 2)
        assert terminals.shape == (B, 1, 16)

    def test_batched_no_terminal(self):
        torch.manual_seed(0)
        model = make_tiny_jepa(embed_dim=16)
        solver = CEMSolver(model, **DEFAULT_KW)

        obs = torch.randn(3, 1, 16)
        goal = torch.randn(3, 1, 16)
        actions, terminals = solver.plan_batched(obs, goal, return_terminal_emb=False)
        assert actions.shape == (3, 2)
        assert terminals is None


# ==================== score_state() ====================


class TestScoreState:
    @pytest.mark.parametrize("n_rounds", [1, 3])
    def test_score_returns_finite_float(self, solver, n_rounds):
        torch.manual_seed(0)
        obs = torch.randn(1, 1, 16)
        goal = torch.randn(1, 1, 16)
        score = solver.score_state(obs, goal, n_rounds=n_rounds)
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_more_rounds_no_worse(self, solver):
        """Mini-CEM with more rounds must produce a score ≤ single-round
        (the legacy contract: best_cost is the running min)."""
        obs = torch.randn(1, 1, 16)
        goal = torch.randn(1, 1, 16)

        torch.manual_seed(0)
        score_1 = solver.score_state(obs.clone(), goal.clone(), n_rounds=1)
        torch.manual_seed(0)
        score_3 = solver.score_state(obs.clone(), goal.clone(), n_rounds=3)

        assert score_3 <= score_1 + 1e-5


# ==================== evaluate_candidates() ====================


class TestEvaluateCandidates:
    def test_costs_shape(self, solver):
        torch.manual_seed(0)
        B, S = 2, 4
        obs = torch.randn(B, 1, 16)
        goal = torch.randn(B, 1, 16)
        T = 1 + solver.horizon
        candidates = torch.randn(B, S, T, 2)

        costs, embs = solver.evaluate_candidates(obs, goal, candidates, S=S, H=1)
        assert costs.shape == (B, S)
        assert embs is None

    def test_return_embs(self, solver):
        torch.manual_seed(0)
        B, S = 1, 4
        obs = torch.randn(B, 1, 16)
        goal = torch.randn(B, 1, 16)
        T = 1 + solver.horizon
        candidates = torch.randn(B, S, T, 2)

        costs, embs = solver.evaluate_candidates(
            obs, goal, candidates, S=S, H=1, return_embs=True
        )
        assert costs.shape == (B, S)
        # T_full = H + horizon + 1 = 1 + 3 + 1 = 5 (initial + 3 future + final)
        assert embs.shape == (B, S, 1 + solver.horizon + 1, 16)

    def test_pluggable_cost(self, solver):
        torch.manual_seed(0)

        # Custom cost that always returns ones — verifies the Cost protocol
        # is actually plumbed through evaluate_candidates.
        class OnesCost:
            def __call__(self, traj, goal):
                return torch.ones(traj.shape[0], traj.shape[1], device=traj.device)

        solver.cost = OnesCost()
        obs = torch.randn(1, 1, 16)
        goal = torch.randn(1, 1, 16)
        candidates = torch.randn(1, 4, 4, 2)

        costs, _ = solver.evaluate_candidates(obs, goal, candidates, S=4, H=1)
        assert torch.allclose(costs, torch.ones(1, 4))


# ==================== Cost protocol integration ====================


class TestDefaultCost:
    def test_default_is_terminal_mse(self):
        model = make_tiny_jepa()
        solver = CEMSolver(model, **DEFAULT_KW)
        assert isinstance(solver.cost, TerminalMSECost)
