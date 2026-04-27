"""Tests for harness/contracts.py — typed RolloutRequest / RolloutResult /
Cost protocol that replace the implicit info_dict shape."""

from __future__ import annotations

import pytest
import torch

from harness.contracts import (
    Cost,
    PlanOutcome,
    PlanRequest,
    RolloutRequest,
    RolloutResult,
    TerminalMSECost,
)


class TestRolloutRequest:
    def test_valid_construction(self):
        state = torch.randn(2, 1, 16)
        actions = torch.randn(2, 4, 5, 3)
        req = RolloutRequest(state=state, actions=actions, history_size=3)
        assert req.B == 2
        assert req.S == 4
        assert req.T == 5
        assert req.D == 16
        assert req.A == 3

    def test_state_dim_validation(self):
        with pytest.raises(ValueError, match="state must be"):
            RolloutRequest(state=torch.randn(2, 16), actions=torch.randn(2, 4, 5, 3))

    def test_actions_dim_validation(self):
        with pytest.raises(ValueError, match="actions must be"):
            RolloutRequest(state=torch.randn(2, 1, 16), actions=torch.randn(2, 5, 3))

    def test_batch_mismatch_validation(self):
        with pytest.raises(ValueError, match="batch mismatch"):
            RolloutRequest(state=torch.randn(2, 1, 16), actions=torch.randn(3, 4, 5, 3))


class TestRolloutResult:
    def test_valid_construction(self):
        traj = torch.randn(2, 4, 6, 16)
        terminal = torch.randn(2, 4, 16)
        res = RolloutResult(trajectory=traj, terminal=terminal)
        assert res.B == 2
        assert res.S == 4
        assert res.T == 6
        assert res.D == 16

    def test_from_trajectory_extracts_terminal(self):
        traj = torch.randn(2, 4, 6, 16)
        res = RolloutResult.from_trajectory(traj)
        assert torch.equal(res.terminal, traj[:, :, -1, :])

    def test_trajectory_dim_validation(self):
        with pytest.raises(ValueError, match="trajectory must be"):
            RolloutResult(trajectory=torch.randn(2, 4, 16), terminal=torch.randn(2, 4, 16))

    def test_terminal_dim_validation(self):
        with pytest.raises(ValueError, match="terminal must be"):
            RolloutResult(trajectory=torch.randn(2, 4, 6, 16), terminal=torch.randn(2, 4, 6, 16))


class TestTerminalMSECost:
    def test_output_shape_B1(self):
        cost = TerminalMSECost()
        traj = torch.randn(1, 4, 6, 16)
        goal = torch.randn(1, 1, 16)
        out = cost(traj, goal)
        assert out.shape == (1, 4)

    def test_output_shape_B_gt_1(self):
        cost = TerminalMSECost()
        traj = torch.randn(3, 8, 5, 16)
        goal = torch.randn(3, 1, 16)
        out = cost(traj, goal)
        assert out.shape == (3, 8)

    def test_zero_distance_zero_cost(self):
        cost = TerminalMSECost()
        terminal = torch.randn(1, 4, 16)
        # Build a trajectory where the terminal equals the goal; cost should be zero.
        traj = torch.cat([torch.randn(1, 4, 5, 16), terminal.unsqueeze(2)], dim=2)
        goal = terminal[:, 0:1, :]  # (1, 1, 16) — but trajectories have 4 samples, not 1
        # Use sample 0's terminal as the goal target:
        goal = terminal[0:1, 0:1, :]  # (1, 1, 16)
        out = cost(traj, goal)
        # Sample 0 should have ~zero cost
        assert out[0, 0].item() < 1e-5

    def test_protocol_compliance(self):
        cost = TerminalMSECost()
        assert isinstance(cost, Cost)


class TestPlanRequestOutcome:
    def test_plan_request_construction(self):
        req = PlanRequest(
            state=torch.randn(1, 1, 16),
            goal=torch.randn(1, 1, 16),
            horizon=5,
        )
        assert req.horizon == 5

    def test_plan_outcome_construction(self):
        out = PlanOutcome(
            action=torch.randn(1, 3),
            terminal=torch.randn(1, 16),
            cost=torch.tensor([0.42]),
        )
        assert out.action.shape == (1, 3)
        assert float(out.cost.item()) == pytest.approx(0.42)
