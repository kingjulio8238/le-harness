"""
Typed contracts for the LeWM engine surface.

Replaces the implicit info_dict shape that was threaded through pipeline.py,
jepa.py, and compiled_inference.py. Used by CEMSolver, rollout primitives,
and Cost implementations.

Shape conventions
-----------------
B: batch (independent planning instances)
S: samples per CEM batch (action candidates)
T: time / sequence length
D: embedding dim
A: action dim
H: history length

Wherever a function says it returns or accepts (B, S, T, D), it means exactly
that — no implicit broadcasting, no None placeholders. Callers must reshape
explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass
class RolloutRequest:
    """Inputs to a model rollout.

    Attributes:
        state: Initial state embeddings, shape (B, T_hist, D).
            T_hist is the history depth; usually 1 for "current observation only"
            but can be longer for models that condition on a window of past states.
        actions: Candidate action sequences, shape (B, S, T_horizon, A).
            T_horizon includes the historical actions (first T_hist) and the
            future actions (remaining T_horizon - T_hist) the engine should
            apply during rollout.
        history_size: Sliding-window size the model uses internally — the
            predictor only attends to the last `history_size` (state, action)
            pairs at each step.
    """

    state: Tensor
    actions: Tensor
    history_size: int = 3

    def __post_init__(self) -> None:
        if self.state.ndim != 3:
            raise ValueError(
                f"state must be (B, T_hist, D); got shape {tuple(self.state.shape)}"
            )
        if self.actions.ndim != 4:
            raise ValueError(
                f"actions must be (B, S, T, A); got shape {tuple(self.actions.shape)}"
            )
        B_state, _, _ = self.state.shape
        B_act, _, _, _ = self.actions.shape
        if B_state != B_act:
            raise ValueError(
                f"batch mismatch: state B={B_state}, actions B={B_act}"
            )

    @property
    def B(self) -> int:
        return self.state.shape[0]

    @property
    def S(self) -> int:
        return self.actions.shape[1]

    @property
    def T(self) -> int:
        return self.actions.shape[2]

    @property
    def D(self) -> int:
        return self.state.shape[-1]

    @property
    def A(self) -> int:
        return self.actions.shape[-1]


@dataclass
class RolloutResult:
    """Output of a model rollout.

    Attributes:
        trajectory: Predicted state embeddings across the rollout, shape
            (B, S, T_full, D). T_full = T_hist + n_future_steps + 1
            (initial states + per-step predictions + terminal prediction).
        terminal: Convenience handle for the final predicted state per
            (batch, sample), shape (B, S, D). Equals trajectory[:, :, -1, :].
    """

    trajectory: Tensor
    terminal: Tensor

    def __post_init__(self) -> None:
        if self.trajectory.ndim != 4:
            raise ValueError(
                f"trajectory must be (B, S, T, D); got shape {tuple(self.trajectory.shape)}"
            )
        if self.terminal.ndim != 3:
            raise ValueError(
                f"terminal must be (B, S, D); got shape {tuple(self.terminal.shape)}"
            )

    @property
    def B(self) -> int:
        return self.trajectory.shape[0]

    @property
    def S(self) -> int:
        return self.trajectory.shape[1]

    @property
    def T(self) -> int:
        return self.trajectory.shape[2]

    @property
    def D(self) -> int:
        return self.trajectory.shape[-1]

    @classmethod
    def from_trajectory(cls, trajectory: Tensor) -> "RolloutResult":
        """Convenience constructor — extracts terminal as trajectory[:, :, -1, :]."""
        return cls(trajectory=trajectory, terminal=trajectory[:, :, -1, :].contiguous())


@runtime_checkable
class Cost(Protocol):
    """Pluggable cost over rolled-out trajectories.

    A Cost takes the predicted trajectory and a goal embedding, and returns
    one scalar cost per (batch, sample). Lower is better — solvers minimize.

    Implementations:
        TerminalMSECost: classic MSE between final predicted state and goal.
        ValueCost: learned value function (V(z_t, z_goal) → progress).
        DreamScorerCost: multi-signal scoring (D4).

    Signature:
        cost(trajectory: (B, S, T, D), goal: (B, T_goal, D)) -> (B, S)
    """

    def __call__(self, trajectory: Tensor, goal: Tensor) -> Tensor: ...


class TerminalMSECost:
    """Default cost: MSE between final predicted state and goal embedding.

    Mirrors the historical pipeline._evaluate_candidates default.
    """

    def __call__(self, trajectory: Tensor, goal: Tensor) -> Tensor:
        # trajectory: (B, S, T, D); goal: (B, T_goal, D) — broadcast last goal step
        goal_last = goal[:, -1:, :].unsqueeze(1)        # (B, 1, 1, D)
        pred_last = trajectory[:, :, -1:, :]              # (B, S, 1, D)
        return ((pred_last - goal_last) ** 2).sum(dim=-1).squeeze(-1)  # (B, S)


@dataclass
class PlanRequest:
    """Inputs to a solver's plan() call.

    Used by CEMSolver and Dream Tree. Wraps state + goal + horizon.

    Attributes:
        state: (B, T_hist, D) initial state embedding(s).
        goal: (B, T_goal, D) goal embedding(s); T_goal usually 1.
        horizon: number of future steps the solver must plan ahead.
    """

    state: Tensor
    goal: Tensor
    horizon: int


@dataclass
class PlanOutcome:
    """Solver result at the engine layer (below PlanResult).

    PlanResult lives in harness/plan_result.py and is the *planner's* output
    contract (consumed by S1.5 / VLM). PlanOutcome is the *solver's* output —
    the planner adds confidence/timing/etc on top.

    Attributes:
        action: (B, A) first action of the best plan, per batch element.
        terminal: (B, D) predicted terminal state of the best plan.
        cost: (B,) best cost achieved.
    """

    action: Tensor
    terminal: Tensor
    cost: Tensor
