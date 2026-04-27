"""
CEMSolver — Cross-Entropy Method action search over a learned dynamics model.

Extracted from harness/pipeline.py (was: _cem_plan, _cem_plan_batched,
_score_state, _evaluate_candidates) so the CEM logic can be tested,
swapped, and reused independently of the planning pipeline.

Design:
    - Pure torch — no stable_pretraining, no stable_worldmodel, no HF.
    - Operates on embeddings only (encoding is the pipeline's job).
    - Cost is pluggable via the Cost protocol from harness.contracts.
    - Pipeline-free: a CEMSolver is constructed with (model, hparams) and
      called with (obs_emb, goal_emb). No pipeline reference required.

The math is a faithful copy of pipeline.py's pre-refactor code; this module
introduces no algorithmic changes. Behavior is regression-locked by
tests/test_cem.py and tests/test_rollout.py.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from harness.contracts import Cost, TerminalMSECost


class _DynamicsModel(Protocol):
    """Minimum surface CEMSolver needs from the model.

    Both the legacy JEPA module and any future imagination-engine model
    can plug in by exposing these methods.
    """

    def predict(self, emb: Tensor, act_emb: Tensor) -> Tensor: ...

    @property
    def action_encoder(self): ...


class CEMSolver:
    """Cross-Entropy Method planner over a learned dynamics model.

    Parameters
    ----------
    model:
        Object exposing `predict(emb, act_emb)` and an `action_encoder`
        callable. JEPA satisfies this; future imagination-engine models
        should too.
    action_dim:
        Dimensionality of a single action vector.
    horizon:
        Number of future steps to roll out.
    history_size:
        Predictor sliding-window size (only the last `history_size`
        (state, action) pairs are fed to `predict`).
    num_samples:
        CEM sample budget per iteration.
    n_steps:
        Number of CEM iterations (sample → score → fit).
    topk:
        Elite-set size used to refit the proposal distribution.
    cost:
        Pluggable trajectory cost; defaults to TerminalMSECost.
    scorer:
        Optional multi-signal scorer with a `score(pred_emb, obs_emb,
        goal_emb)` method (D4 path). When set, it overrides `cost`.
        Kept as a backward-compat hook from the legacy pipeline.
    """

    def __init__(
        self,
        model: _DynamicsModel,
        *,
        action_dim: int,
        horizon: int = 5,
        history_size: int = 3,
        num_samples: int = 128,
        n_steps: int = 15,
        topk: int = 25,
        cost: Optional[Cost] = None,
        scorer=None,
    ) -> None:
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.history_size = history_size
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.cost: Cost = cost if cost is not None else TerminalMSECost()
        self.scorer = scorer

    # --------------------------------------------------------------- API

    @torch.inference_mode()
    def plan(
        self,
        obs_emb: Tensor,
        goal_emb: Tensor,
        *,
        return_terminal_emb: bool = False,
        return_cost: bool = False,
        obs_emb_for_scorer: Optional[Tensor] = None,
    ):
        """Single-stream CEM (B=1).

        Returns
        -------
        action: np.ndarray of shape (action_dim,)
            First action of the best-mean plan.
        terminal_emb: Tensor of shape (1, 1, D), if return_terminal_emb.
        best_cost: float, if return_cost.
        """
        S = self.num_samples
        H = 1
        T = H + self.horizon
        device = obs_emb.device

        mean = torch.zeros(1, T, self.action_dim, device=device)
        var = torch.ones(1, T, self.action_dim, device=device)

        best_cost = None
        for _ in range(self.n_steps):
            noise = torch.randn(1, S, T, self.action_dim, device=device)
            candidates = noise * var.unsqueeze(1) + mean.unsqueeze(1)
            candidates[:, 0] = mean

            costs, _ = self.evaluate_candidates(
                obs_emb,
                goal_emb,
                candidates,
                S=S,
                H=H,
                return_embs=False,
                obs_emb_for_scorer=obs_emb_for_scorer,
            )
            topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
            topk_cands = candidates[0, topk_inds[0]]
            best_cost = float(topk_vals[0, 0])

            mean = topk_cands.mean(dim=0, keepdim=True)
            var = topk_cands.std(dim=0, keepdim=True)

        action = mean[0, 0].detach().cpu().numpy()

        if return_terminal_emb or return_cost:
            terminal_emb = None
            if return_terminal_emb:
                mean_candidate = mean.unsqueeze(1)
                _, mean_embs = self.evaluate_candidates(
                    obs_emb,
                    goal_emb,
                    mean_candidate,
                    S=1,
                    H=H,
                    return_embs=True,
                    obs_emb_for_scorer=obs_emb_for_scorer,
                )
                terminal_emb = mean_embs[:, :, -1:, :].squeeze(1)
            return action, terminal_emb, best_cost

        return action

    @torch.inference_mode()
    def plan_batched(
        self,
        obs_emb: Tensor,
        goal_emb: Tensor,
        *,
        return_terminal_emb: bool = True,
    ):
        """B independent CEM instances in parallel.

        obs_emb: (B, 1, D), goal_emb: (B, 1, D).

        Returns
        -------
        actions: np.ndarray of shape (B, action_dim).
        terminal_embs: Tensor of shape (B, 1, D), if return_terminal_emb.
        """
        B = obs_emb.shape[0]
        S = self.num_samples
        H = 1
        T = H + self.horizon
        device = obs_emb.device

        mean = torch.zeros(B, T, self.action_dim, device=device)
        var = torch.ones(B, T, self.action_dim, device=device)

        for _ in range(self.n_steps):
            noise = torch.randn(B, S, T, self.action_dim, device=device)
            candidates = noise * var.unsqueeze(1) + mean.unsqueeze(1)
            candidates[:, 0] = mean

            costs, _ = self.evaluate_candidates(
                obs_emb, goal_emb, candidates, S=S, H=H, return_embs=False
            )

            _, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
            topk_inds_expanded = topk_inds.unsqueeze(-1).unsqueeze(-1).expand(
                B, self.topk, T, self.action_dim
            )
            topk_cands = torch.gather(candidates, 1, topk_inds_expanded)

            mean = topk_cands.mean(dim=1)
            var = topk_cands.std(dim=1)

        actions = mean[:, 0].detach().cpu().numpy()

        if return_terminal_emb:
            mean_candidates = mean.unsqueeze(1)
            _, mean_embs = self.evaluate_candidates(
                obs_emb, goal_emb, mean_candidates, S=1, H=H, return_embs=True
            )
            terminal_embs = mean_embs[:, 0, -1:, :]
            return actions, terminal_embs

        return actions, None

    @torch.inference_mode()
    def score_state(
        self,
        obs_emb: Tensor,
        goal_emb: Tensor,
        n_rounds: int = 1,
    ) -> float:
        """Score how plannable a state is via lightweight CEM (B=1).

        n_rounds=1 is a single random-sample pass. n_rounds=3-5 is mini-CEM
        that gives better signal. Uses the same num_samples as the main CEM
        for CUDA-graph compatibility (legacy contract).
        """
        S = self.num_samples
        H = 1
        T = H + self.horizon
        device = obs_emb.device

        mean = torch.zeros(1, T, self.action_dim, device=device)
        var = torch.ones(1, T, self.action_dim, device=device)

        best_cost = float("inf")
        for _ in range(n_rounds):
            noise = torch.randn(1, S, T, self.action_dim, device=device)
            candidates = mean.unsqueeze(1) + noise * var.unsqueeze(1).sqrt()
            candidates[:, 0] = mean

            costs, _ = self.evaluate_candidates(
                obs_emb, goal_emb, candidates, S=S, H=H, return_embs=False
            )

            topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
            topk_cands = candidates[0, topk_inds[0]]
            mean = topk_cands.mean(dim=0, keepdim=True)
            var = topk_cands.std(dim=0, keepdim=True)

            round_best = float(topk_vals[0, 0])
            if round_best < best_cost:
                best_cost = round_best

        return best_cost

    @torch.inference_mode()
    def evaluate_candidates(
        self,
        obs_emb: Tensor,
        goal_emb: Tensor,
        candidates: Tensor,
        *,
        S: int,
        H: int,
        return_embs: bool = False,
        obs_emb_for_scorer: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Roll out candidate action sequences and score with the cost.

        Args
        ----
        obs_emb: (B, 1, D) starting embedding(s).
        goal_emb: (B, 1, D) goal embedding(s).
        candidates: (B, S, T, action_dim).
        S: samples per batch element.
        H: history length (typically 1).
        return_embs: if True, also return predicted trajectory.
        obs_emb_for_scorer: optional override fed to the multi-signal scorer
            (used by the planning pipeline for "current obs vs imagined obs").

        Returns
        -------
        costs: (B, S) tensor.
        all_embs: (B, S, T_full, D), if return_embs.
        """
        B = obs_emb.shape[0]
        horizon = self.horizon

        emb = obs_emb.unsqueeze(1).expand(B, S, -1, -1)
        emb = rearrange(emb, "b s t d -> (b s) t d").clone()

        act_0 = candidates[:, :, :H, :]
        act_future = candidates[:, :, H:, :]
        act = rearrange(act_0, "b s t d -> (b s) t d")
        act_future_flat = rearrange(act_future, "b s t d -> (b s) t d")

        HS = self.history_size

        for t in range(horizon):
            start = max(0, emb.shape[1] - HS)
            act_emb = self.model.action_encoder(act[:, start:, :])
            pred = self.model.predict(emb[:, start:, :], act_emb)[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
            act = torch.cat([act, act_future_flat[:, t : t + 1, :]], dim=1)

        # Final predict (matches legacy contract: T_full = H + horizon + 1)
        start = max(0, emb.shape[1] - HS)
        act_emb = self.model.action_encoder(act[:, start:, :])
        pred = self.model.predict(emb[:, start:, :], act_emb)[:, -1:]
        emb = torch.cat([emb, pred], dim=1)

        pred_emb = rearrange(emb, "(b_s) t d -> b_s t d", b_s=B * S)
        pred_emb = pred_emb.view(B, S, pred_emb.shape[1], pred_emb.shape[2])

        if self.scorer is not None and obs_emb_for_scorer is not None:
            cost = self.scorer.score(pred_emb, obs_emb_for_scorer, goal_emb)
        else:
            cost = self.cost(pred_emb, goal_emb)

        if return_embs:
            return cost, pred_emb
        return cost, None
