"""
D4: Multi-Signal Dream Scorer

Replaces raw MSE-to-goal with a learned scoring function that combines:
1. Progress reward: how much closer did the rollout get vs starting point
2. Value estimate: learned V(z_t, z_goal) → progress ∈ [0, 1]
3. Uncertainty penalty: penalize rollouts through uncertain latent regions

The scorer is designed as a drop-in replacement for the MSE cost in
pipeline._evaluate_candidates(). It receives the full rollout embeddings
and returns a cost tensor of shape (B, S).

Usage:
    from harness.dream_scorer import DreamScorer

    scorer = DreamScorer.from_checkpoint("path/to/ensemble.pt")
    # or
    scorer = DreamScorer(ensemble=trained_ensemble)

    # Integrate with pipeline:
    pipeline.scorer = scorer  # pipeline uses scorer in _evaluate_candidates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.dims import LEWM_EMBED_DIM
from harness.value_function import ValueEnsemble


class DreamScorer:
    """Multi-signal scoring for dream trajectories.

    Combines three signals:
    - MSE progress: reduction in MSE from start to end of rollout
    - Value estimate: learned V(z_terminal, z_goal) from ensemble
    - Uncertainty penalty: ensemble std on the trajectory

    Cost = -progress + alpha * uncertainty - beta * value
    (lower is better, like MSE cost)
    """

    def __init__(
        self,
        ensemble: ValueEnsemble | None = None,
        # Signal weights
        w_mse: float = 1.0,         # weight on terminal MSE (baseline signal)
        w_progress: float = 0.5,    # weight on MSE progress (start→end improvement)
        w_value: float = 1.0,       # weight on learned value function
        w_uncertainty: float = 0.5, # weight on uncertainty penalty
        device: str = "cuda",
    ):
        self.ensemble = ensemble
        self.w_mse = w_mse
        self.w_progress = w_progress
        self.w_value = w_value
        self.w_uncertainty = w_uncertainty
        self.device = device

        if ensemble is not None:
            self.ensemble = ensemble.to(device).eval()

    def score(
        self,
        pred_emb: torch.Tensor,
        obs_emb: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Score rollout trajectories.

        Args:
            pred_emb: (B, S, T, D) predicted embeddings from rollout
            obs_emb: (B, 1, D) starting observation embedding (pre-rollout)
            goal_emb: (B, 1, D) goal embedding

        Returns:
            cost: (B, S) — lower is better
        """
        B, S, T, D = pred_emb.shape

        # Goal expanded for broadcasting
        goal_exp = goal_emb[:, -1:, :].unsqueeze(1).expand(B, S, 1, D)  # (B, S, 1, D)
        goal_flat = goal_exp[:, :, 0, :]  # (B, S, D)

        # --- Signal 1: Terminal MSE (same as original) ---
        terminal_emb = pred_emb[:, :, -1, :]  # (B, S, D)
        mse_terminal = ((terminal_emb - goal_flat) ** 2).sum(dim=-1)  # (B, S)

        cost = self.w_mse * mse_terminal

        # --- Signal 2: Progress reward ---
        if self.w_progress > 0:
            # How much closer did we get compared to starting point?
            obs_exp = obs_emb.unsqueeze(1).expand(B, S, 1, D)[:, :, 0, :]  # (B, S, D)
            mse_start = ((obs_exp - goal_flat) ** 2).sum(dim=-1)  # (B, S)
            # Progress = reduction in MSE (positive = got closer)
            progress = mse_start - mse_terminal  # (B, S)
            # Penalize negative progress (moved away from goal)
            cost = cost - self.w_progress * progress

        # --- Signal 3 & 4: Value estimate + uncertainty (requires ensemble) ---
        if self.ensemble is not None and (self.w_value > 0 or self.w_uncertainty > 0):
            with torch.no_grad():
                # Score terminal embedding with value ensemble
                t_flat = terminal_emb.reshape(-1, D)  # (B*S, D)
                g_flat = goal_flat.reshape(-1, D)      # (B*S, D)

                mean_val, std_val = self.ensemble.predict_with_uncertainty(t_flat, g_flat)
                mean_val = mean_val.reshape(B, S)  # (B, S) ∈ [0, 1]
                std_val = std_val.reshape(B, S)    # (B, S)

                # Value: higher value = closer to goal (negate for cost)
                if self.w_value > 0:
                    cost = cost - self.w_value * mean_val

                # Uncertainty: penalize uncertain predictions
                if self.w_uncertainty > 0:
                    cost = cost + self.w_uncertainty * std_val

        return cost

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda", **kwargs):
        """Load scorer from a saved ensemble checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        ensemble = ValueEnsemble(
            n_members=ckpt.get("n_members", 5),
            embed_dim=ckpt.get("embed_dim", LEWM_EMBED_DIM),
            hidden_dim=ckpt.get("hidden_dim", 256),
        )
        ensemble.load_state_dict(ckpt["state_dict"])
        return cls(ensemble=ensemble, device=device, **kwargs)

    def save(self, path: str):
        """Save ensemble checkpoint."""
        if self.ensemble is None:
            raise ValueError("No ensemble to save")
        torch.save({
            "state_dict": self.ensemble.state_dict(),
            "n_members": self.ensemble.n_members,
            "embed_dim": self.ensemble.members[0].net[0].in_features // 2,
            "hidden_dim": self.ensemble.members[0].net[0].out_features,
        }, path)


def warm_average(ensemble: ValueEnsemble) -> ValueEnsemble:
    """Weight-Average Reward Models (WARM).

    Averages the parameters of all ensemble members into a single model,
    then replaces all members with copies of the averaged model.
    Retains only generalizable features, suppresses overfitting.

    Args:
        ensemble: trained ValueEnsemble with N members

    Returns:
        New ValueEnsemble with all members set to the weight-averaged model.
    """
    n = ensemble.n_members
    avg_state = {}

    # Average all member parameters
    for key in ensemble.members[0].state_dict():
        params = torch.stack([m.state_dict()[key].float() for m in ensemble.members])
        avg_state[key] = params.mean(dim=0)

    # Create new ensemble with averaged weights
    sample_member = ensemble.members[0]
    embed_dim = sample_member.net[0].in_features // 2
    hidden_dim = sample_member.net[0].out_features

    warm_ensemble = ValueEnsemble(
        n_members=n,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )

    for member in warm_ensemble.members:
        member.load_state_dict(avg_state)

    return warm_ensemble
