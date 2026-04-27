"""
Phase 4: Learned Per-Step Value Function

V(z_t, z_goal) → progress score ∈ [0, 1]

Architecture: 2-layer MLP with LayerNorm + Mish activations.
Input: concatenation of z_t and z_goal (2*embed_dim).
Output: scalar progress estimate.

Ensemble of N copies trained with different seeds for uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from harness.dims import LEWM_EMBED_DIM


class ValueFunction(nn.Module):
    """Per-step value function V(z_t, z_goal) → progress."""

    def __init__(self, embed_dim: int = LEWM_EMBED_DIM, hidden_dim: int = 256):
        super().__init__()
        input_dim = embed_dim * 2  # concat z_t and z_goal
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, z_t: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: Current state embedding, shape (..., embed_dim)
            z_goal: Goal embedding, shape (..., embed_dim)
        Returns:
            Progress estimate, shape (...)
        """
        x = torch.cat([z_t, z_goal], dim=-1)
        return self.net(x).squeeze(-1)


class ValueEnsemble(nn.Module):
    """Ensemble of value functions for uncertainty estimation."""

    def __init__(
        self,
        n_members: int = 5,
        embed_dim: int = LEWM_EMBED_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_members = n_members
        self.members = nn.ModuleList([
            ValueFunction(embed_dim, hidden_dim) for _ in range(n_members)
        ])

    def forward(
        self, z_t: torch.Tensor, z_goal: torch.Tensor
    ) -> torch.Tensor:
        """Return mean prediction across ensemble members.

        Returns:
            Mean progress estimate, shape (...)
        """
        preds = torch.stack([m(z_t, z_goal) for m in self.members], dim=0)
        return preds.mean(dim=0)

    def predict_with_uncertainty(
        self, z_t: torch.Tensor, z_goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and std across ensemble.

        Returns:
            (mean, std) each shape (...)
        """
        preds = torch.stack([m(z_t, z_goal) for m in self.members], dim=0)
        return preds.mean(dim=0), preds.std(dim=0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_ensemble(
    ensemble: ValueEnsemble,
    z_t: torch.Tensor,
    z_goal: torch.Tensor,
    progress: torch.Tensor,
    n_epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """Train ensemble members independently on the same data with different shuffles.

    Args:
        ensemble: ValueEnsemble to train.
        z_t: State embeddings, shape (N, embed_dim).
        z_goal: Goal embeddings, shape (N, embed_dim).
        progress: Ground-truth progress labels, shape (N,).
        n_epochs: Number of training epochs per member.
        batch_size: Mini-batch size.
        lr: Learning rate.
        val_fraction: Fraction of data for validation.
        device: Device to train on.
        verbose: Print training progress.

    Returns:
        Dict with training history (losses per member).
    """
    ensemble = ensemble.to(device)
    N = z_t.shape[0]
    n_val = int(N * val_fraction)
    n_train = N - n_val

    history = {"train_loss": [], "val_loss": []}

    for member_idx, member in enumerate(ensemble.members):
        if verbose:
            print(f"\nTraining member {member_idx + 1}/{ensemble.n_members}")

        # Different random permutation per member
        gen = torch.Generator().manual_seed(member_idx * 1000 + 42)
        perm = torch.randperm(N, generator=gen)

        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        zt_train = z_t[train_idx].to(device)
        zg_train = z_goal[train_idx].to(device)
        p_train = progress[train_idx].to(device)

        zt_val = z_t[val_idx].to(device)
        zg_val = z_goal[val_idx].to(device)
        p_val = progress[val_idx].to(device)

        optimizer = torch.optim.AdamW(member.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_loss = float("inf")
        member_train_losses = []
        member_val_losses = []

        for epoch in range(n_epochs):
            # Train
            member.train()
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle training data
            shuffle = torch.randperm(n_train, generator=gen)

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                idx = shuffle[start:end]

                pred = member(zt_train[idx], zg_train[idx])
                loss = F.mse_loss(pred, p_train[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / n_batches

            # Validate
            member.eval()
            with torch.no_grad():
                val_pred = member(zt_val, zg_val)
                val_loss = F.mse_loss(val_pred, p_val).item()

            member_train_losses.append(avg_train_loss)
            member_val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}: train={avg_train_loss:.4f} val={val_loss:.4f}")

        history["train_loss"].append(member_train_losses)
        history["val_loss"].append(member_val_losses)

        if verbose:
            print(f"  Best val loss: {best_val_loss:.4f}")

    return history
