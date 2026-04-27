"""Shared fixtures for engine-side tests (CEM, rollout, compiled_inference).

CPU-only and dependency-light, mirroring tests/test_goal_adapter.py.

Notes on import policy:
    `harness.pipeline` pulls in `stable_pretraining` and `stable_worldmodel`,
    which require GPU-side torchvision builds. Local CI may not have them.
    Engine-side tests therefore avoid `from harness.pipeline import ...` —
    they exercise jepa/cem/compiled_inference directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from jepa import JEPA


class TinyEncoderOutput:
    """Mimics transformers' BaseModelOutput shape."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class TinyEncoder(nn.Module):
    """Mini ViT-shaped encoder. Returns object with .last_hidden_state of
    shape (B, T_patches+1, D); index 0 is the CLS token."""

    def __init__(self, embed_dim: int = 32, patch_size: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, interpolate_pos_encoding: bool = False):
        feats = self.proj(x)
        feats = rearrange(feats, "b d h w -> b (h w) d")
        cls = self.cls.expand(feats.size(0), -1, -1)
        out = torch.cat([cls, feats], dim=1)
        return TinyEncoderOutput(self.norm(out))


class TinyPredictor(nn.Module):
    """Mini predictor: concat (emb, act_emb) per step, linear to next emb."""

    def __init__(self, embed_dim: int = 32, action_emb_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(embed_dim + action_emb_dim, embed_dim)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb, act_emb], dim=-1)
        return self.fc(x)


class _PatchEmbedShim(nn.Module):
    """Carries `in_channels` for pipeline._action_dim introspection."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels


class TinyActionEncoder(nn.Module):
    """Mini action encoder: (B, T, A) → (B, T, A_emb)."""

    def __init__(self, action_dim: int = 2, action_emb_dim: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(action_dim, action_emb_dim)
        self.patch_embed = _PatchEmbedShim(in_channels=action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def make_tiny_jepa(
    embed_dim: int = 32, action_dim: int = 2, action_emb_dim: int = 8
) -> JEPA:
    """Construct a CPU-runnable JEPA with random weights for engine-side tests."""
    encoder = TinyEncoder(embed_dim=embed_dim)
    predictor = TinyPredictor(embed_dim=embed_dim, action_emb_dim=action_emb_dim)
    action_encoder = TinyActionEncoder(action_dim=action_dim, action_emb_dim=action_emb_dim)
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
    )
    model.eval()
    model.requires_grad_(False)
    return model


def make_info_dict(
    B: int = 1,
    S: int = 4,
    T: int = 4,
    H: int = 1,
    action_dim: int = 2,
    img_size: int = 64,
) -> dict:
    """Build an info dict matching the legacy JEPA.rollout contract.

    See docs/CONTRACTS.md for the full key/shape table.
    """
    pixels = torch.rand(B, S, H, 3, img_size, img_size)
    actions = torch.randn(B, S, T, action_dim)
    goal = torch.rand(B, S, 1, 3, img_size, img_size)
    return {
        "pixels": pixels,
        "goal": goal,
        "action": actions[:, :, :H],
        "_full_action_sequence": actions,
    }
