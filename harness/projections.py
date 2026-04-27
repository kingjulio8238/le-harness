"""
Projection MLPs mapping VLM embeddings to LeWM's planning space.

These are lightweight modules that can be imported without heavy
dependencies (CLIP, open_clip, etc.).

Default ``out_dim`` comes from ``harness.dims.LEWM_EMBED_DIM`` so that
non-LeWM consumers can override it cleanly.
"""

import torch.nn as nn

from harness.dims import LEWM_EMBED_DIM


class VLMProjection(nn.Module):
    """Generic MLP mapping VLM embeddings to LeWM embedding space.

    Works for any VLM family — just set in_dim to match the source encoder.

    Known dimensions:
        CLIP ViT-B/32: 512
        SigLIP (PaliGemma/Pi0/OpenVLA): 768
        T5-Base (Octo): 768
        Eagle/SmolLM (GR00T N1): 1536
        PaliGemma full (Pi0): 4608
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = LEWM_EMBED_DIM):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CoordProjection(nn.Module):
    """MLP mapping (x, y) coordinates to LeWM embedding space."""

    def __init__(self, hidden_dim=256, out_dim=LEWM_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CLIPProjection(nn.Module):
    """MLP mapping CLIP text features to LeWM embedding space."""

    def __init__(self, in_dim=512, hidden_dim=512, out_dim=LEWM_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# Convenience aliases for common VLM families
def SigLIPProjection(**kwargs):
    return VLMProjection(in_dim=768, **kwargs)

def T5Projection(**kwargs):
    return VLMProjection(in_dim=768, **kwargs)

def EagleProjection(**kwargs):
    return VLMProjection(in_dim=1536, **kwargs)

def PaliGemmaProjection(**kwargs):
    return VLMProjection(in_dim=4608, hidden_dim=1024, **kwargs)
