"""
Canonical rollout primitive.

Replaces the two parallel implementations that existed pre-refactor:
    - jepa.py::JEPA.rollout (torch.cat, training-shaped)
    - compiled_inference._patch_rollout_with_buffers.optimized_rollout
      (pre-allocated buffers, inference-shaped)

The pre-allocated-buffer version is now the only implementation. JEPA.rollout
delegates here; compiled_inference exposes the same function via the
optimize_model patch path.

The ModelAdapter abstraction decouples the rollout loop from JEPA-specific
key names (``info["pixels"]``, ``info["emb"]``, ``info["action"]``) so a
future model with different state/conditioning surfaces can plug in by
implementing the adapter protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from einops import rearrange


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimum surface the canonical rollout needs from a model.

    A ModelAdapter wraps a concrete dynamics model and exposes three calls:

    - encode(info)             → info dict with "emb" populated, shape (B, T, D)
    - action_encode(act)       → action embedding, shape (B, T, A_emb)
    - predict(emb, act_emb)    → next-state prediction, shape (B, T, D)

    Default JepaAdapter wraps a JEPA module. A future imagination-engine
    model writes its own adapter — no need to fork the rollout loop.
    """

    def encode(self, info: dict) -> dict: ...

    def action_encode(self, action: torch.Tensor) -> torch.Tensor: ...

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor: ...


class JepaAdapter:
    """Adapter for a JEPA module — the legacy default."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def encode(self, info: dict) -> dict:
        return self.model.encode(info)

    def action_encode(self, action: torch.Tensor) -> torch.Tensor:
        return self.model.action_encoder(action)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.model.predict(emb, act_emb)


@torch.inference_mode()
def rollout_buffered(
    adapter: ModelAdapter,
    info: dict,
    action_sequence: torch.Tensor,
    history_size: int = 3,
) -> dict:
    """Canonical buffer-pre-allocated rollout.

    Mutates and returns the input ``info`` dict (legacy contract — see
    docs/CONTRACTS.md). Writes:
        info["action"]         ← initial actions, sliced from action_sequence
        info["emb"]            ← broadcast initial embedding (B, S, T_init, D)
        info["predicted_emb"]  ← full trajectory (B, S, T_full, D)

    Args
    ----
    adapter: ModelAdapter exposing encode / action_encode / predict.
    info: legacy info dict; must contain "pixels" of shape (B, S, T_init, ...).
    action_sequence: (B, S, T, A) — full action sequence over the rollout.
    history_size: predictor sliding window.

    Returns
    -------
    The same ``info`` dict, with ``predicted_emb`` populated.
    """
    assert "pixels" in info, "pixels not in info_dict"
    H = info["pixels"].size(2)
    B, S, T = action_sequence.shape[:3]
    act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
    info["action"] = act_0
    n_steps = T - H

    # Encode initial info — take sample 0 since pixels are usually shared.
    _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
    _init = adapter.encode(_init)
    emb_init = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
    info["emb"] = emb_init

    # Flatten batch and sample dims so predict() sees one long batch.
    emb = rearrange(emb_init, "b s ... -> (b s) ...").clone()
    act = rearrange(act_0, "b s ... -> (b s) ...")
    act_future_flat = rearrange(act_future, "b s ... -> (b s) ...")

    BS = B * S
    D = emb.shape[-1]
    total_steps = H + n_steps + 1   # initial + per-step preds + final predict

    emb_buffer = torch.empty(BS, total_steps, D, device=emb.device, dtype=emb.dtype)
    emb_buffer[:, :H, :] = emb

    act_dim = act.shape[-1]
    act_buffer = torch.empty(
        BS, total_steps - 1, act_dim, device=act.device, dtype=act.dtype
    )
    act_buffer[:, :H, :] = act

    HS = history_size
    write_pos = H

    for t in range(n_steps):
        start = max(0, write_pos - HS)
        act_emb = adapter.action_encode(act_buffer[:, start:write_pos, :])
        emb_trunc = emb_buffer[:, start:write_pos, :]

        pred_emb = adapter.predict(emb_trunc, act_emb)[:, -1:]
        emb_buffer[:, write_pos:write_pos + 1, :] = pred_emb
        act_buffer[:, write_pos:write_pos + 1, :] = act_future_flat[:, t:t + 1, :]
        write_pos += 1

    # Final prediction (matches legacy contract: trajectory length = T + 1).
    start = max(0, write_pos - HS)
    act_emb = adapter.action_encode(act_buffer[:, start:write_pos, :])
    emb_trunc = emb_buffer[:, start:write_pos, :]
    pred_emb = adapter.predict(emb_trunc, act_emb)[:, -1:]
    emb_buffer[:, write_pos:write_pos + 1, :] = pred_emb
    write_pos += 1

    pred_rollout = rearrange(
        emb_buffer[:, :write_pos, :], "(b s) t d -> b s t d", b=B, s=S
    )
    info["predicted_emb"] = pred_rollout
    return info
