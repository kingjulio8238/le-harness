"""Base classes for model specs.

A ModelSpec is the declarative entry point for a single model architecture.
The engine reads the spec to decide which runner to use, which conditioning
encoders to load, what cache topology to allocate, etc.

This is a pre-fork sketch: the spec ships, but the engine doesn't yet
consume specs as its primary loading mechanism. After the imagination-engine
fork lands, ``Engine(spec=...)`` becomes the only construction path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class RunnerKind(str, Enum):
    """Which runner shape this model uses."""

    DETERMINISTIC = "deterministic"   # single forward — JEPA, V-JEPA, DINO-WM
    DIFFUSION = "diffusion"           # ODE/SDE solver — Waypoint, Cosmos
    AR_TOKEN = "ar_token"             # autoregressive over discrete latents — Genie
    SEARCH = "search"                 # wraps another runner with CEM/MCTS/MPPI


class OutputKind(str, Enum):
    """What the engine emits per step."""

    EMBEDDING = "embedding"   # next-state latent (LeWM, JEPA family)
    IMAGE = "image"           # decoded pixels (Waypoint, Cosmos)
    DISCRETE_TOKEN = "discrete_token"  # discrete latent token (Genie-style)


@dataclass
class EncoderSpec:
    """Declarative encoder slot.

    The actual encoder module is constructed lazily by the engine using
    ``factory()``; the spec carries metadata only so it can be inspected
    without loading heavy weights.
    """

    name: str                     # e.g. "image", "text", "vlm-siglip"
    out_dim: int                  # dim of the produced embedding
    factory: Optional[Callable[[], Any]] = None


@dataclass
class CacheSpec:
    """Declarative cache topology — None means no cache (e.g. JEPA)."""

    layout: str                   # "ring", "ring+tail", "paged", "none"
    n_layers: int = 0
    head_dim: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class SchedulerSpec:
    """Declarative diffusion scheduler — None for non-diffusion runners."""

    family: str                   # "rectified_flow", "edm", "ddpm", "consistency"
    sigmas: Optional[list] = None  # explicit schedule, if known at spec time
    n_steps: Optional[int] = None  # default solver steps


@dataclass
class ModelSpec:
    """Complete declarative description of a single model.

    Pre-fork: serves as documentation + a typed surface other code can
    consume (e.g. ``LEWM_SPEC.embed_dim`` instead of magic numbers).
    Post-fork: the imagination engine constructs runners + encoders +
    caches from this spec exclusively.
    """

    name: str
    runner: RunnerKind
    output: OutputKind
    embed_dim: int
    action_dim: int
    encoders: dict[str, EncoderSpec] = field(default_factory=dict)
    cache: Optional[CacheSpec] = None
    scheduler: Optional[SchedulerSpec] = None
    quant_blacklist: list[str] = field(default_factory=list)
    inference_patches: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Hook the engine calls to produce the actual nn.Module. Pre-fork
    # this is unused; post-fork the engine wires it up.
    builder: Optional[Callable[[Any], Any]] = None
