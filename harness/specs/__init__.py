"""Model specs — declarative descriptions of every model the engine can run.

Each spec captures the surface that distinguishes one architecture from
another (runner, cache topology, conditioning encoders, output type).
The future imagination-engine fork consumes these specs as the *only*
model-specific entry point.

Currently includes:
    LeWMSpec — the existing LeWM JEPA + CEM stack, declared.

Adding a new model is a one-file change: write a new spec module that
returns a populated ``ModelSpec`` instance, register it, and the engine
runs it.
"""

from harness.specs.base import ModelSpec, OutputKind, RunnerKind
from harness.specs.lewm_spec import LEWM_SPEC, build_lewm_spec

__all__ = [
    "ModelSpec",
    "OutputKind",
    "RunnerKind",
    "LEWM_SPEC",
    "build_lewm_spec",
]
