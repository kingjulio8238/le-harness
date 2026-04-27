"""
Phase 5: Compiled Inference Wrappers

Applies torch.compile to the model's predictor/encoder and patches
``model.rollout`` to use the canonical buffer-pre-allocated implementation
in ``harness.rollout``.

Usage:
    from harness.compiled_inference import optimize_model

    model = swm.policy.AutoCostModel(policy_name)
    model = model.to("cuda").eval()
    model = optimize_model(model)
"""

import torch

from harness.rollout import JepaAdapter, rollout_buffered


def optimize_model(model, compile_predictor=True, compile_encoder=True,
                   backend="inductor", mode="reduce-overhead",
                   adapter_cls=JepaAdapter):
    """Apply torch.compile to model components for faster inference.

    Modifies the model in-place and returns it. Uses 'reduce-overhead' mode
    by default which enables CUDA graphs for ~3.6x predictor speedup.

    Args:
        model: model instance — by default a JEPA from AutoCostModel.
        compile_predictor: whether to compile the predictor.
        compile_encoder: whether to compile the encoder.
        backend: torch.compile backend.
        mode: torch.compile mode ('default', 'reduce-overhead', 'max-autotune').
        adapter_cls: ModelAdapter class wrapping the model. Defaults to
            JepaAdapter; pass a different adapter to support non-JEPA models.
    """
    torch.set_float32_matmul_precision("high")
    model.eval()
    model.requires_grad_(False)

    if compile_predictor:
        print(f"Compiling predictor with backend='{backend}', mode='{mode}'...")
        model.predictor = torch.compile(model.predictor, backend=backend, mode=mode)

    if compile_encoder:
        print(f"Compiling encoder with backend='{backend}', mode='{mode}'...")
        model.encoder = torch.compile(model.encoder, backend=backend, mode=mode)

    _patch_rollout_with_buffers(model, adapter_cls=adapter_cls)
    return model


def _patch_rollout_with_buffers(model, adapter_cls=JepaAdapter):
    """Replace ``model.rollout`` with the canonical buffer-based rollout.

    The legacy two-implementation split is gone — both call sites
    (``JEPA.rollout`` and the compiled-inference patch) now route through
    ``harness.rollout.rollout_buffered``.
    """
    adapter = adapter_cls(model)

    @torch.inference_mode()
    def optimized_rollout(info, action_sequence, history_size=3):
        return rollout_buffered(adapter, info, action_sequence, history_size)

    model.rollout = optimized_rollout
    print("Patched rollout with pre-allocated buffers.")
