"""
Phase 7: End-to-End Planning Pipeline

Clean API wrapping the full LeHarness planning stack:
  observation image → preprocessing → compiled encoder → CEM planning → PlanResult

Usage:
    from harness.pipeline import PlanningPipeline

    pipeline = PlanningPipeline("pusht/lejepa")
    pipeline.warmup()  # triggers torch.compile (one-time)

    # Plan from raw images — returns PlanResult
    result = pipeline.plan(obs_image_np, goal_image_np)
    result.action          # (action_dim,) numpy array
    result.confidence      # 0.0-1.0
    result.needs_replan    # True if confidence < threshold

    # Backward compatible: PlanResult supports numpy array protocol
    result.reshape(5, 2)   # works like np.ndarray
    np.array(result)        # returns the action array

    # Or use in eval loop
    pipeline.set_goal(goal_image_np)
    for obs in observations:
        result = pipeline.plan(obs)
"""

import time
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms import v2 as transforms

from harness.cem import CEMSolver
from harness.compiled_inference import optimize_model
from harness.dims import LEWM_EMBED_DIM
from harness.plan_result import PlanResult


class PlanningPipeline:
    """End-to-end planning pipeline with compiled inference.

    Encapsulates model loading, compilation, image preprocessing,
    encoder caching, and CEM planning in a single clean interface.
    """

    def __init__(
        self,
        policy_name: str = "pusht/lejepa",
        num_samples: int = 128,
        n_steps: int = 15,
        horizon: int = 5,
        history_size: int = 3,
        topk: int = 25,
        device: str = "cuda",
        compile_mode: str = "reduce-overhead",
        replan_threshold: float = 0.3,
        cost_scale: float = 10.0,
        action_dim: int | None = None,
        embed_dim: int | None = None,
    ):
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.horizon = horizon
        self.history_size = history_size
        self.topk = topk
        self.device = device
        self.compile_mode = compile_mode
        self.replan_threshold = replan_threshold
        self.cost_scale = cost_scale

        # Load model
        self.model = swm.policy.AutoCostModel(policy_name)
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.model.interpolate_pos_encoding = True

        # Apply compilation
        self.model = optimize_model(
            self.model,
            compile_predictor=True,
            compile_encoder=True,
            mode=compile_mode,
        )

        # Image preprocessing (same as eval.py)
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ])

        # State
        self._goal_emb = None
        self._obs_emb = None  # cached for scorer's progress signal
        self._compiled = False
        self.language_encoder = None  # lazy-loaded by set_goal_text()

        # Action / embed dim — explicit constructor args win over introspection.
        # Introspection is the legacy fallback that assumes JEPA-shaped models;
        # an imagination-engine fork should pass these explicitly via spec.
        if action_dim is not None:
            self._action_dim = action_dim
        else:
            try:
                self._action_dim = self.model.action_encoder.patch_embed.in_channels
            except AttributeError as e:
                raise RuntimeError(
                    "Could not infer action_dim from model.action_encoder.patch_embed; "
                    "pass action_dim= explicitly to PlanningPipeline."
                ) from e
        self._embed_dim = embed_dim if embed_dim is not None else LEWM_EMBED_DIM

        # CEM solver — owns the action search loop. Pipeline is now a thin
        # facade composing Engine (encode/decode) + CEMSolver (action search).
        self.solver = CEMSolver(
            self.model,
            action_dim=self._action_dim,
            horizon=self.horizon,
            history_size=self.history_size,
            num_samples=self.num_samples,
            n_steps=self.n_steps,
            topk=self.topk,
        )

        # Timing stats
        self.timing = {
            "preprocess_ms": [],
            "encode_ms": [],
            "cem_ms": [],
            "planability_ms": [],
            "total_ms": [],
        }

    # `scorer` is owned by the solver; expose as a property so external code
    # like `pipeline.scorer = MyScorer()` (in scripts/) keeps working.
    @property
    def scorer(self):
        return self.solver.scorer

    @scorer.setter
    def scorer(self, value):
        self.solver.scorer = value

    # Public accessors so consumers like DreamTreePlanner do not need to
    # reach into underscore-prefixed pipeline internals.
    @property
    def action_dim(self) -> int:
        """Action dim used by the solver. Alias for the legacy _action_dim."""
        return self._action_dim

    @property
    def embed_dim(self) -> int:
        """Embedding dim of the model's planning space."""
        return self._embed_dim

    @property
    def obs_emb(self):
        """Most recently encoded observation embedding, cached by plan()."""
        return self._obs_emb

    @property
    def goal_emb(self):
        """Currently set goal embedding (image / text / VLM-projected)."""
        return self._goal_emb

    def warmup(self, n_iters: int = 3):
        """Trigger torch.compile by running dummy inputs."""
        print("Warming up compiled pipeline...")
        dummy_obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_goal = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.set_goal(dummy_goal)
        for i in range(n_iters):
            self.plan(dummy_obs, record_timing=False)
            print(f"  warmup {i+1}/{n_iters}")
        torch.cuda.synchronize()
        self._compiled = True
        print("Pipeline ready.")

    def preprocess(self, image_np: np.ndarray) -> torch.Tensor:
        """Preprocess a raw image for the encoder.

        Args:
            image_np: (H, W, 3) uint8 numpy array

        Returns:
            (1, 1, 3, 224, 224) float32 tensor on device
        """
        tensor = self.transform(image_np)
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed image to embedding.

        Args:
            image_tensor: (1, 1, 3, 224, 224)

        Returns:
            (1, 1, 192) embedding
        """
        with torch.inference_mode():
            result = self.model.encode({"pixels": image_tensor})
            return result["emb"]

    def set_goal(self, goal_image_np: np.ndarray):
        """Set and cache the goal embedding from an image."""
        goal_tensor = self.preprocess(goal_image_np)
        self._goal_emb = self.encode(goal_tensor)

    def set_goal_embedding(self, emb: torch.Tensor):
        """Set goal directly from a pre-computed embedding.

        This is the primary integration point for S2 (VLM) systems — the
        VLM produces an embedding, projects it to 192-dim via GoalAdapter,
        and injects it here.

        Args:
            emb: (1, 1, D) or (1, D) or (D,) tensor in LeWM's 192-dim space
        """
        if emb.dim() == 1:
            emb = emb.unsqueeze(0).unsqueeze(0)
        elif emb.dim() == 2:
            emb = emb.unsqueeze(1)
        self._goal_emb = emb.to(self.device).float()

    def load_language_encoder(self, projection_path: str, mode: str = "coord"):
        """Load the language encoder with a trained projection.

        Args:
            projection_path: path to projection weights
            mode: "coord" (parse coordinates from text), "clip" (CLIP encoder),
                  or "both" (try coord parsing first, fall back to CLIP)
        """
        from harness.language_encoder import LanguageEncoder
        self.language_encoder = LanguageEncoder(
            mode=mode, projection_path=projection_path, device=self.device
        )

    def set_goal_text(self, goal_text: str):
        """Set goal from natural language description.

        Requires load_language_encoder() to have been called first.

        Args:
            goal_text: e.g. "navigate to (0.43, 0.57)"
        """
        assert self.language_encoder is not None, (
            "Call load_language_encoder(projection_path) first"
        )
        self._goal_emb = self.language_encoder.encode_text(goal_text)

    def plan_from_text(
        self, obs_image_np: np.ndarray, goal_text: str, record_timing: bool = True
    ) -> "PlanResult":
        """Plan from observation image + text goal (convenience wrapper)."""
        self.set_goal_text(goal_text)
        return self.plan(obs_image_np, record_timing=record_timing)

    @torch.inference_mode()
    def plan(self, obs_image_np: np.ndarray, goal_image_np: np.ndarray = None,
             record_timing: bool = True) -> "PlanResult":
        """Plan an action from observation (and optionally goal) images.

        Returns a PlanResult with the action and confidence signals.
        Backward compatible: supports numpy array protocol, so
        result.reshape(...) and np.array(result) return the action.

        Args:
            obs_image_np: (H, W, 3) uint8 observation image
            goal_image_np: (H, W, 3) uint8 goal image (optional if set_goal called)
            record_timing: whether to record per-component timing

        Returns:
            PlanResult with action, confidence, terminal_embedding, etc.
        """
        t_total_start = time.perf_counter()

        # Preprocess
        t0 = time.perf_counter()
        obs_tensor = self.preprocess(obs_image_np)
        if goal_image_np is not None:
            self.set_goal(goal_image_np)
        torch.cuda.synchronize()
        t_preprocess = (time.perf_counter() - t0) * 1000

        # Encode observation
        t0 = time.perf_counter()
        obs_emb = self.encode(obs_tensor)
        self._obs_emb = obs_emb  # cache for scorer's progress signal
        torch.cuda.synchronize()
        t_encode = (time.perf_counter() - t0) * 1000

        # CEM planning with cached embeddings — always get cost + terminal emb
        t0 = time.perf_counter()
        action, terminal_emb, best_cost = self._cem_plan(
            obs_emb, self._goal_emb, return_terminal_emb=True, return_cost=True
        )
        torch.cuda.synchronize()
        t_cem = (time.perf_counter() - t0) * 1000

        # Planability: how easy is it to keep planning from the predicted future?
        t0 = time.perf_counter()
        planability = self._score_state(terminal_emb, self._goal_emb, n_rounds=1)
        t_planability = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_total_start) * 1000

        if record_timing:
            self.timing["preprocess_ms"].append(t_preprocess)
            self.timing["encode_ms"].append(t_encode)
            self.timing["cem_ms"].append(t_cem)
            self.timing["planability_ms"].append(t_planability)
            self.timing["total_ms"].append(t_total)

        # Normalize confidence: 1.0 = cost is 0, 0.0 = cost >= cost_scale
        confidence = 1.0 - min(best_cost / self.cost_scale, 1.0)

        return PlanResult(
            action=action,
            planning_cost=best_cost,
            confidence=confidence,
            terminal_embedding=terminal_emb,
            planability=planability,
            planning_ms=t_total,
            replan_threshold=self.replan_threshold,
        )

    # ------------------------------------------------------------------
    # Legacy CEM API — thin shims that delegate to self.solver (CEMSolver).
    # These exist so dream_tree.py and other callers continue to work
    # during migration. New code should call self.solver directly.
    # ------------------------------------------------------------------

    def _cem_plan(self, obs_emb: torch.Tensor, goal_emb: torch.Tensor,
                  return_terminal_emb: bool = False, return_cost: bool = False):
        """Legacy shim — delegates to self.solver.plan."""
        # Keep solver hyperparams in sync with pipeline (cem_steps may have
        # been overridden externally, e.g. by DreamTreePlanner).
        self.solver.n_steps = self.n_steps
        self.solver.num_samples = self.num_samples
        self.solver.horizon = self.horizon
        self.solver.topk = self.topk
        self.solver.history_size = self.history_size
        return self.solver.plan(
            obs_emb,
            goal_emb,
            return_terminal_emb=return_terminal_emb,
            return_cost=return_cost,
            obs_emb_for_scorer=self._obs_emb,
        )

    def _score_state(self, obs_emb: torch.Tensor, goal_emb: torch.Tensor,
                     n_rounds: int = 1) -> float:
        """Legacy shim — delegates to self.solver.score_state."""
        self.solver.n_steps = self.n_steps
        self.solver.num_samples = self.num_samples
        self.solver.horizon = self.horizon
        self.solver.topk = self.topk
        self.solver.history_size = self.history_size
        return self.solver.score_state(obs_emb, goal_emb, n_rounds=n_rounds)

    def _cem_plan_batched(self, obs_emb: torch.Tensor, goal_emb: torch.Tensor,
                          return_terminal_emb: bool = True):
        """Legacy shim — delegates to self.solver.plan_batched."""
        self.solver.n_steps = self.n_steps
        self.solver.num_samples = self.num_samples
        self.solver.horizon = self.horizon
        self.solver.topk = self.topk
        self.solver.history_size = self.history_size
        return self.solver.plan_batched(
            obs_emb, goal_emb, return_terminal_emb=return_terminal_emb
        )

    def _evaluate_candidates(
        self, obs_emb, goal_emb, candidates, S, H, return_embs: bool = False
    ):
        """Legacy shim — delegates to self.solver.evaluate_candidates."""
        self.solver.horizon = self.horizon
        self.solver.history_size = self.history_size
        return self.solver.evaluate_candidates(
            obs_emb,
            goal_emb,
            candidates,
            S=S,
            H=H,
            return_embs=return_embs,
            obs_emb_for_scorer=self._obs_emb,
        )

    def get_timing_summary(self) -> dict:
        """Return timing statistics."""
        if not self.timing["total_ms"]:
            return {}

        summary = {}
        for key, values in self.timing.items():
            arr = np.array(values)
            summary[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p50": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
            }

        summary["effective_hz"] = 1000.0 / summary["total_ms"]["mean"]
        return summary

    def reset_timing(self):
        for key in self.timing:
            self.timing[key].clear()
