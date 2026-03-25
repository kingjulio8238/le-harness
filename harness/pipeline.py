"""
Phase 7: End-to-End Planning Pipeline

Clean API wrapping the full LeHarness planning stack:
  observation image → preprocessing → compiled encoder → CEM planning → action

Usage:
    from harness.pipeline import PlanningPipeline

    pipeline = PlanningPipeline("pusht/lejepa")
    pipeline.warmup()  # triggers torch.compile (one-time)

    # Plan from raw images
    action = pipeline.plan(obs_image_np, goal_image_np)

    # Or use in eval loop
    pipeline.set_goal(goal_image_np)
    for obs in observations:
        action = pipeline.plan(obs)
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

from harness.compiled_inference import optimize_model


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
    ):
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.horizon = horizon
        self.history_size = history_size
        self.topk = topk
        self.device = device

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
            mode="reduce-overhead",
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
        self._compiled = False
        self._action_dim = 10  # PushT: 2 * action_block(5)

        # Timing stats
        self.timing = {
            "preprocess_ms": [],
            "encode_ms": [],
            "cem_ms": [],
            "total_ms": [],
        }

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
        """Set and cache the goal embedding."""
        goal_tensor = self.preprocess(goal_image_np)
        self._goal_emb = self.encode(goal_tensor)

    @torch.inference_mode()
    def plan(self, obs_image_np: np.ndarray, goal_image_np: np.ndarray = None,
             record_timing: bool = True) -> np.ndarray:
        """Plan an action from observation (and optionally goal) images.

        Args:
            obs_image_np: (H, W, 3) uint8 observation image
            goal_image_np: (H, W, 3) uint8 goal image (optional if set_goal called)
            record_timing: whether to record per-component timing

        Returns:
            action: (action_dim,) numpy array
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
        torch.cuda.synchronize()
        t_encode = (time.perf_counter() - t0) * 1000

        # CEM planning with cached embeddings
        t0 = time.perf_counter()
        action = self._cem_plan(obs_emb, self._goal_emb)
        torch.cuda.synchronize()
        t_cem = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_total_start) * 1000

        if record_timing:
            self.timing["preprocess_ms"].append(t_preprocess)
            self.timing["encode_ms"].append(t_encode)
            self.timing["cem_ms"].append(t_cem)
            self.timing["total_ms"].append(t_total)

        return action

    def _cem_plan(self, obs_emb: torch.Tensor, goal_emb: torch.Tensor) -> np.ndarray:
        """Run CEM optimization with cached embeddings.

        Args:
            obs_emb: (1, 1, 192) observation embedding
            goal_emb: (1, 1, 192) goal embedding

        Returns:
            action: (action_dim,) numpy array — first action from best plan
        """
        S = self.num_samples
        H = 1  # history length (from obs)
        T = H + self.horizon
        D = obs_emb.shape[-1]
        device = obs_emb.device

        # Initialize CEM distribution
        mean = torch.zeros(1, T, self._action_dim, device=device)
        var = torch.ones(1, T, self._action_dim, device=device)

        for cem_iter in range(self.n_steps):
            # Sample candidates
            noise = torch.randn(1, S, T, self._action_dim, device=device)
            candidates = noise * var.unsqueeze(1) + mean.unsqueeze(1)
            candidates[:, 0] = mean  # mean is always a candidate

            # Evaluate: rollout + cost
            costs = self._evaluate_candidates(obs_emb, goal_emb, candidates, S, H)

            # Select top-k
            topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
            topk_cands = candidates[0, topk_inds[0]]  # (topk, T, action_dim)

            # Update distribution
            mean = topk_cands.mean(dim=0, keepdim=True)
            var = topk_cands.std(dim=0, keepdim=True)

        # Return first action from mean plan
        return mean[0, 0].cpu().numpy()

    def _evaluate_candidates(
        self, obs_emb, goal_emb, candidates, S, H
    ) -> torch.Tensor:
        """Evaluate candidate action sequences via rollout + MSE cost."""
        B = 1
        horizon = self.horizon

        # Expand obs for all samples
        emb = obs_emb.unsqueeze(1).expand(B, S, -1, -1)
        emb = rearrange(emb, "b s t d -> (b s) t d").clone()

        # Split candidates
        act_0 = candidates[:, :, :H, :]
        act_future = candidates[:, :, H:, :]
        act = rearrange(act_0, "b s t d -> (b s) t d")
        act_future_flat = rearrange(act_future, "b s t d -> (b s) t d")

        HS = self.history_size

        # Autoregressive rollout
        for t in range(horizon):
            start = max(0, emb.shape[1] - HS)
            act_emb = self.model.action_encoder(act[:, start:, :])
            pred = self.model.predict(emb[:, start:, :], act_emb)[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
            act = torch.cat([act, act_future_flat[:, t : t + 1, :]], dim=1)

        # Final predict
        start = max(0, emb.shape[1] - HS)
        act_emb = self.model.action_encoder(act[:, start:, :])
        pred = self.model.predict(emb[:, start:, :], act_emb)[:, -1:]
        emb = torch.cat([emb, pred], dim=1)

        # Cost: MSE between last predicted embedding and goal
        pred_emb = rearrange(emb, "(b s) t d -> b s t d", b=B, s=S)
        goal_exp = goal_emb[:, -1:, :].unsqueeze(1).expand(B, S, 1, -1)
        cost = ((pred_emb[:, :, -1:, :] - goal_exp) ** 2).sum(dim=-1).squeeze(-1)

        return cost

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
