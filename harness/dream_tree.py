"""
D3: Dream Trees — tree-structured lookahead using compiled CEM.

Uses the pipeline's compiled _cem_plan for root action generation and
cheap single-pass scoring for depth evaluation. Tree structure provides
lookahead by scoring how plannable each root candidate's future state is.

Architecture:
  Root: K calls to _cem_plan → K diverse (action, terminal_emb) pairs
  Depth scoring: _score_state on each terminal_emb (single pass, no CEM)
  Select: root action whose terminal state has the lowest score

The key insight: flat CEM picks the action with the best immediate cost.
Dream Tree picks the action whose predicted future is easiest to plan from.

Usage:
    from harness.pipeline import PlanningPipeline
    from harness.dream_tree import DreamTreePlanner

    pipeline = PlanningPipeline("pusht/lejepa")
    pipeline.warmup()

    tree_planner = DreamTreePlanner(pipeline)
    action = tree_planner.plan(obs_image, goal_image)
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class DreamNode:
    """A node in the dream tree."""
    latent_state: torch.Tensor       # (1, 1, D) predicted embedding
    action: np.ndarray | None        # (action_dim,) first action of the CEM plan
    cost: float = float("inf")       # MSE cost from this node's CEM
    depth: int = 0
    children: list = field(default_factory=list)
    value: float = float("inf")      # backpropagated value

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class DreamTreePlanner:
    """Tree-structured planner built on pipeline's compiled CEM.

    For each planning step:
    1. Run K root CEM calls → K diverse (action, terminal_emb) pairs
    2. For each terminal_emb, run cheap depth scoring (single pass, no CEM)
    3. Pick the root action whose depth score is lowest
    """

    def __init__(
        self,
        pipeline,
        num_roots: int = 4,
        max_depth: int = 2,
        cheap_depth: bool = True,
    ):
        self.pipeline = pipeline
        self.device = pipeline.device
        self.num_roots = num_roots
        self.max_depth = max_depth
        self.cheap_depth = cheap_depth
        self._action_dim = pipeline._action_dim

        # Cache uncompiled predictor for flexible depth scoring
        # (compiled predictor uses CUDA graphs with fixed shapes)
        predictor = pipeline.model.predictor
        self._uncompiled_predictor = (
            predictor._orig_mod if hasattr(predictor, '_orig_mod') else None
        )

        self.timing = {"total_ms": [], "root_ms": [], "expansion_ms": []}
        self.stats = {"total_cem_calls": [], "total_nodes": []}

    @torch.inference_mode()
    def plan(self, obs_image_np: np.ndarray, goal_image_np: np.ndarray) -> np.ndarray:
        """Plan via dream tree search."""
        t_start = time.perf_counter()

        # Encode
        obs_tensor = self.pipeline.preprocess(obs_image_np)
        goal_tensor = self.pipeline.preprocess(goal_image_np)
        obs_emb = self.pipeline.encode(obs_tensor)
        goal_emb = self.pipeline.encode(goal_tensor)

        # Phase 1: Generate diverse root candidates via full CEM
        t_root = time.perf_counter()
        root_candidates = []
        for _ in range(self.num_roots):
            action, terminal_emb = self.pipeline._cem_plan(
                obs_emb, goal_emb, return_terminal_emb=True
            )
            cost = self._cost(terminal_emb, goal_emb)
            root_candidates.append((action, cost, terminal_emb))
        t_root = (time.perf_counter() - t_root) * 1000

        cem_calls = self.num_roots

        # Phase 2: Score each root candidate's future
        t_expand = time.perf_counter()

        best_action = root_candidates[0][0]
        best_value = float("inf")

        # Cheap depth uses _score_state (S=128, same shape as compiled path).
        # Full depth uses _cem_plan which needs return_terminal_emb (already compiled).
        # No predictor swap needed for either path.

        for action, root_cost, terminal_emb in root_candidates:
            if self.max_depth >= 2:
                if self.cheap_depth:
                    # Mini-CEM scoring: 3 rounds of 128 samples (vs 15 for full CEM)
                    d2_cost = self.pipeline._score_state(terminal_emb, goal_emb, n_rounds=3)
                else:
                    # Full CEM at depth (compiled, slower)
                    _, d2_terminal = self.pipeline._cem_plan(
                        terminal_emb, goal_emb, return_terminal_emb=True
                    )
                    d2_cost = self._cost(d2_terminal, goal_emb)
                cem_calls += 1

                if self.max_depth >= 3:
                    if self.cheap_depth:
                        value = d2_cost
                    else:
                        _, d3_terminal = self.pipeline._cem_plan(
                            d2_terminal, goal_emb, return_terminal_emb=True
                        )
                        d3_cost = self._cost(d3_terminal, goal_emb)
                        cem_calls += 1
                        value = d3_cost
                else:
                    value = d2_cost
            else:
                value = root_cost

            if value < best_value:
                best_value = value
                best_action = action

        # No swap needed — both paths use compiled-compatible shapes

        t_expand = (time.perf_counter() - t_expand) * 1000
        t_total = (time.perf_counter() - t_start) * 1000

        self.timing["total_ms"].append(t_total)
        self.timing["root_ms"].append(t_root)
        self.timing["expansion_ms"].append(t_expand)
        self.stats["total_cem_calls"].append(cem_calls)
        self.stats["total_nodes"].append(1 + self.num_roots * min(self.max_depth, 2))

        return best_action

    def _cost(self, emb, goal_emb):
        """MSE cost between embedding and goal."""
        return float(((emb - goal_emb) ** 2).sum())

    def get_timing_summary(self):
        if not self.timing["total_ms"]:
            return {}

        summary = {}
        for key, values in self.timing.items():
            arr = np.array(values)
            summary[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p50": float(np.median(arr)),
            }

        summary["effective_hz"] = (
            1000.0 / summary["total_ms"]["mean"]
            if summary["total_ms"]["mean"] > 0 else 0
        )

        for key, values in self.stats.items():
            summary[key] = float(np.mean(values)) if values else 0

        return summary

    def reset_timing(self):
        for key in self.timing:
            self.timing[key].clear()
        for key in self.stats:
            self.stats[key].clear()
