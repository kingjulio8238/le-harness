"""
D3: Dream Trees — tree-structured lookahead using compiled CEM.

Uses the pipeline's existing compiled _cem_plan as the atomic operation.
Tree structure provides lookahead by expanding promising root actions
into deeper plans, then backpropagating scores to select the best
root action.

Architecture:
  Root: pipeline._cem_plan → best action + terminal embedding
  Depth 2: run _cem_plan from terminal embedding → child cost
  Repeat for K diverse root candidates (from CEM's final elite set)
  Select root action whose subtree has the lowest cost

The key insight: flat CEM picks the root action with the lowest
immediate cost. Dream Tree picks the root action whose *future*
(after re-planning from the predicted state) has the lowest cost.
This should prefer actions that lead to states where planning
is easier, even if the immediate cost is slightly higher.

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
    1. Run root CEM → get best action + terminal embedding
    2. Run root CEM again with different seeds to get K diverse candidates
    3. For each candidate, run CEM from its terminal state (depth 2 lookahead)
    4. Pick the root candidate whose depth-2 cost is lowest
    """

    def __init__(
        self,
        pipeline,
        num_roots: int = 4,
        max_depth: int = 2,
    ):
        self.pipeline = pipeline
        self.device = pipeline.device
        self.num_roots = num_roots
        self.max_depth = max_depth
        self._action_dim = pipeline._action_dim

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

        # Phase 1: Generate diverse root candidates
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

        # Phase 2: Expand each root candidate to depth 2+
        t_expand = time.perf_counter()

        best_action = root_candidates[0][0]
        best_value = float("inf")

        for action, root_cost, terminal_emb in root_candidates:
            if self.max_depth >= 2:
                # Run CEM from predicted terminal state
                _, d2_terminal = self.pipeline._cem_plan(
                    terminal_emb, goal_emb, return_terminal_emb=True
                )
                d2_cost = self._cost(d2_terminal, goal_emb)
                cem_calls += 1

                if self.max_depth >= 3:
                    _, d3_terminal = self.pipeline._cem_plan(
                        d2_terminal, goal_emb, return_terminal_emb=True
                    )
                    d3_cost = self._cost(d3_terminal, goal_emb)
                    cem_calls += 1
                    # Value = deepest cost — prefer actions with best futures
                    value = d3_cost
                else:
                    # Value = depth-2 cost — prefer actions leading to plannable states
                    value = d2_cost
            else:
                value = root_cost

            if value < best_value:
                best_value = value
                best_action = action

        t_expand = (time.perf_counter() - t_expand) * 1000
        t_total = (time.perf_counter() - t_start) * 1000

        self.timing["total_ms"].append(t_total)
        self.timing["root_ms"].append(t_root)
        self.timing["expansion_ms"].append(t_expand)
        self.stats["total_cem_calls"].append(cem_calls)
        self.stats["total_nodes"].append(1 + self.num_roots * self.max_depth)

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
