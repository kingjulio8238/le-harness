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
    2. For each terminal_emb, run depth scoring via CEM
    3. Pick the root action whose depth-2 future cost is lowest

    Supports both sequential mode (CUDA-graph compatible, B=1 per call)
    and batched mode (B=K per call, faster but requires flexible compile).
    """

    def __init__(
        self,
        pipeline,
        num_roots: int = 4,
        max_depth: int = 2,
        cheap_depth: bool = True,
        batched: bool = False,
        cem_steps: int | None = None,
    ):
        self.pipeline = pipeline
        self.solver = pipeline.solver  # public CEMSolver handle
        self.device = pipeline.device
        self.num_roots = num_roots
        self.max_depth = max_depth
        self.cheap_depth = cheap_depth
        self.batched = batched
        self._action_dim = pipeline.action_dim
        # Batched CEM converges in ~7 steps (1% cost gap vs 15).
        # Default to 7 when batched to hit <300ms gate.
        self.cem_steps = cem_steps if cem_steps is not None else (7 if batched else None)

        self.timing = {"total_ms": [], "root_ms": [], "expansion_ms": []}
        self.stats = {"total_cem_calls": [], "total_nodes": []}

    @torch.inference_mode()
    def plan(self, obs_image_np: np.ndarray, goal_image_np: np.ndarray) -> np.ndarray:
        """Plan via dream tree search."""
        t_start = time.perf_counter()

        # Encode
        obs_tensor = self.pipeline.preprocess(obs_image_np)
        goal_tensor = self.pipeline.preprocess(goal_image_np)
        obs_emb = self.pipeline.encode(obs_tensor)   # (1, 1, D)
        goal_emb = self.pipeline.encode(goal_tensor)  # (1, 1, D)

        # Temporarily override solver CEM iterations if cem_steps is set.
        orig_n_steps = None
        if self.cem_steps is not None:
            orig_n_steps = self.solver.n_steps
            self.solver.n_steps = self.cem_steps
            # Mirror to pipeline.n_steps so the legacy shims stay consistent.
            self.pipeline.n_steps = self.cem_steps

        if self.batched:
            action = self._plan_batched(obs_emb, goal_emb)
        else:
            action = self._plan_sequential(obs_emb, goal_emb)

        if orig_n_steps is not None:
            self.solver.n_steps = orig_n_steps
            self.pipeline.n_steps = orig_n_steps

        t_total = (time.perf_counter() - t_start) * 1000
        self.timing["total_ms"].append(t_total)

        return action

    def _plan_batched(self, obs_emb, goal_emb):
        """Batched tree: root and depth CEM calls run as batched GPU ops."""
        K = self.num_roots

        # Phase 1: Batched root CEM — K parallel CEM instances
        t_root = time.perf_counter()
        batched_obs = obs_emb.expand(K, -1, -1)    # (K, 1, D)
        batched_goal = goal_emb.expand(K, -1, -1)  # (K, 1, D)

        root_actions, root_terminal_embs = self.solver.plan_batched(
            batched_obs, batched_goal, return_terminal_emb=True
        )
        # root_actions: (K, action_dim) numpy
        # root_terminal_embs: (K, 1, D) tensor
        t_root = (time.perf_counter() - t_root) * 1000
        self.timing["root_ms"].append(t_root)

        cem_calls = 1  # one batched call

        # Phase 2: Batched depth scoring
        t_expand = time.perf_counter()

        if self.max_depth >= 2:
            if self.cheap_depth:
                # Score each root's future with mini-CEM (sequential — score_state is B=1)
                depth_costs = []
                for i in range(K):
                    c = self.solver.score_state(
                        root_terminal_embs[i:i+1], goal_emb, n_rounds=3
                    )
                    depth_costs.append(c)
                cem_calls += K
            else:
                # Full batched CEM at depth
                depth_goal = goal_emb.expand(K, -1, -1)  # (K, 1, D)
                _, depth_terminal_embs = self.solver.plan_batched(
                    root_terminal_embs, depth_goal, return_terminal_emb=True
                )
                # Score: MSE between depth terminal and goal
                depth_costs = [
                    self._cost(depth_terminal_embs[i:i+1], goal_emb)
                    for i in range(K)
                ]
                cem_calls += 1  # one batched call
        else:
            depth_costs = [
                self._cost(root_terminal_embs[i:i+1], goal_emb)
                for i in range(K)
            ]

        t_expand = (time.perf_counter() - t_expand) * 1000
        self.timing["expansion_ms"].append(t_expand)

        # Select best root
        best_idx = int(np.argmin(depth_costs))
        best_action = root_actions[best_idx]

        self.stats["total_cem_calls"].append(cem_calls)
        self.stats["total_nodes"].append(1 + K * min(self.max_depth, 2))

        return best_action

    def _plan_sequential(self, obs_emb, goal_emb):
        """Sequential tree: original implementation with one CEM call at a time."""
        # Phase 1: Generate diverse root candidates via full CEM
        t_root = time.perf_counter()
        root_candidates = []
        for _ in range(self.num_roots):
            action, terminal_emb, _ = self.solver.plan(
                obs_emb, goal_emb,
                return_terminal_emb=True, return_cost=True,
                obs_emb_for_scorer=self.pipeline.obs_emb,
            )
            cost = self._cost(terminal_emb, goal_emb)
            root_candidates.append((action, cost, terminal_emb))
        t_root = (time.perf_counter() - t_root) * 1000
        self.timing["root_ms"].append(t_root)

        cem_calls = self.num_roots

        # Phase 2: Score each root candidate's future
        t_expand = time.perf_counter()

        best_action = root_candidates[0][0]
        best_value = float("inf")

        for action, root_cost, terminal_emb in root_candidates:
            if self.max_depth >= 2:
                if self.cheap_depth:
                    d2_cost = self.solver.score_state(terminal_emb, goal_emb, n_rounds=3)
                else:
                    _, d2_terminal, _ = self.solver.plan(
                        terminal_emb, goal_emb,
                        return_terminal_emb=True, return_cost=True,
                    )
                    d2_cost = self._cost(d2_terminal, goal_emb)
                cem_calls += 1

                if self.max_depth >= 3:
                    if self.cheap_depth:
                        value = d2_cost
                    else:
                        _, d3_terminal, _ = self.solver.plan(
                            d2_terminal, goal_emb,
                            return_terminal_emb=True, return_cost=True,
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

        t_expand = (time.perf_counter() - t_expand) * 1000
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
