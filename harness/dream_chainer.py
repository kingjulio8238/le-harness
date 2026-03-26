"""
D2: Dream Chaining

Chains K sequential CEM solves for long-horizon planning.
Each chain targets a subgoal interpolated linearly in latent space
between the current state and the final goal.

Usage:
    from harness.pipeline import PlanningPipeline
    from harness.dream_chainer import DreamChainer

    pipeline = PlanningPipeline("pusht/lejepa")
    pipeline.warmup()

    chainer = DreamChainer(pipeline, num_chains=3)
    action = chainer.plan(obs_image, goal_image)
"""

import time

import numpy as np
import torch


class DreamChainer:
    """Chains multiple CEM planning horizons for long-range planning.

    Each chain:
      1. Interpolates a subgoal between current embedding and final goal
      2. Runs CEM to plan toward that subgoal
      3. Extracts the predicted terminal embedding as the start for the next chain

    In receding-horizon mode (default), only the first chain's action is returned
    and executed. The full chain provides lookahead that informs the first action.
    """

    def __init__(self, pipeline, num_chains: int = 3):
        """
        Args:
            pipeline: A PlanningPipeline instance (already warmed up)
            num_chains: Number of sequential CEM solves to chain
        """
        self.pipeline = pipeline
        self.num_chains = num_chains

        # Timing
        self.timing = {
            "chain_ms": [],
            "total_ms": [],
            "per_chain_ms": [],
        }

        # Drift tracking (populated when measure_drift=True)
        self.drift_data = []

    def plan(self, obs_image_np: np.ndarray, goal_image_np: np.ndarray,
             return_all_actions: bool = False,
             measure_drift: bool = False):
        """Plan via chained dreams.

        Args:
            obs_image_np: (H, W, 3) uint8 current observation
            goal_image_np: (H, W, 3) uint8 goal image
            return_all_actions: if True, return actions from all chains (for analysis)
            measure_drift: if True, also return predicted terminal embedding
                from chain 1 for drift comparison

        Returns:
            action from the first chain (receding horizon), or list of all
            chain actions if return_all_actions=True.
            If measure_drift=True, returns (action, predicted_terminal_emb).
        """
        t_start = time.perf_counter()

        with torch.inference_mode():
            # Encode observation and goal
            obs_tensor = self.pipeline.preprocess(obs_image_np)
            goal_tensor = self.pipeline.preprocess(goal_image_np)

            obs_emb = self.pipeline.encode(obs_tensor)   # (1, 1, D)
            goal_emb = self.pipeline.encode(goal_tensor)  # (1, 1, D)

            # Generate subgoals via linear interpolation in latent space
            subgoals = self._interpolate_subgoals(obs_emb, goal_emb, self.num_chains)

            # Chain CEM solves
            current_emb = obs_emb
            chain_actions = []
            chain_times = []
            chain_terminal_embs = []

            for i in range(self.num_chains):
                t_chain = time.perf_counter()

                # CEM plans from current_emb toward subgoal[i]
                action, terminal_emb = self.pipeline._cem_plan(
                    current_emb, subgoals[i], return_terminal_emb=True
                )

                chain_actions.append(action)
                chain_times.append((time.perf_counter() - t_chain) * 1000)
                chain_terminal_embs.append(terminal_emb)

                # The next chain starts from where this dream ended
                current_emb = terminal_emb

        t_total = (time.perf_counter() - t_start) * 1000

        # Record timing
        self.timing["total_ms"].append(t_total)
        self.timing["per_chain_ms"].append(chain_times)
        self.timing["chain_ms"].append(sum(chain_times))

        if return_all_actions:
            return chain_actions

        first_action = chain_actions[0]
        if measure_drift:
            return first_action, chain_terminal_embs[0]

        # Receding horizon: execute only the first chain's action
        return first_action

    @torch.inference_mode()
    def plan_from_embeddings(self, obs_emb: torch.Tensor, goal_emb: torch.Tensor,
                             return_all_actions: bool = False):
        """Plan from pre-computed embeddings (for use in eval loops where
        the environment provides embeddings directly).

        Args:
            obs_emb: (1, 1, D) observation embedding
            goal_emb: (1, 1, D) goal embedding

        Returns:
            Same as plan()
        """
        subgoals = self._interpolate_subgoals(obs_emb, goal_emb, self.num_chains)

        current_emb = obs_emb
        chain_actions = []

        for i in range(self.num_chains):
            action, terminal_emb = self.pipeline._cem_plan(
                current_emb, subgoals[i], return_terminal_emb=True
            )
            chain_actions.append(action)
            current_emb = terminal_emb

        if return_all_actions:
            return chain_actions
        return chain_actions[0]

    @staticmethod
    def _interpolate_subgoals(obs_emb: torch.Tensor, goal_emb: torch.Tensor,
                              num_chains: int) -> list:
        """Linearly interpolate subgoals in latent space.

        For K chains, generates K subgoals:
          subgoal[0] = lerp(obs, goal, 1/K)
          subgoal[1] = lerp(obs, goal, 2/K)
          ...
          subgoal[K-1] = goal  (final chain always targets the actual goal)

        Args:
            obs_emb: (1, 1, D) starting embedding
            goal_emb: (1, 1, D) final goal embedding
            num_chains: number of chains

        Returns:
            List of K tensors, each (1, 1, D)
        """
        # All chains target the final goal directly — no interpolated subgoals.
        # Interpolation produced semantically invalid waypoints that degraded planning.
        return [goal_emb] * num_chains

    def get_timing_summary(self) -> dict:
        """Return timing statistics."""
        if not self.timing["total_ms"]:
            return {}

        total = np.array(self.timing["total_ms"])
        chain = np.array(self.timing["chain_ms"])

        summary = {
            "total_ms": {
                "mean": float(np.mean(total)),
                "std": float(np.std(total)),
                "p50": float(np.median(total)),
            },
            "chain_ms": {
                "mean": float(np.mean(chain)),
                "std": float(np.std(chain)),
            },
            "num_chains": self.num_chains,
            "effective_hz": 1000.0 / float(np.mean(total)) if np.mean(total) > 0 else 0,
        }

        # Per-chain breakdown
        if self.timing["per_chain_ms"]:
            per_chain = np.array(self.timing["per_chain_ms"])
            summary["per_chain_mean_ms"] = [
                float(np.mean(per_chain[:, i])) for i in range(per_chain.shape[1])
            ]

        return summary

    def reset_timing(self):
        for key in self.timing:
            self.timing[key].clear()
        self.drift_data.clear()
