"""
Real (non-mock) S1.5 components for simulation environments.

SimVLM: Dataset-grounded goal provider — provides real goals from the dataset
    and selects alternative goals on replan requests.

SimMotorPolicy: Environment-stepping motor policy — executes actions through
    world.envs.step() and returns real observations from world.envs.render().

Both implement the protocols defined in harness/protocols.py and can be used
directly with S15ControlLoop for on-pod evaluation with zero mocks.
"""

import numpy as np
import torch
from omegaconf import OmegaConf


class SimVLM:
    """Dataset-grounded S2 VLM that provides real goals.

    Provides initial goal from the dataset. On replan requests, uses
    one of several strategies to provide an adjusted goal.

    Replan strategies:
        "nearby" — select an earlier goal from the dataset (original behavior)
        "waypoint" — provide a midpoint subgoal between current obs and goal
        "persist" — return the same goal (VLM insists, let planner keep trying)

    This is not a mock: it uses real dataset images/embeddings and
    real pipeline encoding.
    """

    def __init__(
        self,
        pipeline,
        goal_image: np.ndarray,
        dataset=None,
        episode_indices: np.ndarray = None,
        goal_step: int = None,
        start_step: int = None,
        replan_offset: int = 5,
        replan_strategy: str = "waypoint",
    ):
        """
        Args:
            pipeline: PlanningPipeline for encoding goal images.
            goal_image: (H, W, 3) uint8 primary goal image.
            dataset: HDF5Dataset (optional — enables replan with alternative goals).
            episode_indices: indices for the current episode in the dataset.
            goal_step: step index of the primary goal within the episode.
            start_step: step index of the start state within the episode.
            replan_offset: for "nearby" strategy, how many steps earlier to sample.
            replan_strategy: "nearby" | "waypoint" | "persist"
        """
        self.pipeline = pipeline
        self.goal_image = goal_image
        self.dataset = dataset
        self.episode_indices = episode_indices
        self.goal_step = goal_step
        self.start_step = start_step
        self.replan_offset = replan_offset
        self.replan_strategy = replan_strategy

        # Encode the primary goal
        with torch.inference_mode():
            goal_tensor = pipeline.preprocess(goal_image)
            self._goal_emb = pipeline.encode(goal_tensor)

        self._replan_count = 0
        self._replan_history: list[dict] = []

    @property
    def replan_count(self) -> int:
        return self._replan_count

    @property
    def replan_history(self) -> list[dict]:
        return self._replan_history

    def get_initial_goal(self) -> dict:
        """Return the primary goal embedding (encoded from real image)."""
        return {"type": "embedding", "value": self._goal_emb}

    def replan(self, reason: str, obs: np.ndarray = None, **kwargs) -> dict:
        """Replan using the configured strategy.

        Args:
            reason: "low_confidence" or "drift_detected"
            obs: current observation image (used by "waypoint" strategy)
            **kwargs: context (planning_cost, confidence, step, etc.)
        """
        self._replan_count += 1
        self._replan_history.append({
            "reason": reason,
            "step": kwargs.get("step"),
            "strategy": self.replan_strategy,
            **kwargs,
        })

        if self.replan_strategy == "nearby":
            return self._replan_nearby()
        elif self.replan_strategy == "waypoint":
            return self._replan_waypoint(obs)
        elif self.replan_strategy == "persist":
            return self._replan_persist()
        else:
            return self._replan_persist()

    def _replan_nearby(self) -> dict:
        """Select an earlier goal from the dataset."""
        if self.dataset is not None and self.episode_indices is not None and self.goal_step is not None:
            alt_step = max(0, self.goal_step - self.replan_offset * self._replan_count)
            alt_step = min(alt_step, len(self.episode_indices) - 1)
            alt_row = self.dataset.get_row_data(int(self.episode_indices[alt_step]))
            alt_image = alt_row["pixels"]
            if isinstance(alt_image, np.ndarray) and alt_image.dtype != np.uint8:
                alt_image = (alt_image * 255).astype(np.uint8) if alt_image.max() <= 1.0 else alt_image.astype(np.uint8)

            with torch.inference_mode():
                alt_tensor = self.pipeline.preprocess(alt_image)
                alt_emb = self.pipeline.encode(alt_tensor)
            return {"type": "embedding", "value": alt_emb}

        return self._replan_persist()

    def _replan_waypoint(self, obs: np.ndarray = None) -> dict:
        """Provide a midpoint subgoal between current observation and final goal.

        If dataset is available, selects the observation at the midpoint between
        the current position in the episode and the goal step. This gives the
        planner a closer, more achievable target.

        Falls back to persist if obs is not available or dataset is missing.
        """
        if (self.dataset is not None and self.episode_indices is not None
                and self.goal_step is not None and self.start_step is not None):
            # Estimate current position: start + elapsed steps
            current_est = min(
                self.start_step + (self._replan_count * 5),  # rough estimate
                self.goal_step - 1,
            )
            # Midpoint between estimated current position and goal
            mid_step = (current_est + self.goal_step) // 2
            mid_step = max(0, min(mid_step, len(self.episode_indices) - 1))

            mid_row = self.dataset.get_row_data(int(self.episode_indices[mid_step]))
            mid_image = mid_row["pixels"]
            if isinstance(mid_image, np.ndarray) and mid_image.dtype != np.uint8:
                mid_image = (mid_image * 255).astype(np.uint8) if mid_image.max() <= 1.0 else mid_image.astype(np.uint8)

            with torch.inference_mode():
                mid_tensor = self.pipeline.preprocess(mid_image)
                mid_emb = self.pipeline.encode(mid_tensor)
            return {"type": "embedding", "value": mid_emb}

        return self._replan_persist()

    def _replan_persist(self) -> dict:
        """Return the same goal embedding (VLM re-confirms its goal)."""
        return {"type": "embedding", "value": self._goal_emb}

    def reset(self):
        self._replan_count = 0
        self._replan_history.clear()


class SimMotorPolicy:
    """Environment-stepping S1 motor policy.

    Wraps a sim environment (world.envs) to execute actions through
    world.envs.step() and return real observations from world.envs.render().

    This is not a mock: it actually steps the environment and renders
    real pixel observations.
    """

    def __init__(self, world, process: dict, action_dim: int):
        """
        Args:
            world: swm.World instance with world.envs gym environment.
            process: dict of sklearn scalers (must contain "action" key).
            action_dim: total action dimension from pipeline (frameskip * raw_dim).
        """
        self.world = world
        self.process = process
        self._action_dim = action_dim

        raw_action_dim = process["action"].scale_.shape[0] if "action" in process else 2
        self._action_block = action_dim // raw_action_dim
        self._raw_action_dim = raw_action_dim

        self._history: list[np.ndarray] = []
        self._is_success = False
        self._truncated = False
        self._env_steps = 0

    @property
    def execution_count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> list[np.ndarray]:
        return self._history

    @property
    def is_success(self) -> bool:
        return self._is_success

    @property
    def env_steps(self) -> int:
        """Number of raw environment steps (including frameskip sub-steps)."""
        return self._env_steps

    def execute(self, action: np.ndarray) -> np.ndarray:
        """Execute action through the sim environment.

        Reshapes the action into sub-actions (for frameskip), applies
        inverse scaling, steps the environment, and returns the rendered
        observation.

        Args:
            action: (action_dim,) action from the planner.

        Returns:
            (H, W, 3) uint8 next observation image.
        """
        self._history.append(action.copy())

        sub_actions = action.reshape(self._action_block, self._raw_action_dim)
        for sub_action in sub_actions:
            if "action" in self.process:
                sub_action = self.process["action"].inverse_transform(
                    sub_action.reshape(1, -1)
                ).squeeze()

            obs_dict, reward, terminated, truncated, info = self.world.envs.step(
                np.array([sub_action])
            )
            self._env_steps += 1

            if isinstance(terminated, (list, np.ndarray)):
                terminated = bool(terminated[0])
            if isinstance(truncated, (list, np.ndarray)):
                truncated = bool(truncated[0])
            if isinstance(info, (list, tuple)):
                step_info = info[0] if info else {}
            elif isinstance(info, dict):
                step_info = info
            else:
                step_info = {}

            if terminated or step_info.get("is_success", False):
                self._is_success = True
                break

            if truncated:
                # Episode hit max_episode_steps — env auto-resets, stop stepping
                self._truncated = True
                break

        obs = self.world.envs.render()[0]
        return obs

    @property
    def is_done(self) -> bool:
        """Whether the episode is over (success or truncation)."""
        return self._is_success or self._truncated

    def reset(self):
        self._history.clear()
        self._is_success = False
        self._truncated = False
        self._env_steps = 0

    def reset_env(self, cfg, dataset, episode_idx, col_name,
                  ep_id, start_step, goal_offset):
        """Reset the environment for a new episode.

        Follows the eval_dream_tree.py pattern: world.envs.reset() +
        apply callables from config.

        Returns:
            (start_pixels, goal_pixels, ep_indices, goal_step)
        """
        ep_mask = episode_idx == ep_id
        ep_indices = np.where(ep_mask)[0]

        start_row = dataset.get_row_data(int(ep_indices[start_step]))
        goal_step_idx = min(start_step + goal_offset, len(ep_indices) - 1)
        goal_row = dataset.get_row_data(int(ep_indices[goal_step_idx]))

        start_pixels = start_row["pixels"]
        goal_pixels = goal_row["pixels"]

        if isinstance(start_pixels, np.ndarray) and start_pixels.dtype != np.uint8:
            start_pixels = (start_pixels * 255).astype(np.uint8) if start_pixels.max() <= 1.0 else start_pixels.astype(np.uint8)
        if isinstance(goal_pixels, np.ndarray) and goal_pixels.dtype != np.uint8:
            goal_pixels = (goal_pixels * 255).astype(np.uint8) if goal_pixels.max() <= 1.0 else goal_pixels.astype(np.uint8)

        # Reset env — clear autoreset flag that gymnasium sets after terminated=True
        # Walk to the SyncVectorEnv (has _autoreset_envs) and clear it BEFORE reset
        env = self.world.envs
        while hasattr(env, 'env'):
            if hasattr(env, '_autoreset_envs'):
                env._autoreset_envs[:] = False
            env = env.env
        if hasattr(env, '_autoreset_envs'):
            env._autoreset_envs[:] = False
        self.world.envs.reset()
        unwrapped_env = self.world.envs.envs[0].unwrapped

        callables = OmegaConf.to_container(
            cfg.eval.get("callables"), resolve=True
        ) if cfg.eval.get("callables") else []

        for spec in callables:
            method_name = spec["method"]
            if not hasattr(unwrapped_env, method_name):
                continue
            method = getattr(unwrapped_env, method_name)
            prepared_args = {}
            for arg_name, arg_data in spec.get("args", {}).items():
                value_key = arg_data.get("value", None)
                if value_key is None:
                    continue
                if value_key.startswith("goal_"):
                    col = value_key[5:]
                    if col in goal_row:
                        prepared_args[arg_name] = goal_row[col]
                else:
                    if value_key in start_row:
                        prepared_args[arg_name] = start_row[value_key]
            if prepared_args:
                method(**prepared_args)

        self.reset()  # reset motor state for new episode

        return start_pixels, goal_pixels, ep_indices, goal_step_idx
