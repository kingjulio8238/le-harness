"""
S1.5 Control Loop: Full VLM → LeHarness → Motor Policy integration.

Orchestrates the three-layer robotics stack:
  S2 (VLM) provides goals → S1.5 (LeHarness) plans → S1 (motor policy) executes

With closed-loop feedback:
  - Low confidence → ask VLM to replan
  - Drift detected → ask VLM to replan

Components implement protocols from harness/protocols.py:
  - VLMProtocol: MockVLM (tests), SimVLM (sim), (future) RealVLM
  - MotorProtocol: MockMotorPolicy (tests), SimMotorPolicy (sim)

Usage (on-pod with real components):
    from harness.sim_components import SimVLM, SimMotorPolicy
    from harness.s15_loop import S15ControlLoop

    vlm = SimVLM(pipeline, goal_image=goal_img, dataset=dataset, ...)
    motor = SimMotorPolicy(world, process, pipeline._action_dim)
    loop = S15ControlLoop(pipeline, vlm, motor)

    stats = loop.run_episode(initial_obs=obs_img, max_steps=100)

Usage (unit tests with mocks):
    from harness.s15_loop import MockVLM, MockMotorPolicy

    vlm = MockVLM(goal_embedding=emb)
    motor = MockMotorPolicy()
    loop = S15ControlLoop(pipeline, vlm, motor)

    stats = loop.run_episode(initial_obs=obs, max_steps=10)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from harness.drift_detector import DriftDetector
from harness.plan_result import PlanResult

if TYPE_CHECKING:
    from harness.protocols import VLMProtocol, MotorProtocol


# ==================== Mock Components (for unit tests) ====================


class MockVLM:
    """Mock S2 VLM for unit tests. Implements VLMProtocol.

    Provides fixed/noisy goals without real model inference.
    For real evaluation, use SimVLM from harness/sim_components.py.
    """

    def __init__(
        self,
        goal_image: np.ndarray = None,
        goal_embedding: torch.Tensor = None,
        replan_strategy: str = "same",
    ):
        self.goal_image = goal_image
        self.goal_embedding = goal_embedding
        self.replan_strategy = replan_strategy

        self._replan_count = 0
        self._replan_history: list[dict] = []
        self._replan_callback = None

    @property
    def replan_count(self) -> int:
        return self._replan_count

    @property
    def replan_history(self) -> list[dict]:
        return self._replan_history

    def on_replan(self, callback):
        """Register a custom replan callback: fn(reason, obs, **kwargs) -> goal."""
        self._replan_callback = callback
        self.replan_strategy = "callback"

    def get_initial_goal(self):
        if self.goal_embedding is not None:
            return {"type": "embedding", "value": self.goal_embedding}
        return {"type": "image", "value": self.goal_image}

    def replan(self, reason: str, obs: np.ndarray = None, **kwargs):
        self._replan_count += 1
        self._replan_history.append({"reason": reason, "step": kwargs.get("step"), **kwargs})

        if self.replan_strategy == "same":
            return self.get_initial_goal()
        elif self.replan_strategy == "noisy" and self.goal_embedding is not None:
            noise = torch.randn_like(self.goal_embedding) * 0.01
            return {"type": "embedding", "value": self.goal_embedding + noise}
        elif self.replan_strategy == "callback" and self._replan_callback is not None:
            return self._replan_callback(reason, obs, **kwargs)

        return self.get_initial_goal()

    def reset(self):
        self._replan_count = 0
        self._replan_history.clear()


class MockMotorPolicy:
    """Mock S1 motor policy for unit tests. Implements MotorProtocol.

    Records actions and returns random observations. For real evaluation,
    use SimMotorPolicy from harness/sim_components.py.
    """

    def __init__(self, obs_shape=(224, 224, 3)):
        self._history: list[np.ndarray] = []
        self._obs_shape = obs_shape
        self._is_success = False

    def execute(self, action: np.ndarray) -> np.ndarray:
        """Record action and return a random observation."""
        self._history.append(action.copy())
        return np.random.randint(0, 255, self._obs_shape, dtype=np.uint8)

    @property
    def is_success(self) -> bool:
        return self._is_success

    @property
    def execution_count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> list[np.ndarray]:
        return self._history

    def reset(self):
        self._history.clear()
        self._is_success = False


# ==================== Episode Statistics ====================


@dataclass
class EpisodeStats:
    """Statistics from a single S1.5 episode."""

    steps: int = 0
    success: bool = False
    replans_confidence: int = 0
    replans_drift: int = 0
    drift_events: int = 0
    mean_confidence: float = 0.0
    mean_planning_cost: float = 0.0
    mean_drift_mse: float = 0.0
    total_planning_ms: float = 0.0

    # Per-step history
    confidences: list = field(default_factory=list)
    planning_costs: list = field(default_factory=list)
    drift_mses: list = field(default_factory=list)

    @property
    def total_replans(self) -> int:
        return self.replans_confidence + self.replans_drift

    def finalize(self):
        """Compute summary stats from per-step data."""
        if self.confidences:
            self.mean_confidence = float(np.mean(self.confidences))
        if self.planning_costs:
            self.mean_planning_cost = float(np.mean(self.planning_costs))
        if self.drift_mses:
            self.mean_drift_mse = float(np.mean(self.drift_mses))


# ==================== Control Loop ====================


class S15ControlLoop:
    """S1.5 control loop orchestrator.

    Runs the full three-layer stack:
        S2 (VLM) → S1.5 (LeHarness) → PlanResult → S1 (motor)
    with closed-loop confidence and drift feedback to S2.

    Accepts any VLM/motor implementing VLMProtocol/MotorProtocol:
        - MockVLM + MockMotorPolicy for unit tests
        - SimVLM + SimMotorPolicy for on-pod sim evaluation
        - (future) RealVLM + RealMotorPolicy for hardware
    """

    def __init__(
        self,
        pipeline,
        vlm: "VLMProtocol",
        motor: "MotorProtocol",
        drift_threshold: float = float("inf"),
        drift_window: int = 5,
        max_replans_per_episode: int = 10,
    ):
        self.pipeline = pipeline
        self.vlm = vlm
        self.motor = motor
        self.drift_detector = DriftDetector(
            threshold=drift_threshold, window=drift_window
        )
        self.max_replans = max_replans_per_episode

    def _set_goal(self, goal_dict: dict):
        """Set pipeline goal from VLM output."""
        if goal_dict["type"] == "embedding":
            self.pipeline.set_goal_embedding(goal_dict["value"])
        elif goal_dict["type"] == "image":
            self.pipeline.set_goal(goal_dict["value"])
        else:
            raise ValueError(f"Unknown goal type: {goal_dict['type']}")

    def run_episode(
        self,
        initial_obs: np.ndarray,
        max_steps: int = 100,
    ) -> EpisodeStats:
        """Run a single S1.5 episode.

        The motor policy handles environment stepping and returns observations.
        Success is determined by motor.is_success.

        Args:
            initial_obs: (H, W, 3) starting observation image.
            max_steps: maximum planning steps per episode.

        Returns:
            EpisodeStats with per-step tracking data.
        """
        stats = EpisodeStats()
        self.vlm.reset()
        self.motor.reset()
        self.drift_detector.reset()

        # S2: Get initial goal
        goal_dict = self.vlm.get_initial_goal()
        self._set_goal(goal_dict)

        obs = initial_obs
        prev_result = None

        for step in range(max_steps):
            # S1.5: Plan
            result = self.pipeline.plan(obs)

            # Track stats
            stats.steps += 1
            stats.confidences.append(result.confidence)
            stats.planning_costs.append(result.planning_cost)
            stats.total_planning_ms += result.planning_ms

            # Check confidence → replan if needed
            if result.needs_replan and self.vlm.replan_count < self.max_replans:
                stats.replans_confidence += 1
                new_goal = self.vlm.replan(
                    reason="low_confidence",
                    obs=obs,
                    step=step,
                    planning_cost=result.planning_cost,
                    confidence=result.confidence,
                )
                self._set_goal(new_goal)
                continue  # re-plan with new goal before executing

            # S1: Execute action → get next observation from motor
            next_obs = self.motor.execute(result.action)

            # Drift detection (compare predicted terminal vs actual)
            if prev_result is not None:
                drift_signal = self.drift_detector.check(
                    predicted=prev_result.terminal_embedding,
                    actual_emb=self.pipeline.encode(
                        self.pipeline.preprocess(next_obs)
                    ),
                )
                stats.drift_mses.append(drift_signal.drift_mse)

                if drift_signal.drift_exceeded:
                    stats.drift_events += 1

                if (drift_signal.escalate_to_s2
                        and self.vlm.replan_count < self.max_replans):
                    stats.replans_drift += 1
                    new_goal = self.vlm.replan(
                        reason="drift_detected",
                        obs=next_obs,
                        step=step,
                        drift_mse=drift_signal.drift_mse,
                    )
                    self._set_goal(new_goal)

            prev_result = result
            obs = next_obs

            # Check success via motor policy
            if self.motor.is_success:
                stats.success = True
                break
            # Check truncation (env hit max_episode_steps)
            if hasattr(self.motor, 'is_done') and self.motor.is_done:
                break

        stats.finalize()
        return stats
