"""
Protocols (interfaces) for the S1.5 control loop components.

Any class implementing these protocols can be used with S15ControlLoop —
mock, sim, or real hardware.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class VLMProtocol(Protocol):
    """Interface for S2 (VLM) goal providers.

    Implementations:
        - MockVLM: returns fixed/noisy goals for unit tests
        - SimVLM: provides goals from a real dataset, replans with alternatives
        - (future) RealVLM: wraps an actual VLM API
    """

    @property
    def replan_count(self) -> int: ...

    @property
    def replan_history(self) -> list[dict]: ...

    def get_initial_goal(self) -> dict:
        """Return the initial goal.

        Returns:
            {"type": "embedding"|"image", "value": tensor|ndarray}
        """
        ...

    def replan(self, reason: str, obs: np.ndarray = None, **kwargs) -> dict:
        """Handle a replan request from S1.5.

        Args:
            reason: "low_confidence" or "drift_detected"
            obs: current observation image
            **kwargs: context (planning_cost, drift_mse, step, etc.)

        Returns:
            {"type": "embedding"|"image", "value": tensor|ndarray}
        """
        ...

    def reset(self) -> None: ...


@runtime_checkable
class MotorProtocol(Protocol):
    """Interface for S1 motor policies.

    Implementations:
        - MockMotorPolicy: records actions without executing (unit tests)
        - SimMotorPolicy: wraps a sim environment (world.envs.step)
        - (future) RealMotorPolicy: sends commands to real robot hardware
    """

    @property
    def execution_count(self) -> int: ...

    @property
    def history(self) -> list[np.ndarray]: ...

    def execute(self, action: np.ndarray) -> np.ndarray:
        """Execute an action and return the next observation.

        Args:
            action: (action_dim,) action to execute

        Returns:
            (H, W, 3) uint8 next observation image
        """
        ...

    @property
    def is_success(self) -> bool:
        """Whether the task is complete after the last execution."""
        ...

    def reset(self) -> None: ...
