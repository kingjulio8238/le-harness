"""Tests for S1.5 control loop, MockVLM, and MockMotorPolicy."""

import numpy as np
import pytest
import torch

from harness.plan_result import PlanResult
from harness.s15_loop import S15ControlLoop, MockVLM, MockMotorPolicy, EpisodeStats


# --- Mock pipeline that returns controllable PlanResults ---

class MockPipeline:
    """Pipeline mock that returns configurable PlanResults."""

    def __init__(self, action_dim=10, confidence=0.8, planning_cost=2.0):
        self.device = "cpu"
        self._goal_emb = None
        self._action_dim = action_dim
        self._default_confidence = confidence
        self._default_cost = planning_cost
        self._plan_count = 0
        self._step_overrides = {}

    def preprocess(self, img):
        return torch.randn(1, 1, 3, 224, 224)

    def encode(self, tensor):
        return torch.randn(1, 1, 192)

    def set_goal(self, goal_image_np):
        self._goal_emb = torch.randn(1, 1, 192)

    def set_goal_embedding(self, emb):
        if emb.dim() == 1:
            emb = emb.unsqueeze(0).unsqueeze(0)
        elif emb.dim() == 2:
            emb = emb.unsqueeze(1)
        self._goal_emb = emb.float()

    def plan(self, obs_image_np=None, goal_image_np=None, record_timing=True):
        step = self._plan_count
        self._plan_count += 1

        overrides = self._step_overrides.get(step, {})
        confidence = overrides.get("confidence", self._default_confidence)
        cost = overrides.get("planning_cost", self._default_cost)
        replan_threshold = overrides.get("replan_threshold", 0.3)

        return PlanResult(
            action=np.random.randn(self._action_dim).astype(np.float32),
            planning_cost=cost,
            confidence=confidence,
            terminal_embedding=torch.randn(1, 1, 192),
            planability=1.5,
            planning_ms=50.0,
            replan_threshold=replan_threshold,
        )

    def override_step(self, step: int, **kwargs):
        self._step_overrides[step] = kwargs

    def reset_plan_count(self):
        self._plan_count = 0
        self._step_overrides.clear()


class SuccessMotorPolicy(MockMotorPolicy):
    """MockMotorPolicy that reports success after N executions."""

    def __init__(self, succeed_after: int = 3):
        super().__init__()
        self._succeed_after = succeed_after

    def execute(self, action: np.ndarray) -> np.ndarray:
        obs = super().execute(action)
        if self.execution_count >= self._succeed_after:
            self._is_success = True
        return obs


# ==================== MockVLM Tests ====================

class TestMockVLM:
    def test_initial_goal_image(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        vlm = MockVLM(goal_image=img)
        goal = vlm.get_initial_goal()
        assert goal["type"] == "image"
        np.testing.assert_array_equal(goal["value"], img)

    def test_initial_goal_embedding(self):
        emb = torch.randn(1, 1, 192)
        vlm = MockVLM(goal_embedding=emb)
        goal = vlm.get_initial_goal()
        assert goal["type"] == "embedding"
        assert torch.equal(goal["value"], emb)

    def test_replan_same_strategy(self):
        emb = torch.randn(1, 1, 192)
        vlm = MockVLM(goal_embedding=emb, replan_strategy="same")
        result = vlm.replan(reason="low_confidence")
        assert torch.equal(result["value"], emb)
        assert vlm.replan_count == 1

    def test_replan_noisy_strategy(self):
        emb = torch.randn(1, 1, 192)
        vlm = MockVLM(goal_embedding=emb, replan_strategy="noisy")
        result = vlm.replan(reason="drift_detected")
        assert result["type"] == "embedding"
        assert not torch.equal(result["value"], emb)
        assert vlm.replan_count == 1

    def test_replan_callback(self):
        vlm = MockVLM(goal_embedding=torch.randn(1, 1, 192))
        custom_emb = torch.randn(1, 1, 192)

        def my_callback(reason, obs, **kwargs):
            return {"type": "embedding", "value": custom_emb}

        vlm.on_replan(my_callback)
        result = vlm.replan(reason="test")
        assert torch.equal(result["value"], custom_emb)

    def test_replan_history(self):
        vlm = MockVLM(goal_embedding=torch.randn(1, 1, 192))
        vlm.replan(reason="low_confidence", step=5, planning_cost=3.0)
        vlm.replan(reason="drift_detected", step=12, drift_mse=0.5)

        assert vlm.replan_count == 2
        assert vlm.replan_history[0]["reason"] == "low_confidence"
        assert vlm.replan_history[0]["step"] == 5
        assert vlm.replan_history[1]["reason"] == "drift_detected"

    def test_reset(self):
        vlm = MockVLM(goal_embedding=torch.randn(1, 1, 192))
        vlm.replan(reason="test")
        vlm.reset()
        assert vlm.replan_count == 0
        assert len(vlm.replan_history) == 0


# ==================== MockMotorPolicy Tests ====================

class TestMockMotorPolicy:
    def test_execute_records_and_returns_obs(self):
        motor = MockMotorPolicy()
        action = np.array([1.0, 2.0, 3.0])
        obs = motor.execute(action)
        assert motor.execution_count == 1
        np.testing.assert_array_equal(motor.history[0], action)
        assert obs.shape == (224, 224, 3)
        assert obs.dtype == np.uint8

    def test_execute_copies(self):
        motor = MockMotorPolicy()
        action = np.array([1.0, 2.0])
        motor.execute(action)
        action[0] = 999.0
        assert motor.history[0][0] == 1.0

    def test_is_success_default_false(self):
        motor = MockMotorPolicy()
        assert motor.is_success is False

    def test_reset(self):
        motor = MockMotorPolicy()
        motor.execute(np.zeros(3))
        motor.reset()
        assert motor.execution_count == 0
        assert motor.is_success is False


# ==================== EpisodeStats Tests ====================

class TestEpisodeStats:
    def test_total_replans(self):
        stats = EpisodeStats(replans_confidence=3, replans_drift=2)
        assert stats.total_replans == 5

    def test_finalize(self):
        stats = EpisodeStats()
        stats.confidences = [0.5, 0.7, 0.9]
        stats.planning_costs = [3.0, 2.0, 1.0]
        stats.drift_mses = [0.1, 0.2]
        stats.finalize()
        assert stats.mean_confidence == pytest.approx(0.7, abs=0.01)
        assert stats.mean_planning_cost == pytest.approx(2.0, abs=0.01)
        assert stats.mean_drift_mse == pytest.approx(0.15, abs=0.01)

    def test_finalize_empty(self):
        stats = EpisodeStats()
        stats.finalize()
        assert stats.mean_confidence == 0.0


# ==================== S15ControlLoop Tests ====================

class TestS15ControlLoop:
    def _make_loop(self, pipeline=None, vlm=None, motor=None, **kwargs):
        pipeline = pipeline or MockPipeline()
        vlm = vlm or MockVLM(goal_embedding=torch.randn(1, 1, 192))
        motor = motor or MockMotorPolicy()
        return S15ControlLoop(pipeline, vlm, motor, **kwargs)

    def test_runs_to_max_steps(self):
        loop = self._make_loop()
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        stats = loop.run_episode(obs, max_steps=10)
        assert stats.steps == 10
        assert stats.success is False

    def test_success_stops_early(self):
        motor = SuccessMotorPolicy(succeed_after=3)
        loop = self._make_loop(motor=motor)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        stats = loop.run_episode(obs, max_steps=100)
        assert stats.success is True
        assert stats.steps == 3

    def test_confidence_replan_triggers(self):
        pipeline = MockPipeline(confidence=0.1)
        loop = self._make_loop(pipeline=pipeline, max_replans_per_episode=3)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        stats = loop.run_episode(obs, max_steps=10)
        assert stats.replans_confidence > 0
        assert stats.replans_confidence <= 3

    def test_no_replan_on_high_confidence(self):
        pipeline = MockPipeline(confidence=0.9)
        loop = self._make_loop(pipeline=pipeline, drift_threshold=1e6)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        stats = loop.run_episode(obs, max_steps=5)
        assert stats.replans_confidence == 0
        assert stats.replans_drift == 0

    def test_motor_receives_actions(self):
        motor = MockMotorPolicy()
        pipeline = MockPipeline(confidence=0.9)
        loop = self._make_loop(pipeline=pipeline, motor=motor)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        loop.run_episode(obs, max_steps=5)
        assert motor.execution_count == 5

    def test_stats_tracking(self):
        pipeline = MockPipeline(confidence=0.75, planning_cost=2.5)
        loop = self._make_loop(pipeline=pipeline)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        stats = loop.run_episode(obs, max_steps=5)
        assert len(stats.confidences) == 5
        assert len(stats.planning_costs) == 5
        assert stats.mean_confidence == pytest.approx(0.75, abs=0.01)
        assert stats.total_planning_ms > 0

    def test_goal_set_from_embedding(self):
        pipeline = MockPipeline()
        emb = torch.randn(1, 1, 192)
        vlm = MockVLM(goal_embedding=emb)
        loop = self._make_loop(pipeline=pipeline, vlm=vlm)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        loop.run_episode(obs, max_steps=1)
        assert pipeline._goal_emb is not None

    def test_goal_set_from_image(self):
        pipeline = MockPipeline()
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        vlm = MockVLM(goal_image=img)
        loop = self._make_loop(pipeline=pipeline, vlm=vlm)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        loop.run_episode(obs, max_steps=1)
        assert pipeline._goal_emb is not None

    def test_replan_with_step_specific_confidence(self):
        pipeline = MockPipeline(confidence=0.9)
        pipeline.override_step(3, confidence=0.1)

        loop = self._make_loop(pipeline=pipeline, max_replans_per_episode=5)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        stats = loop.run_episode(obs, max_steps=6)
        assert stats.replans_confidence == 1

    def test_max_replans_respected(self):
        pipeline = MockPipeline(confidence=0.1)
        loop = self._make_loop(pipeline=pipeline, max_replans_per_episode=2)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        stats = loop.run_episode(obs, max_steps=20)
        assert stats.replans_confidence <= 2

    def test_steps_equals_motor_plus_replans(self):
        pipeline = MockPipeline(confidence=0.9)
        pipeline.override_step(2, confidence=0.1)
        pipeline.override_step(4, confidence=0.1)
        motor = MockMotorPolicy()
        loop = self._make_loop(
            pipeline=pipeline, motor=motor,
            max_replans_per_episode=10, drift_threshold=1e6,
        )
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        stats = loop.run_episode(obs, max_steps=8)
        assert stats.steps == 8
        assert motor.execution_count == stats.steps - stats.replans_confidence

    def test_drift_replan_triggers(self):
        pipeline = MockPipeline(confidence=0.9)
        loop = self._make_loop(
            pipeline=pipeline,
            drift_threshold=0.001,
            drift_window=2,
            max_replans_per_episode=10,
        )
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        stats = loop.run_episode(obs, max_steps=10)
        assert stats.drift_events > 0

    def test_vlm_reset_verified(self):
        pipeline = MockPipeline(confidence=0.1)
        vlm = MockVLM(goal_embedding=torch.randn(1, 1, 192))
        motor = MockMotorPolicy()
        loop = self._make_loop(
            pipeline=pipeline, vlm=vlm, motor=motor,
            max_replans_per_episode=3, drift_threshold=1e6,
        )
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # First episode
        stats1 = loop.run_episode(obs, max_steps=5)
        assert stats1.total_replans > 0

        # Second episode — VLM was reset
        pipeline.reset_plan_count()
        stats2 = loop.run_episode(obs, max_steps=5)
        assert vlm.replan_count <= 3
        assert motor.execution_count == stats2.steps - stats2.replans_confidence

    def test_zero_max_steps(self):
        loop = self._make_loop()
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        stats = loop.run_episode(obs, max_steps=0)
        assert stats.steps == 0
        assert stats.success is False

    def test_unknown_goal_type_raises(self):
        loop = self._make_loop()
        with pytest.raises(ValueError, match="Unknown goal type"):
            loop._set_goal({"type": "unknown", "value": None})
