"""Tests for protocols, SimVLM, and SimMotorPolicy.

SimVLM and SimMotorPolicy require real pipeline/world which are only available
on-pod. These tests verify the protocol conformance and logic using lightweight
stand-ins that match the real interfaces without GPU/model dependencies.
"""

import numpy as np
import pytest
import torch

from harness.protocols import VLMProtocol, MotorProtocol
from harness.s15_loop import MockVLM, MockMotorPolicy


# ==================== Protocol Conformance Tests ====================

class TestProtocolConformance:
    """Verify that mock/sim classes satisfy the protocol interfaces."""

    def test_mock_vlm_satisfies_protocol(self):
        vlm = MockVLM(goal_embedding=torch.randn(1, 1, 192))
        assert isinstance(vlm, VLMProtocol)

    def test_mock_motor_satisfies_protocol(self):
        motor = MockMotorPolicy()
        assert isinstance(motor, MotorProtocol)

    def test_protocol_requires_get_initial_goal(self):
        """A class without get_initial_goal doesn't satisfy VLMProtocol."""

        class BadVLM:
            replan_count = 0
            replan_history = []
            def replan(self, reason, obs=None, **kwargs): ...
            def reset(self): ...

        assert not isinstance(BadVLM(), VLMProtocol)

    def test_protocol_requires_execute(self):
        """A class without execute doesn't satisfy MotorProtocol."""

        class BadMotor:
            execution_count = 0
            history = []
            is_success = False
            def reset(self): ...

        assert not isinstance(BadMotor(), MotorProtocol)


# ==================== SimVLM Tests (with lightweight stand-in) ====================

class LightweightPipeline:
    """Minimal pipeline that encodes images without GPU.
    Matches the interface SimVLM calls: preprocess() and encode()."""

    def __init__(self):
        self.device = "cpu"
        self._encode_count = 0

    def preprocess(self, image_np):
        return torch.randn(1, 1, 3, 224, 224)

    def encode(self, tensor):
        self._encode_count += 1
        # Return deterministic embedding based on call count
        emb = torch.zeros(1, 1, 192)
        emb[0, 0, 0] = float(self._encode_count)
        return emb


class LightweightDataset:
    """Minimal dataset stand-in for SimVLM tests."""

    def __init__(self, n_rows=50):
        self._rows = {}
        for i in range(n_rows):
            self._rows[i] = {
                "pixels": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            }

    def get_row_data(self, idx):
        return self._rows.get(idx, self._rows[0])


class TestSimVLM:
    """Test SimVLM logic using lightweight pipeline/dataset (no GPU)."""

    def _make_sim_vlm(self, pipeline=None, dataset=None, replan_strategy="nearby"):
        from harness.sim_components import SimVLM

        pipeline = pipeline or LightweightPipeline()
        goal_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        ep_indices = np.arange(50)

        return SimVLM(
            pipeline=pipeline,
            goal_image=goal_img,
            dataset=dataset or LightweightDataset(),
            episode_indices=ep_indices,
            goal_step=30,
            start_step=5,
            replan_offset=5,
            replan_strategy=replan_strategy,
        )

    def test_satisfies_vlm_protocol(self):
        vlm = self._make_sim_vlm()
        assert isinstance(vlm, VLMProtocol)

    def test_initial_goal_is_embedding(self):
        vlm = self._make_sim_vlm()
        goal = vlm.get_initial_goal()
        assert goal["type"] == "embedding"
        assert goal["value"].shape == (1, 1, 192)

    def test_initial_goal_encoded_from_real_image(self):
        pipeline = LightweightPipeline()
        vlm = self._make_sim_vlm(pipeline=pipeline)
        # Constructor encodes the goal image
        assert pipeline._encode_count == 1

    def test_replan_returns_different_embedding(self):
        vlm = self._make_sim_vlm()
        initial = vlm.get_initial_goal()["value"]
        replanned = vlm.replan(reason="low_confidence")["value"]
        # Should be a different embedding (from different dataset row)
        assert replanned.shape == (1, 1, 192)

    def test_replan_increments_count(self):
        vlm = self._make_sim_vlm()
        assert vlm.replan_count == 0
        vlm.replan(reason="low_confidence", step=5)
        assert vlm.replan_count == 1
        vlm.replan(reason="drift_detected", step=10)
        assert vlm.replan_count == 2

    def test_replan_history_tracked(self):
        vlm = self._make_sim_vlm()
        vlm.replan(reason="low_confidence", step=5, planning_cost=3.0)
        vlm.replan(reason="drift_detected", step=12, drift_mse=0.5)

        assert len(vlm.replan_history) == 2
        assert vlm.replan_history[0]["reason"] == "low_confidence"
        assert vlm.replan_history[0]["step"] == 5
        assert vlm.replan_history[1]["reason"] == "drift_detected"

    def test_replan_selects_earlier_goal_step(self):
        """Each replan should select a progressively earlier goal step."""
        vlm = self._make_sim_vlm()  # goal_step=30, replan_offset=5

        vlm.replan(reason="test")  # should look at step 30 - 5*1 = 25
        vlm.replan(reason="test")  # should look at step 30 - 5*2 = 20
        vlm.replan(reason="test")  # should look at step 30 - 5*3 = 15

        # All three replans should have succeeded
        assert vlm.replan_count == 3

    def test_reset(self):
        vlm = self._make_sim_vlm()
        vlm.replan(reason="test")
        vlm.reset()
        assert vlm.replan_count == 0
        assert len(vlm.replan_history) == 0

    def test_without_dataset_returns_original_goal(self):
        """Without dataset, replan re-encodes the original goal image."""
        from harness.sim_components import SimVLM

        pipeline = LightweightPipeline()
        goal_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        vlm = SimVLM(pipeline=pipeline, goal_image=goal_img)

        result = vlm.replan(reason="test")
        assert result["type"] == "embedding"
        assert result["value"].shape == (1, 1, 192)

    # --- Strategy-specific tests ---

    def test_nearby_strategy(self):
        vlm = self._make_sim_vlm(replan_strategy="nearby")
        result = vlm.replan(reason="test")
        assert result["type"] == "embedding"
        assert result["value"].shape == (1, 1, 192)
        assert vlm.replan_history[0]["strategy"] == "nearby"

    def test_waypoint_strategy(self):
        vlm = self._make_sim_vlm(replan_strategy="waypoint")
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = vlm.replan(reason="test", obs=obs)
        assert result["type"] == "embedding"
        assert result["value"].shape == (1, 1, 192)
        assert vlm.replan_history[0]["strategy"] == "waypoint"

    def test_persist_strategy_returns_same_goal(self):
        vlm = self._make_sim_vlm(replan_strategy="persist")
        initial = vlm.get_initial_goal()["value"]
        replanned = vlm.replan(reason="test")["value"]
        assert torch.equal(initial, replanned)
        assert vlm.replan_history[0]["strategy"] == "persist"

    def test_waypoint_without_dataset_falls_back_to_persist(self):
        from harness.sim_components import SimVLM
        pipeline = LightweightPipeline()
        goal_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        vlm = SimVLM(pipeline=pipeline, goal_image=goal_img, replan_strategy="waypoint")
        initial = vlm.get_initial_goal()["value"]
        result = vlm.replan(reason="test")
        assert torch.equal(initial, result["value"])

    def test_strategy_recorded_in_history(self):
        vlm = self._make_sim_vlm(replan_strategy="waypoint")
        vlm.replan(reason="low_confidence", step=3)
        assert vlm.replan_history[0]["strategy"] == "waypoint"


# ==================== SimMotorPolicy Protocol Tests ====================

class TestSimMotorProtocol:
    """Test that SimMotorPolicy satisfies MotorProtocol.

    Full SimMotorPolicy tests require world.envs which is only available
    on-pod. Here we verify the interface compliance using a mock world.
    """

    def _make_motor(self):
        from harness.sim_components import SimMotorPolicy

        class MockWorld:
            class MockEnvs:
                def step(self, action):
                    return {}, 0.0, False, False, {}
                def render(self):
                    return [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)]
                def reset(self):
                    return {}, {}
                @property
                def envs(self):
                    return [type('E', (), {'unwrapped': type('U', (), {})()})()]

            envs = MockEnvs()

        from sklearn.preprocessing import StandardScaler
        process = {"action": StandardScaler()}
        process["action"].fit(np.random.randn(100, 2))

        return SimMotorPolicy(MockWorld(), process, action_dim=10)

    def test_satisfies_motor_protocol(self):
        motor = self._make_motor()
        assert isinstance(motor, MotorProtocol)

    def test_execute_returns_observation(self):
        motor = self._make_motor()
        action = np.random.randn(10).astype(np.float32)
        obs = motor.execute(action)
        assert obs.shape == (224, 224, 3)
        assert obs.dtype == np.uint8

    def test_execution_count(self):
        motor = self._make_motor()
        motor.execute(np.random.randn(10).astype(np.float32))
        motor.execute(np.random.randn(10).astype(np.float32))
        assert motor.execution_count == 2

    def test_history_records_actions(self):
        motor = self._make_motor()
        action = np.array([1.0] * 10, dtype=np.float32)
        motor.execute(action)
        assert len(motor.history) == 1
        np.testing.assert_array_equal(motor.history[0], action)

    def test_history_copies_actions(self):
        motor = self._make_motor()
        action = np.array([1.0] * 10, dtype=np.float32)
        motor.execute(action)
        action[0] = 999.0
        assert motor.history[0][0] == 1.0

    def test_is_success_initially_false(self):
        motor = self._make_motor()
        assert motor.is_success is False

    def test_reset(self):
        motor = self._make_motor()
        motor.execute(np.random.randn(10).astype(np.float32))
        motor.reset()
        assert motor.execution_count == 0
        assert motor.is_success is False
        assert motor.env_steps == 0

    def test_env_steps_count_frameskip(self):
        """With action_dim=10 and raw_action_dim=2, action_block=5."""
        motor = self._make_motor()
        motor.execute(np.random.randn(10).astype(np.float32))
        # action_dim=10, raw_action_dim=2 → action_block=5
        assert motor.env_steps == 5
