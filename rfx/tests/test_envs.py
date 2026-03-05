"""Tests for rfx.envs module."""

from __future__ import annotations

import numpy as np
import pytest

from rfx.envs import Box, Go2Env, VecEnv, _PythonMockBackend, make_vec_env

# ---------------------------------------------------------------------------
# Go2Env is abstract (missing render()), so we use a concrete subclass.
# ---------------------------------------------------------------------------


class _TestableGo2Env(Go2Env):
    """Concrete Go2Env with render() stub for testing."""

    def render(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------


class TestBox:
    def test_shape_and_dtype(self) -> None:
        b = Box(low=-1.0, high=1.0, shape=(4,))
        assert b.low.shape == (4,)
        assert b.high.shape == (4,)
        assert b.dtype == np.float32

    def test_scalar_bounds_broadcast(self) -> None:
        b = Box(low=0.0, high=10.0, shape=(3,))
        np.testing.assert_array_equal(b.low, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(b.high, np.full(3, 10.0, dtype=np.float32))

    def test_array_bounds(self) -> None:
        lo = np.array([-1.0, -2.0])
        hi = np.array([1.0, 2.0])
        b = Box(low=lo, high=hi, shape=(2,))
        np.testing.assert_array_equal(b.low, lo.astype(np.float32))
        np.testing.assert_array_equal(b.high, hi.astype(np.float32))

    def test_sample_within_bounds(self) -> None:
        b = Box(low=-5.0, high=5.0, shape=(10,))
        for _ in range(50):
            s = b.sample()
            assert s.shape == (10,)
            assert s.dtype == np.float32
            assert np.all(s >= b.low)
            assert np.all(s <= b.high)

    def test_contains_valid(self) -> None:
        b = Box(low=0.0, high=1.0, shape=(3,))
        assert b.contains(np.array([0.5, 0.5, 0.5], dtype=np.float32))

    def test_contains_out_of_bounds(self) -> None:
        b = Box(low=0.0, high=1.0, shape=(3,))
        assert not b.contains(np.array([0.5, 1.5, 0.5], dtype=np.float32))

    def test_contains_wrong_shape(self) -> None:
        b = Box(low=0.0, high=1.0, shape=(3,))
        assert not b.contains(np.array([0.5, 0.5], dtype=np.float32))

    def test_clip(self) -> None:
        b = Box(low=-1.0, high=1.0, shape=(3,))
        x = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
        clipped = b.clip(x)
        np.testing.assert_array_equal(clipped, np.array([-1.0, 0.0, 1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# _PythonMockBackend
# ---------------------------------------------------------------------------


class TestPythonMockBackend:
    def test_initial_state(self) -> None:
        backend = _PythonMockBackend()
        np.testing.assert_array_equal(backend.joint_positions, Go2Env.DEFAULT_STANDING)
        np.testing.assert_array_equal(backend.joint_velocities, np.zeros(12))
        assert backend.time == 0.0

    def test_reset(self) -> None:
        backend = _PythonMockBackend()
        # Mutate state
        backend.joint_positions[:] = 99.0
        backend.time = 10.0
        backend.reset()
        np.testing.assert_array_equal(backend.joint_positions, Go2Env.DEFAULT_STANDING)
        assert backend.time == 0.0

    def test_step_moves_joints(self) -> None:
        backend = _PythonMockBackend()
        target = Go2Env.DEFAULT_STANDING + 0.1
        backend.step(target)
        # After one step, positions should have moved toward target
        assert not np.allclose(backend.joint_positions, Go2Env.DEFAULT_STANDING)
        assert backend.time > 0.0

    def test_step_clamps_to_limits(self) -> None:
        backend = _PythonMockBackend()
        # Push way beyond limits
        target = np.full(12, 100.0)
        for _ in range(500):
            backend.step(target)
        assert np.all(backend.joint_positions <= Go2Env.JOINT_LIMITS_HIGH)
        assert np.all(backend.joint_positions >= Go2Env.JOINT_LIMITS_LOW)

    def test_velocity_clamp(self) -> None:
        backend = _PythonMockBackend()
        target = np.full(12, 100.0)
        for _ in range(100):
            backend.step(target)
        assert np.all(backend.joint_velocities <= 20.0)
        assert np.all(backend.joint_velocities >= -20.0)


# ---------------------------------------------------------------------------
# Go2Env (sim mode with Python mock fallback)
# ---------------------------------------------------------------------------


class TestGo2Env:
    def test_creates_in_sim_mode(self) -> None:
        env = _TestableGo2Env(sim=True)
        assert env.sim is True
        assert env.observation_space.shape == (48,)
        assert env.action_space.shape == (12,)

    def test_reset_returns_correct_shape(self) -> None:
        env = _TestableGo2Env(sim=True)
        obs = env.reset()
        assert obs.shape == (48,)
        assert obs.dtype == np.float32

    def test_step_returns_tuple(self) -> None:
        env = _TestableGo2Env(sim=True)
        env.reset()
        action = np.zeros(12, dtype=np.float32)
        result = env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert obs.shape == (48,)
        assert np.isscalar(reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "step" in info

    def test_step_clips_action(self) -> None:
        env = _TestableGo2Env(sim=True)
        env.reset()
        # Pass action outside [-1, 1] — should be clipped internally
        big_action = np.full(12, 5.0, dtype=np.float32)
        obs, _, _, _ = env.step(big_action)
        # Should not crash; last_action should be clipped
        np.testing.assert_array_equal(env._last_action, np.ones(12))

    def test_step_count_increments(self) -> None:
        env = _TestableGo2Env(sim=True)
        env.reset()
        assert env._step_count == 0
        env.step(np.zeros(12))
        assert env._step_count == 1
        env.step(np.zeros(12))
        assert env._step_count == 2

    def test_reset_clears_state(self) -> None:
        env = _TestableGo2Env(sim=True)
        env.reset()
        env.step(np.ones(12))
        env.step(np.ones(12))
        env.reset()
        assert env._step_count == 0
        assert env._phase == 0.0
        np.testing.assert_array_equal(env._last_action, np.zeros(12))

    def test_set_commands(self) -> None:
        env = _TestableGo2Env(sim=True)
        env.set_commands(vx=1.0, vy=-0.5, yaw_rate=0.3)
        np.testing.assert_array_almost_equal(env._commands, [1.0, -0.5, 0.3])

    def test_commands_appear_in_observation(self) -> None:
        env = _TestableGo2Env(sim=True)
        env.set_commands(vx=2.0, vy=0.0, yaw_rate=0.0)
        obs = env.reset()
        # Commands are at indices 30:33 (after joint_pos=12, joint_vel=12, ang_vel=3, gravity=3)
        np.testing.assert_array_almost_equal(obs[30:33], [2.0, 0.0, 0.0])

    def test_reward_includes_survival_bonus(self) -> None:
        env = _TestableGo2Env(sim=True)
        env.reset()
        # Zero action → reward should be ~0.1 (survival bonus, no action penalty)
        _, reward, _, _ = env.step(np.zeros(12))
        assert reward == pytest.approx(0.1)

    def test_reward_penalizes_large_actions(self) -> None:
        env = _TestableGo2Env(sim=True)
        env.reset()
        _, r_zero, _, _ = env.step(np.zeros(12))
        env.reset()
        _, r_big, _, _ = env.step(np.ones(12))
        assert r_big < r_zero

    def test_custom_control_dt(self) -> None:
        env = _TestableGo2Env(sim=True, control_dt=0.01)
        assert env.control_dt == 0.01

    def test_custom_action_scale(self) -> None:
        env = _TestableGo2Env(sim=True, action_scale=0.25)
        assert env.action_scale == 0.25

    def test_close_does_not_raise(self) -> None:
        env = _TestableGo2Env(sim=True)
        env.reset()
        env.close()  # Should not raise


# ---------------------------------------------------------------------------
# VecEnv
# ---------------------------------------------------------------------------


class TestVecEnv:
    def _make_vec(self, n: int = 3) -> VecEnv:
        return VecEnv(lambda: _TestableGo2Env(sim=True), num_envs=n)

    def test_num_envs(self) -> None:
        vec = self._make_vec(4)
        assert vec.num_envs == 4
        assert len(vec.envs) == 4

    def test_spaces_match_single_env(self) -> None:
        vec = self._make_vec(2)
        assert vec.observation_space.shape == (48,)
        assert vec.action_space.shape == (12,)

    def test_reset_shape(self) -> None:
        vec = self._make_vec(3)
        obs = vec.reset()
        assert obs.shape == (3, 48)
        assert obs.dtype == np.float32

    def test_step_shapes(self) -> None:
        vec = self._make_vec(3)
        vec.reset()
        actions = np.zeros((3, 12), dtype=np.float32)
        obs, rewards, dones, infos = vec.step(actions)
        assert obs.shape == (3, 48)
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert len(infos) == 3

    def test_close_all(self) -> None:
        vec = self._make_vec(2)
        vec.reset()
        vec.close()  # Should not raise


# ---------------------------------------------------------------------------
# make_vec_env
# ---------------------------------------------------------------------------


class TestMakeVecEnv:
    def test_single_env_returns_base(self) -> None:
        env = make_vec_env(_TestableGo2Env, num_envs=1, sim=True)
        assert isinstance(env, Go2Env)
        env.close()

    def test_multi_env_returns_vec(self) -> None:
        env = make_vec_env(_TestableGo2Env, num_envs=3, sim=True)
        assert isinstance(env, VecEnv)
        assert env.num_envs == 3
        env.close()

    def test_kwargs_forwarded(self) -> None:
        env = make_vec_env(_TestableGo2Env, num_envs=1, sim=True, action_scale=0.1)
        assert isinstance(env, Go2Env)
        assert env.action_scale == 0.1
        env.close()
