"""Integration tests for pi Rust bindings.

These tests require the Rust extension to be built and installed.
They verify that the Python bindings work correctly with the Rust core.
"""

import math
import pytest
from typing import Tuple

# Skip all tests if the Rust extension is not available
pytest.importorskip("_pi", reason="Rust extension not built")


class TestQuaternion:
    """Tests for the Quaternion class from Rust bindings."""

    def test_identity(self) -> None:
        """Test identity quaternion."""
        from _pi import Quaternion

        q = Quaternion.identity()

        assert q.w == pytest.approx(1.0)
        assert q.x == pytest.approx(0.0)
        assert q.y == pytest.approx(0.0)
        assert q.z == pytest.approx(0.0)

    def test_from_euler(self) -> None:
        """Test creating quaternion from Euler angles."""
        from _pi import Quaternion

        q = Quaternion.from_euler(0.0, 0.0, 0.0)

        assert q.w == pytest.approx(1.0, abs=1e-6)

    def test_to_euler_roundtrip(self) -> None:
        """Test Euler angle roundtrip conversion."""
        from _pi import Quaternion

        roll_in, pitch_in, yaw_in = 0.1, 0.2, 0.3
        q = Quaternion.from_euler(roll_in, pitch_in, yaw_in)
        roll_out, pitch_out, yaw_out = q.to_euler()

        assert roll_out == pytest.approx(roll_in, abs=1e-6)
        assert pitch_out == pytest.approx(pitch_in, abs=1e-6)
        assert yaw_out == pytest.approx(yaw_in, abs=1e-6)

    def test_rotate_vector(self) -> None:
        """Test vector rotation."""
        from _pi import Quaternion

        # 90 degree rotation around Z axis
        q = Quaternion.from_euler(0.0, 0.0, math.pi / 2)
        v = (1.0, 0.0, 0.0)
        rotated = q.rotate_vector(v)

        assert rotated[0] == pytest.approx(0.0, abs=1e-6)
        assert rotated[1] == pytest.approx(1.0, abs=1e-6)
        assert rotated[2] == pytest.approx(0.0, abs=1e-6)

    def test_slerp(self) -> None:
        """Test spherical interpolation."""
        from _pi import Quaternion

        q1 = Quaternion.identity()
        q2 = Quaternion.from_euler(0.0, 0.0, math.pi)

        q_half = q1.slerp(q2, 0.5)

        # Halfway between no rotation and 180 deg should be 90 deg
        _, _, yaw = q_half.to_euler()
        assert abs(yaw) == pytest.approx(math.pi / 2, abs=1e-6)

    def test_inverse(self) -> None:
        """Test quaternion inverse."""
        from _pi import Quaternion

        q = Quaternion.from_euler(0.1, 0.2, 0.3)
        q_inv = q.inverse()

        # q * q_inv should be identity
        result = q * q_inv
        assert result.w == pytest.approx(1.0, abs=1e-6)

    def test_multiply(self) -> None:
        """Test quaternion multiplication."""
        from _pi import Quaternion

        q1 = Quaternion.from_euler(0.1, 0.0, 0.0)
        q2 = Quaternion.from_euler(0.0, 0.1, 0.0)

        result = q1 * q2

        # Result should be a valid quaternion
        norm_sq = result.w**2 + result.x**2 + result.y**2 + result.z**2
        assert norm_sq == pytest.approx(1.0, abs=1e-6)

    def test_properties(self) -> None:
        """Test roll, pitch, yaw properties."""
        from _pi import Quaternion

        q = Quaternion.from_euler(0.1, 0.2, 0.3)

        assert q.roll == pytest.approx(0.1, abs=1e-6)
        assert q.pitch == pytest.approx(0.2, abs=1e-6)
        assert q.yaw == pytest.approx(0.3, abs=1e-6)


class TestTransform:
    """Tests for the Transform class from Rust bindings."""

    def test_identity(self) -> None:
        """Test identity transform."""
        from _pi import Transform

        t = Transform.identity()

        assert t.position == (0.0, 0.0, 0.0)
        assert t.x == 0.0
        assert t.y == 0.0
        assert t.z == 0.0

    def test_from_position(self) -> None:
        """Test creating transform from position."""
        from _pi import Transform

        t = Transform.from_position(1.0, 2.0, 3.0)

        assert t.x == pytest.approx(1.0)
        assert t.y == pytest.approx(2.0)
        assert t.z == pytest.approx(3.0)

    def test_from_euler(self) -> None:
        """Test creating transform from Euler angles."""
        from _pi import Transform

        t = Transform.from_euler([1.0, 2.0, 3.0], 0.1, 0.2, 0.3)

        assert t.x == pytest.approx(1.0)
        assert t.roll == pytest.approx(0.1, abs=1e-6)
        assert t.pitch == pytest.approx(0.2, abs=1e-6)
        assert t.yaw == pytest.approx(0.3, abs=1e-6)

    def test_transform_point(self) -> None:
        """Test point transformation."""
        from _pi import Transform

        t = Transform.from_position(1.0, 0.0, 0.0)
        point = (0.0, 0.0, 0.0)
        result = t.transform_point(point)

        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_inverse(self) -> None:
        """Test transform inverse."""
        from _pi import Transform

        t = Transform.from_position(1.0, 2.0, 3.0)
        t_inv = t.inverse()
        result = t.compose(t_inv)

        assert result.x == pytest.approx(0.0, abs=1e-6)
        assert result.y == pytest.approx(0.0, abs=1e-6)
        assert result.z == pytest.approx(0.0, abs=1e-6)

    def test_compose(self) -> None:
        """Test transform composition."""
        from _pi import Transform

        t1 = Transform.from_position(1.0, 0.0, 0.0)
        t2 = Transform.from_position(0.0, 2.0, 0.0)
        result = t1.compose(t2)

        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)

    def test_multiply_operator(self) -> None:
        """Test * operator for composition."""
        from _pi import Transform

        t1 = Transform.from_position(1.0, 0.0, 0.0)
        t2 = Transform.from_position(0.0, 1.0, 0.0)
        result = t1 * t2

        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(1.0)

    def test_lerp(self) -> None:
        """Test linear interpolation."""
        from _pi import Transform

        t1 = Transform.from_position(0.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 4.0, 6.0)
        result = t1.lerp(t2, 0.5)

        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)


class TestLowPassFilter:
    """Tests for the LowPassFilter class from Rust bindings."""

    def test_create_with_alpha(self) -> None:
        """Test creating filter with alpha."""
        from _pi import LowPassFilter

        lpf = LowPassFilter(0.5)

        assert lpf.alpha == pytest.approx(0.5)

    def test_create_from_cutoff(self) -> None:
        """Test creating filter from cutoff frequency."""
        from _pi import LowPassFilter

        lpf = LowPassFilter.from_cutoff(10.0, 100.0)

        assert 0.0 < lpf.alpha < 1.0

    def test_update(self) -> None:
        """Test filter update."""
        from _pi import LowPassFilter

        lpf = LowPassFilter(0.5)

        # First value should pass through
        result1 = lpf.update(10.0)
        result2 = lpf.update(10.0)

        assert result2 == pytest.approx(10.0, abs=0.1)

    def test_reset(self) -> None:
        """Test filter reset."""
        from _pi import LowPassFilter

        lpf = LowPassFilter(0.5)
        lpf.update(100.0)
        lpf.reset()

        # After reset, should start fresh
        assert lpf.value == pytest.approx(0.0)

    def test_smoothing(self) -> None:
        """Test that filter actually smooths."""
        from _pi import LowPassFilter

        lpf = LowPassFilter(0.1)  # Heavy smoothing

        # Apply step input
        lpf.update(0.0)
        result = lpf.update(100.0)

        # Should be less than step value due to smoothing
        assert result < 100.0

    def test_invalid_alpha_raises(self) -> None:
        """Test that invalid alpha raises error."""
        from _pi import LowPassFilter

        with pytest.raises(ValueError):
            LowPassFilter(1.5)

        with pytest.raises(ValueError):
            LowPassFilter(-0.1)


class TestPid:
    """Tests for the PID controller from Rust bindings."""

    def test_p_controller(self) -> None:
        """Test P-only controller."""
        from _pi import Pid

        pid = Pid.p(2.0)
        output = pid.update(10.0, 5.0, 0.01)

        # Error = 5, P term = 2 * 5 = 10
        assert output == pytest.approx(10.0)

    def test_pi_controller(self) -> None:
        """Test PI controller."""
        from _pi import Pid

        pid = Pid.pi(1.0, 0.5)

        output1 = pid.update(10.0, 5.0, 0.1)
        output2 = pid.update(10.0, 5.0, 0.1)

        # Integral should accumulate
        assert output2 > output1

    def test_pd_controller(self) -> None:
        """Test PD controller."""
        from _pi import Pid

        pid = Pid.pd(1.0, 0.1)
        _ = pid.update(10.0, 5.0, 0.01)

    def test_pid_controller(self) -> None:
        """Test full PID controller."""
        from _pi import Pid

        pid = Pid.pid(1.0, 0.1, 0.05)
        output = pid.update(10.0, 5.0, 0.01)

        assert output > 0.0

    def test_reset(self) -> None:
        """Test controller reset."""
        from _pi import Pid

        pid = Pid.pi(1.0, 1.0)

        # Accumulate integral
        for _ in range(10):
            pid.update(10.0, 5.0, 0.1)

        pid.reset()

        # After reset, integral should be zero
        output = pid.update(0.0, 0.0, 0.1)
        assert output == pytest.approx(0.0)

    def test_set_gains(self) -> None:
        """Test updating gains."""
        from _pi import Pid

        pid = Pid.p(1.0)
        pid.set_gains(2.0, 0.0, 0.0)

        output = pid.update(10.0, 5.0, 0.01)
        assert output == pytest.approx(10.0)


class TestPidConfig:
    """Tests for PidConfig from Rust bindings."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        from _pi import PidConfig

        config = PidConfig()

        assert config.kp == 1.0
        assert config.ki == 0.0
        assert config.kd == 0.0

    def test_custom_gains(self) -> None:
        """Test custom gains."""
        from _pi import PidConfig

        config = PidConfig(kp=10.0, ki=1.0, kd=0.5)

        assert config.kp == 10.0
        assert config.ki == 1.0
        assert config.kd == 0.5

    def test_with_limits(self) -> None:
        """Test adding limits."""
        from _pi import PidConfig, Pid

        config = PidConfig(kp=100.0).with_limits(-10.0, 10.0)
        pid = Pid(config)

        output = pid.update(100.0, 0.0, 0.01)
        assert output == pytest.approx(10.0)

    def test_with_integral_limit(self) -> None:
        """Test adding integral limit."""
        from _pi import PidConfig

        config = PidConfig(kp=1.0, ki=1.0).with_integral_limit(5.0)
        # Config should be created successfully
        assert config.ki == 1.0


class TestTopic:
    """Tests for Topic from Rust bindings."""

    def test_create_topic(self) -> None:
        """Test creating a topic."""
        from _pi import Topic

        topic = Topic("test_topic")

        assert topic.name == "test_topic"

    def test_subscriber_count(self) -> None:
        """Test subscriber count."""
        from _pi import Topic

        topic = Topic("test")

        assert topic.subscriber_count >= 0

    def test_publish(self) -> None:
        """Test publishing messages."""
        from _pi import Topic

        topic = Topic("test")
        topic.publish('{"data": 123}')  # Should not raise


class TestMotorConstants:
    """Tests for motor index constants."""

    def test_motor_idx_exists(self) -> None:
        """Test that motor_idx submodule exists."""
        from _pi import motor_idx

        assert motor_idx.FR_HIP == 0
        assert motor_idx.FR_THIGH == 1
        assert motor_idx.FR_CALF == 2
        assert motor_idx.FL_HIP == 3
        assert motor_idx.NUM_MOTORS == 12

    def test_motor_names(self) -> None:
        """Test MOTOR_NAMES list."""
        from _pi import MOTOR_NAMES

        assert len(MOTOR_NAMES) == 12
        assert "FR_hip" in MOTOR_NAMES
        assert "FL_thigh" in MOTOR_NAMES

    def test_motor_index_by_name(self) -> None:
        """Test motor_index_by_name function."""
        from _pi import motor_index_by_name

        assert motor_index_by_name("FR_hip") == 0
        assert motor_index_by_name("FL_hip") == 3
        assert motor_index_by_name("nonexistent") is None


class TestSimConfig:
    """Tests for simulation configuration."""

    def test_mock_config(self) -> None:
        """Test mock simulation config."""
        from _pi import SimConfig

        config = SimConfig.mock()

        assert config.backend == "mock"
        assert config.num_envs == 1

    def test_with_num_envs(self) -> None:
        """Test setting number of environments."""
        from _pi import SimConfig

        config = SimConfig.mock().with_num_envs(16)

        assert config.num_envs == 16


class TestPhysicsConfig:
    """Tests for physics configuration."""

    def test_default_config(self) -> None:
        """Test default physics config."""
        from _pi import PhysicsConfig

        config = PhysicsConfig()

        assert config.dt > 0
        assert config.substeps > 0

    def test_fast_config(self) -> None:
        """Test fast preset."""
        from _pi import PhysicsConfig

        config = PhysicsConfig.fast()
        assert config.dt > 0

    def test_accurate_config(self) -> None:
        """Test accurate preset."""
        from _pi import PhysicsConfig

        config = PhysicsConfig.accurate()
        assert config.substeps >= 1


class TestMockSimBackend:
    """Tests for MockSimBackend."""

    def test_create_backend(self) -> None:
        """Test creating mock backend."""
        from _pi import MockSimBackend

        sim = MockSimBackend()

        assert sim.name() == "mock"

    def test_reset(self) -> None:
        """Test resetting simulation."""
        from _pi import MockSimBackend

        sim = MockSimBackend()
        state = sim.reset()

        assert state.sim_time >= 0.0
        assert not state.terminated

    def test_step(self) -> None:
        """Test stepping simulation."""
        from _pi import MockSimBackend

        sim = MockSimBackend()
        sim.reset()

        actions = [0.0] * 12
        state, done = sim.step(actions)

        assert state.sim_time > 0.0
        assert isinstance(done, bool)

    def test_state(self) -> None:
        """Test getting current state."""
        from _pi import MockSimBackend

        sim = MockSimBackend()
        sim.reset()

        state = sim.state()

        assert hasattr(state, "sim_time")
        assert hasattr(state, "terminated")
        assert hasattr(state, "truncated")

    def test_sim_time(self) -> None:
        """Test getting simulation time."""
        from _pi import MockSimBackend

        sim = MockSimBackend()
        sim.reset()

        time1 = sim.sim_time()
        sim.step([0.0] * 12)
        time2 = sim.sim_time()

        assert time2 > time1


class TestControlLoop:
    """Tests for control loop functionality."""

    def test_run_control_loop(self) -> None:
        """Test running a control loop."""
        from _pi import run_control_loop

        iterations = []

        def callback(iteration: int, dt: float) -> bool:
            iterations.append(iteration)
            return iteration < 5

        stats = run_control_loop(1000.0, callback, max_iterations=10)

        assert stats.iterations == 5

    def test_control_loop_stats(self) -> None:
        """Test control loop statistics."""
        from _pi import run_control_loop

        def callback(iteration: int, dt: float) -> bool:
            return iteration < 10

        stats = run_control_loop(1000.0, callback)

        assert stats.iterations == 10
        assert stats.avg_iteration_time_ms >= 0
        assert stats.max_iteration_time_ms >= 0

    def test_control_loop_named(self) -> None:
        """Test named control loop."""
        from _pi import run_control_loop

        def callback(iteration: int, dt: float) -> bool:
            return iteration < 3

        stats = run_control_loop(1000.0, callback, name="test_loop")

        assert stats.iterations == 3


class TestVersion:
    """Tests for version information."""

    def test_version_exists(self) -> None:
        """Test that version info is available."""
        from _pi import __version__, VERSION

        assert isinstance(__version__, str)
        assert isinstance(VERSION, str)
        assert len(__version__) > 0
