"""Tests for rfx.decorators module."""

import pytest
from typing import Any

from rfx.decorators import control_loop, policy, MotorCommands


class TestControlLoopDecorator:
    """Tests for the @control_loop decorator."""

    def test_control_loop_default(self) -> None:
        """Test decorator with default settings."""
        @control_loop()
        def my_policy(state: Any) -> Any:
            """Test policy."""
            return state

        assert hasattr(my_policy, "_pi_control_loop")
        assert my_policy._pi_control_loop is True
        assert my_policy._pi_rate_hz == 500.0
        assert my_policy._pi_name == "my_policy"

    def test_control_loop_custom_rate(self) -> None:
        """Test decorator with custom rate."""
        @control_loop(rate_hz=1000.0)
        def fast_policy(state: Any) -> Any:
            """Fast policy."""
            return state

        assert fast_policy._pi_rate_hz == 1000.0

    def test_control_loop_custom_name(self) -> None:
        """Test decorator with custom name."""
        @control_loop(name="balance")
        def my_policy(state: Any) -> Any:
            """Balance policy."""
            return state

        assert my_policy._pi_name == "balance"

    def test_control_loop_preserves_function(self) -> None:
        """Test that decorated function still works."""
        @control_loop(rate_hz=100.0)
        def double_it(x: int) -> int:
            """Double the input."""
            return x * 2

        result = double_it(5)
        assert result == 10

    def test_control_loop_preserves_docstring(self) -> None:
        """Test that docstring is preserved."""
        @control_loop()
        def documented_policy(state: Any) -> Any:
            """This is the docstring."""
            return state

        assert documented_policy.__doc__ == "This is the docstring."

    def test_control_loop_preserves_name(self) -> None:
        """Test that function name is preserved."""
        @control_loop()
        def original_name(state: Any) -> Any:
            """Test."""
            return state

        assert original_name.__name__ == "original_name"


class TestPolicyDecorator:
    """Tests for the @policy decorator."""

    def test_policy_default(self) -> None:
        """Test decorator with default settings."""
        @policy()
        def my_policy(state: Any) -> Any:
            """Test policy."""
            return state

        assert hasattr(my_policy, "_pi_policy")
        assert my_policy._pi_policy is True
        assert my_policy._pi_model is None
        assert my_policy._pi_jit is False

    def test_policy_with_model_raises(self) -> None:
        """Test that specifying model raises NotImplementedError."""
        @policy(model="walking.onnx")
        def neural_policy(state: Any) -> Any:
            """Neural policy."""
            return state

        with pytest.raises(NotImplementedError, match="Neural network policies"):
            neural_policy({})

    def test_policy_with_jit(self) -> None:
        """Test decorator with JIT flag."""
        @policy(jit=True)
        def jit_policy(state: Any) -> Any:
            """JIT policy."""
            return state

        assert jit_policy._pi_jit is True

    def test_policy_preserves_function(self) -> None:
        """Test that decorated function still works (when no model)."""
        @policy()
        def passthrough(x: int) -> int:
            """Pass through."""
            return x

        result = passthrough(42)
        assert result == 42

    def test_policy_preserves_docstring(self) -> None:
        """Test that docstring is preserved."""
        @policy()
        def documented_policy(state: Any) -> Any:
            """Policy documentation."""
            return state

        assert documented_policy.__doc__ == "Policy documentation."


class TestMotorCommands:
    """Tests for the MotorCommands class."""

    def test_motor_commands_init_empty(self) -> None:
        """Test creating empty motor commands."""
        cmd = MotorCommands()

        assert cmd.positions == {}
        assert cmd.velocities == {}
        assert cmd.torques == {}
        assert cmd.kp == 20.0
        assert cmd.kd == 0.5

    def test_motor_commands_init_with_positions(self) -> None:
        """Test creating commands with positions."""
        cmd = MotorCommands(positions={"FR_hip": 0.5, "FL_hip": -0.5})

        assert cmd.positions == {"FR_hip": 0.5, "FL_hip": -0.5}

    def test_motor_commands_init_with_gains(self) -> None:
        """Test creating commands with custom gains."""
        cmd = MotorCommands(kp=30.0, kd=1.0)

        assert cmd.kp == 30.0
        assert cmd.kd == 1.0

    def test_motor_commands_from_positions(self) -> None:
        """Test creating from positions factory."""
        cmd = MotorCommands.from_positions(
            {"FR_hip": 0.1, "FR_thigh": 0.2},
            kp=25.0,
            kd=0.8,
        )

        assert cmd.positions == {"FR_hip": 0.1, "FR_thigh": 0.2}
        assert cmd.kp == 25.0
        assert cmd.kd == 0.8

    def test_motor_commands_from_velocities(self) -> None:
        """Test creating from velocities factory."""
        cmd = MotorCommands.from_velocities(
            {"FR_hip": 1.0, "FL_hip": -1.0},
            kd=2.0,
        )

        assert cmd.velocities == {"FR_hip": 1.0, "FL_hip": -1.0}
        assert cmd.kd == 2.0

    def test_motor_commands_from_torques(self) -> None:
        """Test creating from torques factory."""
        cmd = MotorCommands.from_torques({"FR_hip": 5.0, "FL_hip": 5.0})

        assert cmd.torques == {"FR_hip": 5.0, "FL_hip": 5.0}

    def test_motor_commands_repr_empty(self) -> None:
        """Test repr for empty commands."""
        cmd = MotorCommands()
        repr_str = repr(cmd)

        assert "MotorCommands()" in repr_str

    def test_motor_commands_repr_with_positions(self) -> None:
        """Test repr with positions."""
        cmd = MotorCommands(positions={"FR_hip": 0.5})
        repr_str = repr(cmd)

        assert "MotorCommands(" in repr_str
        assert "positions=" in repr_str
        assert "FR_hip" in repr_str


class TestMotorCommandsToArray:
    """Tests for MotorCommands.to_array() method."""

    def test_to_array_empty(self) -> None:
        """Test converting empty commands to array."""
        cmd = MotorCommands()
        arr = cmd.to_array()

        assert arr == [0.0] * 12

    def test_to_array_custom_size(self) -> None:
        """Test converting with custom motor count."""
        cmd = MotorCommands()
        arr = cmd.to_array(num_motors=6)

        assert len(arr) == 6
        assert arr == [0.0] * 6

    def test_to_array_with_positions(self) -> None:
        """Test converting positions to array."""
        # Note: This test depends on motor_index_by_name from the Rust bindings
        # We're testing the Python logic here, actual index lookup requires bindings
        cmd = MotorCommands(positions={"FR_hip": 0.5})
        arr = cmd.to_array()

        # FR_hip should be at index 0
        assert len(arr) == 12
        # The actual value depends on motor_index_by_name working


class TestMotorCommandsCombined:
    """Tests for using multiple command types together."""

    def test_combined_commands(self) -> None:
        """Test combining positions, velocities, and torques."""
        cmd = MotorCommands(
            positions={"FR_hip": 0.5},
            velocities={"FL_hip": 1.0},
            torques={"RR_hip": 2.0},
        )

        assert cmd.positions == {"FR_hip": 0.5}
        assert cmd.velocities == {"FL_hip": 1.0}
        assert cmd.torques == {"RR_hip": 2.0}

    def test_repr_combined(self) -> None:
        """Test repr with multiple types."""
        cmd = MotorCommands(
            positions={"FR_hip": 0.5},
            velocities={"FL_hip": 1.0},
        )
        repr_str = repr(cmd)

        assert "positions=" in repr_str
        assert "velocities=" in repr_str
