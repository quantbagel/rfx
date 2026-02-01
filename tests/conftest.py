"""Pytest configuration and shared fixtures."""

import pytest
from typing import Any, Callable


@pytest.fixture
def sample_skill_func() -> Callable[..., str]:
    """A sample function to use as a skill."""
    def walk_forward(distance: float = 1.0) -> str:
        """Walk forward by the specified distance in meters."""
        return f"Walked {distance}m"
    return walk_forward


@pytest.fixture
def sample_skill_with_types() -> Callable[..., dict]:
    """A sample function with full type annotations."""
    def move_robot(
        x: float,
        y: float,
        speed: float = 0.5,
        blocking: bool = True,
    ) -> dict:
        """
        Move the robot to a position.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            speed: Movement speed in m/s
            blocking: Whether to wait for completion

        Returns:
            Movement result dictionary
        """
        return {"x": x, "y": y, "speed": speed, "blocking": blocking}
    return move_robot


@pytest.fixture
def mock_robot_state() -> dict:
    """Mock robot state for testing."""
    return {
        "tick": 12345,
        "timestamp": 1.234,
        "imu": {
            "roll": 0.01,
            "pitch": 0.02,
            "yaw": 0.0,
            "gyroscope": [0.0, 0.0, 0.0],
            "accelerometer": [0.0, 0.0, 9.81],
        },
        "joint_positions": [0.0] * 12,
        "joint_velocities": [0.0] * 12,
        "foot_contact": [True, True, True, True],
    }
