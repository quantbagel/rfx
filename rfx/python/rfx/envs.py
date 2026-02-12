"""
Gym-like environment wrappers for robot control

Provides unified interfaces for simulation and real robot control,
following the standard Gym API (reset, step, observation_space, action_space).

Example:
    >>> from rfx.envs import Go2Env
    >>> env = Go2Env(sim=True)
    >>> obs = env.reset()
    >>> for _ in range(1000):
    ...     action = env.action_space.sample()
    ...     obs, reward, done, info = env.step(action)
    ...     if done:
    ...         obs = env.reset()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Box:
    """
    A box in R^n, representing continuous observation/action spaces.

    Attributes:
        low: Lower bounds for each dimension
        high: Upper bounds for each dimension
        shape: Shape of the space
        dtype: Data type (default: np.float32)
    """

    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype = np.float32

    def __post_init__(self):
        self.low = np.asarray(self.low, dtype=self.dtype)
        self.high = np.asarray(self.high, dtype=self.dtype)
        if self.low.shape != self.shape:
            self.low = np.full(self.shape, self.low.flat[0], dtype=self.dtype)
        if self.high.shape != self.shape:
            self.high = np.full(self.shape, self.high.flat[0], dtype=self.dtype)

    def sample(self) -> np.ndarray:
        """Sample a random point from the space."""
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        """Check if x is a valid member of the space."""
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Clip x to be within bounds."""
        return np.clip(x, self.low, self.high)


class BaseEnv(ABC):
    """
    Abstract base class for robot environments.

    All environments should implement this interface for compatibility
    with training utilities.
    """

    @property
    @abstractmethod
    def observation_space(self) -> Box:
        """Get the observation space."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> Box:
        """Get the action space."""
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial observation.

        Returns:
            Initial observation
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass

    def render(self) -> None:
        """Render the environment (optional)."""
        pass


class Go2Env(BaseEnv):
    """
    Gym-like interface for the Unitree Go2 robot.

    Can run in simulation mode (using MockSimBackend) or on real hardware.

    Args:
        sim: If True, use simulation backend; if False, connect to real robot
        sim_config: Configuration for simulation (only used if sim=True)
        robot_ip: IP address of real robot (only used if sim=False)
        control_dt: Control timestep (default: 0.02s = 50Hz)

    Observation Space (48 dimensions):
        - Joint positions (12): Current joint angles
        - Joint velocities (12): Current joint velocities
        - Base angular velocity (3): IMU angular velocity
        - Projected gravity (3): Gravity vector in body frame
        - Commands (3): Velocity commands (vx, vy, yaw_rate)
        - Last actions (12): Previous action
        - Clock (3): Phase variables for gait timing

    Action Space (12 dimensions):
        - Joint position targets, normalized to [-1, 1]
        - Scaled by action_scale to get actual joint deltas

    Example:
        >>> env = Go2Env(sim=True)
        >>> obs = env.reset()
        >>> print(f"Observation shape: {obs.shape}")
        >>> action = np.zeros(12)  # Stand still
        >>> obs, reward, done, info = env.step(action)
    """

    # Go2 joint limits (radians)
    JOINT_LIMITS_LOW = np.array(
        [
            -0.863,
            -0.686,
            -2.818,  # FR: hip, thigh, calf
            -0.863,
            -0.686,
            -2.818,  # FL
            -0.863,
            -0.686,
            -2.818,  # RR
            -0.863,
            -0.686,
            -2.818,  # RL
        ]
    )
    JOINT_LIMITS_HIGH = np.array(
        [
            0.863,
            2.818,
            -0.888,  # FR
            0.863,
            2.818,
            -0.888,  # FL
            0.863,
            2.818,
            -0.888,  # RR
            0.863,
            2.818,
            -0.888,  # RL
        ]
    )

    # Default standing pose
    DEFAULT_STANDING = np.array(
        [
            0.0,
            0.8,
            -1.5,  # FR
            0.0,
            0.8,
            -1.5,  # FL
            0.0,
            0.8,
            -1.5,  # RR
            0.0,
            0.8,
            -1.5,  # RL
        ]
    )

    def __init__(
        self,
        sim: bool = True,
        sim_config: Any = None,
        robot_ip: str | None = None,
        control_dt: float = 0.02,
        action_scale: float = 0.5,
    ):
        self.sim = sim
        self.control_dt = control_dt
        self.action_scale = action_scale

        # Observation and action spaces
        self._observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(48,),
        )
        self._action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
        )

        # State tracking
        self._last_action = np.zeros(12)
        self._commands = np.zeros(3)  # vx, vy, yaw_rate
        self._phase = 0.0
        self._step_count = 0

        # Initialize backend
        if sim:
            self._init_sim_backend(sim_config)
        else:
            self._init_real_backend(robot_ip)

    def _init_sim_backend(self, config: Any) -> None:
        """Initialize simulation backend."""
        try:
            from ._rfx import MockSimBackend, SimConfig

            if config is None:
                config = SimConfig.mock()
            self._backend = MockSimBackend(config)
            self._using_rfx_backend = True
        except ImportError:
            # Fallback to pure Python mock
            self._backend = _PythonMockBackend()
            self._using_rfx_backend = False

    def _init_real_backend(self, robot_ip: str | None) -> None:
        """Initialize real robot backend."""
        try:
            from ._rfx import Go2, Go2Config

            config = Go2Config()
            if robot_ip:
                config = config.with_ip(robot_ip)
            self._backend = Go2(config)
            self._backend.connect()
            self._using_rfx_backend = True
        except ImportError:
            raise RuntimeError(
                "Real robot control requires the rfx Rust extension. Build with: maturin develop"
            )

    @property
    def observation_space(self) -> Box:
        return self._observation_space

    @property
    def action_space(self) -> Box:
        return self._action_space

    def set_commands(self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0) -> None:
        """
        Set velocity commands for the robot.

        Args:
            vx: Forward velocity command
            vy: Lateral velocity command
            yaw_rate: Yaw rate command
        """
        self._commands = np.array([vx, vy, yaw_rate])

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        if self._using_rfx_backend:
            self._backend.reset()

        self._last_action = np.zeros(12)
        self._phase = 0.0
        self._step_count = 0

        return self._get_observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment."""
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Scale action to joint deltas
        scaled_action = action * self.action_scale

        # Compute target joint positions
        target_positions = self.DEFAULT_STANDING + scaled_action

        # Clip to joint limits
        target_positions = np.clip(
            target_positions,
            self.JOINT_LIMITS_LOW,
            self.JOINT_LIMITS_HIGH,
        )

        # Step the backend
        if self._using_rfx_backend:
            state, done = self._backend.step(target_positions.tolist())
        else:
            self._backend.step(target_positions)
            done = False

        # Update state
        self._last_action = action
        self._phase += 2 * np.pi * self.control_dt / 0.5  # 0.5s gait period
        self._step_count += 1

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        info = {"step": self._step_count}

        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        if self._using_rfx_backend:
            state = self._backend.state() if hasattr(self._backend, "state") else None
            if state is not None and hasattr(state, "joint_positions"):
                joint_pos = np.array(state.joint_positions())
                joint_vel = np.array(state.joint_velocities())
            else:
                joint_pos = self.DEFAULT_STANDING.copy()
                joint_vel = np.zeros(12)
        else:
            joint_pos = self._backend.joint_positions.copy()
            joint_vel = self._backend.joint_velocities.copy()

        # Base angular velocity (from IMU)
        base_ang_vel = np.zeros(3)

        # Projected gravity (assuming upright)
        projected_gravity = np.array([0.0, 0.0, -1.0])

        # Clock signals for gait timing
        clock = np.array(
            [
                np.sin(self._phase),
                np.cos(self._phase),
                np.sin(self._phase / 2),
            ]
        )

        # Concatenate observation
        obs = np.concatenate(
            [
                joint_pos,  # 12
                joint_vel,  # 12
                base_ang_vel,  # 3
                projected_gravity,  # 3
                self._commands,  # 3
                self._last_action,  # 12
                clock,  # 3
            ]
        )

        return obs.astype(np.float32)

    def _compute_reward(self) -> float:
        """
        Compute reward for the current state.

        Basic reward structure for locomotion:
        - Forward velocity tracking
        - Penalty for energy usage
        - Penalty for joint velocity
        """
        # Placeholder reward (user should customize for their task)
        # Reward forward velocity, penalize energy
        reward = 0.0

        # Small survival bonus
        reward += 0.1

        # Penalize large actions (energy efficiency)
        action_penalty = -0.01 * np.sum(self._last_action**2)
        reward += action_penalty

        return reward

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self._backend, "disconnect"):
            self._backend.disconnect()


class _PythonMockBackend:
    """Pure Python mock backend for testing without Rust extension."""

    def __init__(self):
        self.joint_positions = Go2Env.DEFAULT_STANDING.copy()
        self.joint_velocities = np.zeros(12)
        self.time = 0.0

    def reset(self) -> None:
        self.joint_positions = Go2Env.DEFAULT_STANDING.copy()
        self.joint_velocities = np.zeros(12)
        self.time = 0.0

    def step(self, target_positions: np.ndarray, dt: float = 0.02) -> None:
        """Simple PD control simulation."""
        kp = 20.0
        kd = 0.5

        # PD control
        pos_error = target_positions - self.joint_positions
        accel = kp * pos_error - kd * self.joint_velocities

        # Integrate
        self.joint_velocities += accel * dt
        self.joint_positions += self.joint_velocities * dt

        # Clamp
        self.joint_positions = np.clip(
            self.joint_positions,
            Go2Env.JOINT_LIMITS_LOW,
            Go2Env.JOINT_LIMITS_HIGH,
        )
        self.joint_velocities = np.clip(self.joint_velocities, -20.0, 20.0)

        self.time += dt


class VecEnv:
    """
    Vectorized environment for parallel rollouts.

    Runs multiple environments in parallel for faster data collection.

    Args:
        env_fn: Function that creates a single environment
        num_envs: Number of parallel environments
    """

    def __init__(self, env_fn: callable, num_envs: int):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs

        # Get spaces from first env
        self._observation_space = self.envs[0].observation_space
        self._action_space = self.envs[0].action_space

    @property
    def observation_space(self) -> Box:
        return self._observation_space

    @property
    def action_space(self) -> Box:
        return self._action_space

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        return np.array([env.reset() for env in self.envs])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Step all environments."""
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]

        # Auto-reset done environments
        for i, done in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()

        return obs, rewards, dones, infos

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()


def make_vec_env(
    env_class: type = Go2Env,
    num_envs: int = 1,
    **kwargs,
) -> VecEnv | BaseEnv:
    """
    Create a (vectorized) environment.

    Args:
        env_class: Environment class to instantiate
        num_envs: Number of parallel environments
        **kwargs: Arguments passed to environment constructor

    Returns:
        VecEnv if num_envs > 1, otherwise single environment
    """
    if num_envs == 1:
        return env_class(**kwargs)

    return VecEnv(lambda: env_class(**kwargs), num_envs)


__all__ = [
    "Box",
    "BaseEnv",
    "Go2Env",
    "VecEnv",
    "make_vec_env",
]
