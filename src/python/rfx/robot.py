"""
rfx.Robot - The core interface for all robots

Three methods. That's it.

Example:
    >>> robot = rfx.Robot.from_config("so101.yaml")
    >>> obs = robot.observe()           # Dict[str, Tensor]
    >>> robot.act(action)               # Execute action
    >>> obs = robot.reset()             # Reset and get initial obs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class Robot(Protocol):
    """
    The entire robot interface. Real or simulated, same API.

    Properties:
        num_envs: Number of parallel environments (1 for real robots)
        state_dim: Actual state dimensionality (before padding)
        action_dim: Actual action dimensionality (before padding)
        max_state_dim: Padded state dim for multi-embodiment training
        max_action_dim: Padded action dim for multi-embodiment training
        device: Device for tensors ("cuda" or "cpu")

    Methods:
        observe(): Get current state as Dict[str, Tensor]
        act(action): Execute action tensor
        reset(env_ids): Reset envs and return initial observation
    """

    @property
    def num_envs(self) -> int:
        """Number of parallel environments (1 for real robots)."""
        ...

    @property
    def state_dim(self) -> int:
        """Actual state dimensionality (e.g., 6 for SO-101)."""
        ...

    @property
    def action_dim(self) -> int:
        """Actual action dimensionality (e.g., 6 for SO-101)."""
        ...

    @property
    def max_state_dim(self) -> int:
        """Padded state dim for multi-embodiment training."""
        ...

    @property
    def max_action_dim(self) -> int:
        """Padded action dim for multi-embodiment training."""
        ...

    @property
    def device(self) -> str:
        """Device for tensors ('cuda' or 'cpu')."""
        ...

    def observe(self) -> Dict[str, "torch.Tensor"]:
        """
        Get current observation as a dictionary of tensors.

        Returns:
            Dict containing:
                "state": (num_envs, max_state_dim) - joint positions/velocities
                "images": (num_envs, num_cams, H, W, 3) - RGB images (optional)
                "language": (num_envs, seq_len) - tokenized instruction (optional)
        """
        ...

    def act(self, action: "torch.Tensor") -> None:
        """
        Execute an action on the robot.

        Args:
            action: (num_envs, max_action_dim) or (num_envs, horizon, max_action_dim)
        """
        ...

    def reset(self, env_ids: Optional["torch.Tensor"] = None) -> Dict[str, "torch.Tensor"]:
        """
        Reset environments and return initial observation.

        Args:
            env_ids: (N,) tensor of environment indices to reset.
                    If None, resets all environments.

        Returns:
            Initial observation dictionary (same format as observe()).
        """
        ...


class RobotBase(ABC):
    """
    Abstract base class for Robot implementations.

    Provides common functionality and enforces the interface.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_envs: int = 1,
        max_state_dim: int = 64,
        max_action_dim: int = 64,
        device: str = "cpu",
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._num_envs = num_envs
        self._max_state_dim = max(max_state_dim, state_dim)
        self._max_action_dim = max(max_action_dim, action_dim)
        self._device = device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def max_state_dim(self) -> int:
        return self._max_state_dim

    @property
    def max_action_dim(self) -> int:
        return self._max_action_dim

    @property
    def device(self) -> str:
        return self._device

    @abstractmethod
    def observe(self) -> Dict[str, "torch.Tensor"]:
        """Get current observation."""
        ...

    @abstractmethod
    def act(self, action: "torch.Tensor") -> None:
        """Execute action."""
        ...

    @abstractmethod
    def reset(self, env_ids: Optional["torch.Tensor"] = None) -> Dict[str, "torch.Tensor"]:
        """Reset environments."""
        ...

    @classmethod
    def from_config(cls, config_path: str, **kwargs) -> "RobotBase":
        """Create a robot from a YAML config file."""
        from .config import load_config

        config = load_config(config_path)
        config.update(kwargs)
        return cls(**config)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_envs={self.num_envs}, "
            f"state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"device='{self.device}')"
        )
