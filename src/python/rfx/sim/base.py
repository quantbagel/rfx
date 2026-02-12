"""
rfx.sim.base - Base class for simulation robots
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Union
from pathlib import Path

from ..robot import RobotBase
from ..config import RobotConfig

if TYPE_CHECKING:
    import torch


class SimRobot(RobotBase):
    """
    Simulated robot with parallel environments.

    Example:
        >>> robot = SimRobot.from_config("so101.yaml", num_envs=4096, backend="genesis")
        >>> obs = robot.observe()       # (4096, 64)
        >>> robot.act(action)           # Step all 4096 envs
    """

    def __init__(
        self,
        config: Union[str, Path, RobotConfig, Dict],
        num_envs: int = 1,
        backend: str = "mock",
        device: str = "cuda",
        **kwargs,
    ):
        if isinstance(config, (str, Path)):
            self._config = RobotConfig.from_yaml(config)
        elif isinstance(config, dict):
            self._config = RobotConfig.from_dict(config)
        else:
            self._config = config

        super().__init__(
            state_dim=self._config.state_dim,
            action_dim=self._config.action_dim,
            num_envs=num_envs,
            max_state_dim=self._config.max_state_dim,
            max_action_dim=self._config.max_action_dim,
            device=device,
        )

        self._backend_name = backend
        self._backend = self._create_backend(backend, **kwargs)

    def _create_backend(self, backend: str, **kwargs):
        if backend == "genesis":
            from .genesis import GenesisBackend

            return GenesisBackend(self._config, self._num_envs, self._device, **kwargs)
        elif backend == "mjx":
            from .mjx import MjxBackend

            return MjxBackend(self._config, self._num_envs, self._device, **kwargs)
        elif backend == "mock":
            from .mock import MockBackend

            return MockBackend(self._config, self._num_envs, self._device, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @property
    def backend(self) -> str:
        return self._backend_name

    @property
    def config(self) -> RobotConfig:
        return self._config

    def observe(self) -> Dict[str, "torch.Tensor"]:
        return self._backend.observe()

    def act(self, action: "torch.Tensor") -> None:
        self._backend.act(action)

    def reset(self, env_ids: Optional["torch.Tensor"] = None) -> Dict[str, "torch.Tensor"]:
        return self._backend.reset(env_ids)

    def get_reward(self) -> "torch.Tensor":
        return self._backend.get_reward()

    def get_done(self) -> "torch.Tensor":
        return self._backend.get_done()

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        num_envs: int = 1,
        backend: str = "mock",
        device: str = None,
        **kwargs,
    ) -> "SimRobot":
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(config_path, num_envs=num_envs, backend=backend, device=device, **kwargs)

    def __repr__(self) -> str:
        return (
            f"SimRobot(backend='{self._backend_name}', "
            f"num_envs={self._num_envs}, "
            f"state_dim={self._state_dim}, "
            f"device='{self._device}')"
        )
