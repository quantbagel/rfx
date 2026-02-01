"""
rfx.real.base - Base class for real hardware robots
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Union
from pathlib import Path

from ..robot import RobotBase
from ..config import RobotConfig

if TYPE_CHECKING:
    import torch


class RealRobot(RobotBase):
    """Real hardware robot with unified interface."""

    def __init__(
        self,
        config: Union[str, Path, RobotConfig, Dict],
        robot_type: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(config, (str, Path)):
            self._config = RobotConfig.from_yaml(config)
        elif isinstance(config, dict):
            self._config = RobotConfig.from_dict(config)
        else:
            self._config = config

        hardware_config = {**self._config.hardware, **kwargs}

        super().__init__(
            state_dim=self._config.state_dim,
            action_dim=self._config.action_dim,
            num_envs=1,
            max_state_dim=self._config.max_state_dim,
            max_action_dim=self._config.max_action_dim,
            device="cpu",
        )

        if robot_type is None:
            robot_type = self._detect_robot_type()

        self._robot_type = robot_type
        self._backend = self._create_backend(robot_type, hardware_config)

    def _detect_robot_type(self) -> str:
        name = self._config.name.lower()
        if "so101" in name or "so-101" in name:
            return "so101"
        elif "go2" in name:
            return "go2"
        else:
            raise ValueError(f"Cannot detect robot type from: {self._config.name}")

    def _create_backend(self, robot_type: str, hardware_config: Dict):
        if robot_type == "so101":
            from .so101 import So101Backend
            return So101Backend(config=self._config, **hardware_config)
        elif robot_type == "go2":
            from .go2 import Go2Backend
            return Go2Backend(config=self._config, **hardware_config)
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")

    @property
    def robot_type(self) -> str:
        return self._robot_type

    @property
    def config(self) -> RobotConfig:
        return self._config

    def observe(self) -> Dict[str, "torch.Tensor"]:
        return self._backend.observe()

    def act(self, action: "torch.Tensor") -> None:
        self._backend.act(action)

    def reset(self, env_ids: Optional["torch.Tensor"] = None) -> Dict[str, "torch.Tensor"]:
        return self._backend.reset()

    def go_home(self) -> None:
        self._backend.go_home()

    def disconnect(self) -> None:
        self._backend.disconnect()

    @classmethod
    def from_config(cls, config_path: Union[str, Path], robot_type: Optional[str] = None, **kwargs) -> "RealRobot":
        return cls(config_path, robot_type=robot_type, **kwargs)

    def __repr__(self) -> str:
        return f"RealRobot(type='{self._robot_type}', state_dim={self._state_dim}, connected={self._backend.is_connected()})"

    def __del__(self):
        if hasattr(self, "_backend"):
            self._backend.disconnect()
