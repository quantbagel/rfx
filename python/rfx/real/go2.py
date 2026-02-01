"""
rfx.real.go2 - Unitree Go2 hardware backend
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from ..config import RobotConfig, GO2_CONFIG
from ..observation import make_observation

if TYPE_CHECKING:
    pass


class Go2Backend:
    """Unitree Go2 hardware backend using Rust DDS driver."""

    def __init__(
        self,
        config: RobotConfig,
        ip_address: str = "192.168.123.161",
        edu_mode: bool = False,
        **kwargs,
    ):
        self.config = config
        self.ip_address = ip_address
        self.edu_mode = edu_mode

        try:
            from rfx._rfx import Go2, Go2Config
            self._Go2 = Go2
            self._Go2Config = Go2Config
        except ImportError:
            raise ImportError("rfx Rust bindings not available. Build with: maturin develop")

        rust_config = Go2Config(ip_address)
        if edu_mode:
            rust_config = rust_config.with_edu_mode()

        self._robot = Go2.connect(rust_config)

    def is_connected(self) -> bool:
        return self._robot.is_connected()

    def observe(self) -> Dict[str, torch.Tensor]:
        state = self._robot.state()
        positions = torch.tensor(state.joint_positions(), dtype=torch.float32)
        velocities = torch.tensor(state.joint_velocities(), dtype=torch.float32)
        imu = state.imu
        orientation = torch.tensor(imu.quaternion, dtype=torch.float32)
        angular_vel = torch.tensor(imu.gyroscope, dtype=torch.float32)
        linear_acc = torch.tensor(imu.accelerometer, dtype=torch.float32)

        raw_state = torch.cat([positions, velocities, orientation, angular_vel, linear_acc]).unsqueeze(0)

        return make_observation(
            state=raw_state,
            state_dim=self.config.state_dim,
            max_state_dim=self.config.max_state_dim,
            device="cpu",
        )

    def act(self, action: torch.Tensor) -> None:
        if not self.edu_mode:
            vx = action[0, 0].item()
            vy = action[0, 1].item()
            vyaw = action[0, 2].item()
            self._robot.walk(vx, vy, vyaw)
        else:
            action_12dof = action[0, :12].cpu().numpy()
            kp = action[0, 12].item() if action.shape[1] > 12 else 20.0
            kd = action[0, 13].item() if action.shape[1] > 13 else 0.5
            self._robot.set_motor_positions(list(action_12dof), kp, kd)

    def reset(self) -> Dict[str, torch.Tensor]:
        self._robot.stand()
        return self.observe()

    def go_home(self) -> None:
        self._robot.stand()

    def disconnect(self) -> None:
        self._robot.disconnect()

    def stand(self) -> None:
        self._robot.stand()

    def sit(self) -> None:
        self._robot.sit()

    def walk(self, vx: float, vy: float, vyaw: float) -> None:
        self._robot.walk(vx, vy, vyaw)


class Go2Robot:
    """Convenience class for Go2 robot."""

    def __new__(cls, ip_address: str = "192.168.123.161", **kwargs):
        from .base import RealRobot
        return RealRobot(config=GO2_CONFIG, robot_type="go2", ip_address=ip_address, **kwargs)
