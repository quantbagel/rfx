"""
rfx.real.so101 - SO-101 arm hardware backend
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from ..config import RobotConfig, SO101_CONFIG
from ..observation import make_observation

if TYPE_CHECKING:
    pass


class So101Backend:
    """SO-101 hardware backend using Rust driver."""

    def __init__(
        self,
        config: RobotConfig,
        port: str = "/dev/ttyACM0",
        baudrate: int = 1000000,
        is_leader: bool = False,
        **kwargs,
    ):
        self.config = config
        self.port = port
        self.is_leader = is_leader

        try:
            from rfx._rfx import So101, So101Config

            self._So101 = So101
            self._So101Config = So101Config
        except ImportError:
            raise ImportError("rfx Rust bindings not available. Build with: maturin develop")

        if is_leader:
            rust_config = So101Config.leader(port)
        else:
            rust_config = So101Config.follower(port)

        if baudrate != 1000000:
            rust_config = rust_config.with_baudrate(baudrate)

        self._arm = So101.connect(rust_config)

        if is_leader:
            self._arm.set_torque_enable(False)

        self._camera = None
        camera_id = kwargs.get("camera_id")
        if camera_id is not None:
            from .camera import Camera

            self._camera = Camera(device_id=camera_id)

    def is_connected(self) -> bool:
        return self._arm.is_connected()

    def observe(self) -> Dict[str, torch.Tensor]:
        state = self._arm.state()
        positions = torch.tensor(state.joint_positions(), dtype=torch.float32)
        velocities = torch.tensor(state.joint_velocities(), dtype=torch.float32)
        raw_state = torch.cat([positions, velocities]).unsqueeze(0)

        obs = make_observation(
            state=raw_state,
            state_dim=self.config.state_dim,
            max_state_dim=self.config.max_state_dim,
            device="cpu",
        )

        if self._camera is not None:
            image = self._camera.capture()
            obs["images"] = image.unsqueeze(0).unsqueeze(0)

        return obs

    def act(self, action: torch.Tensor) -> None:
        if self.is_leader:
            return
        action_6dof = action[0, : self.config.action_dim].cpu().numpy()
        self._arm.set_positions(list(action_6dof))

    def reset(self) -> Dict[str, torch.Tensor]:
        if not self.is_leader:
            self._arm.go_home()
        return self.observe()

    def go_home(self) -> None:
        if not self.is_leader:
            self._arm.go_home()

    def disconnect(self) -> None:
        if self._camera is not None:
            self._camera.release()
        self._arm.disconnect()

    def read_positions(self) -> torch.Tensor:
        return torch.tensor(self._arm.read_positions(), dtype=torch.float32)


class So101Robot:
    """Convenience class for SO-101 robot."""

    def __new__(cls, port: str = "/dev/ttyACM0", **kwargs):
        from .base import RealRobot

        return RealRobot(config=SO101_CONFIG, robot_type="so101", port=port, **kwargs)


class So101LeaderFollower:
    """Leader-follower teleoperation setup."""

    def __init__(
        self, leader_port: str = "/dev/ttyACM0", follower_port: str = "/dev/ttyACM1", **kwargs
    ):
        self._leader = So101Backend(config=SO101_CONFIG, port=leader_port, is_leader=True)
        self._follower = So101Backend(config=SO101_CONFIG, port=follower_port, is_leader=False)

    @property
    def leader(self) -> So101Backend:
        return self._leader

    @property
    def follower(self) -> So101Backend:
        return self._follower

    def step(self) -> torch.Tensor:
        positions = self._leader.read_positions()
        padded = torch.zeros(1, self._follower.config.max_action_dim)
        padded[0, :6] = positions
        self._follower.act(padded)
        return positions

    def run(self, callback=None):
        import time

        print("Starting teleoperation. Press Ctrl+C to stop.")
        try:
            while True:
                positions = self.step()
                if callback:
                    callback(positions)
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            self.disconnect()

    def disconnect(self):
        self._leader.disconnect()
        self._follower.disconnect()
