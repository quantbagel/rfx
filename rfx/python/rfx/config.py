"""
rfx.config - Configuration loading and multi-embodiment support
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    """Configuration for a camera."""

    name: str = "camera"
    width: int = 640
    height: int = 480
    fps: int = 30
    device_id: int | str = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class JointConfig:
    """Configuration for a single joint."""

    name: str
    index: int
    position_min: float = -3.14159
    position_max: float = 3.14159
    velocity_max: float = 10.0
    effort_max: float = 100.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JointConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RobotConfig:
    """
    Configuration for a robot.

    Attributes:
        name: Human-readable name (e.g., "SO-101")
        urdf_path: Path to URDF file (for simulation)
        state_dim: Actual DOF for state
        action_dim: Actual DOF for actions
        max_state_dim: Pad state to this for multi-robot training
        max_action_dim: Pad action to this for multi-robot training
        cameras: List of camera configurations
        joints: List of joint configurations
        control_freq_hz: Control loop frequency
        hardware: Hardware-specific settings
    """

    name: str = "robot"
    urdf_path: str | None = None
    state_dim: int = 12
    action_dim: int = 6
    max_state_dim: int = 64
    max_action_dim: int = 64
    cameras: list[CameraConfig] = field(default_factory=list)
    joints: list[JointConfig] = field(default_factory=list)
    control_freq_hz: int = 50
    hardware: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RobotConfig:
        cameras = [CameraConfig.from_dict(c) for c in d.pop("cameras", [])]
        joints = [JointConfig.from_dict(j) for j in d.pop("joints", [])]
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(cameras=cameras, joints=joints, **filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotConfig:
        import yaml  # type: ignore[import-untyped]

        path = Path(path)
        if not path.exists():
            for search_path in _get_config_search_paths():
                full_path = search_path / path
                if full_path.exists():
                    path = full_path
                    break
            else:
                raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "urdf_path": self.urdf_path,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "cameras": [vars(c) for c in self.cameras],
            "joints": [vars(j) for j in self.joints],
            "control_freq_hz": self.control_freq_hz,
            "hardware": self.hardware,
        }


def _get_config_search_paths() -> list[Path]:
    paths = [Path.cwd()]
    package_dir = Path(__file__).parent.parent.parent
    paths.append(package_dir / "configs")
    if "RFX_CONFIG_DIR" in os.environ:
        paths.append(Path(os.environ["RFX_CONFIG_DIR"]))
    return paths


def load_config(config_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path, dict):
        return config_path
    config = RobotConfig.from_yaml(config_path)
    return config.to_dict()


# Pre-defined configs
SO101_CONFIG = RobotConfig(
    name="SO-101",
    state_dim=12,
    action_dim=6,
    max_state_dim=64,
    max_action_dim=64,
    control_freq_hz=50,
    joints=[
        JointConfig(name="shoulder_pan", index=0),
        JointConfig(name="shoulder_lift", index=1),
        JointConfig(name="elbow", index=2),
        JointConfig(name="wrist_pitch", index=3),
        JointConfig(name="wrist_roll", index=4),
        JointConfig(name="gripper", index=5),
    ],
    hardware={"port": "/dev/ttyACM0", "baudrate": 1000000},
)

GO2_CONFIG = RobotConfig(
    name="Unitree Go2",
    state_dim=36,
    action_dim=12,
    max_state_dim=64,
    max_action_dim=64,
    control_freq_hz=200,
    hardware={
        "ip_address": "192.168.123.161",
        # dds_backend: "zenoh" | "dust" | "cyclone" | None (auto)
        # zenoh_endpoint: "tcp/192.168.123.161:7447" (optional)
    },
)
