"""
rfx.teleop.config - Teleoperation session configuration models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Literal, cast


TransportBackend = Literal["inproc", "zenoh", "dds"]
JitBackend = Literal["auto", "cpu", "cuda", "metal"]


@dataclass(frozen=True)
class ArmPairConfig:
    """Leader/follower pair definition for SO-101 teleoperation."""

    name: str
    leader_port: str
    follower_port: str


@dataclass(frozen=True)
class CameraStreamConfig:
    """Camera capture settings for asynchronous recording."""

    name: str
    device_id: int | str
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class TransportConfig:
    """Transport behavior for teleop runtime paths."""

    backend: TransportBackend = "inproc"
    zero_copy_hot_path: bool = True
    queue_capacity: int = 1024


@dataclass(frozen=True)
class JitPolicyConfig:
    """Optional runtime compilation settings for policy execution."""

    enabled: bool = False
    backend: JitBackend = "auto"
    strict: bool = False


def _default_arm_pairs() -> tuple[ArmPairConfig, ...]:
    return (
        ArmPairConfig(name="left", leader_port="/dev/ttyACM0", follower_port="/dev/ttyACM1"),
        ArmPairConfig(name="right", leader_port="/dev/ttyACM2", follower_port="/dev/ttyACM3"),
    )


def _default_cameras() -> tuple[CameraStreamConfig, ...]:
    return (
        CameraStreamConfig(name="cam0", device_id=0, width=640, height=480, fps=30),
        CameraStreamConfig(name="cam1", device_id=1, width=640, height=480, fps=30),
        CameraStreamConfig(name="cam2", device_id=2, width=640, height=480, fps=30),
    )


@dataclass(frozen=True)
class TeleopSessionConfig:
    """Top-level teleoperation session configuration."""

    rate_hz: float = 350.0
    output_dir: Path = Path("demos")
    arm_pairs: tuple[ArmPairConfig, ...] = field(default_factory=_default_arm_pairs)
    cameras: tuple[CameraStreamConfig, ...] = field(default_factory=_default_cameras)
    transport: TransportConfig = field(default_factory=TransportConfig)
    jit: JitPolicyConfig = field(default_factory=JitPolicyConfig)
    max_timing_samples: int = 250_000

    def __post_init__(self) -> None:
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        if self.max_timing_samples <= 0:
            raise ValueError("max_timing_samples must be > 0")
        if not self.arm_pairs:
            raise ValueError("At least one arm pair is required")

        object.__setattr__(self, "output_dir", Path(self.output_dir))

        pair_names = [pair.name for pair in self.arm_pairs]
        if len(pair_names) != len(set(pair_names)):
            raise ValueError("arm_pairs names must be unique")

        camera_names = [camera.name for camera in self.cameras]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("camera names must be unique")

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz

    @classmethod
    def bimanual(
        cls,
        *,
        left_leader_port: str = "/dev/ttyACM0",
        left_follower_port: str = "/dev/ttyACM1",
        right_leader_port: str = "/dev/ttyACM2",
        right_follower_port: str = "/dev/ttyACM3",
        **kwargs: Any,
    ) -> "TeleopSessionConfig":
        arm_pairs = (
            ArmPairConfig(
                name="left", leader_port=left_leader_port, follower_port=left_follower_port
            ),
            ArmPairConfig(
                name="right",
                leader_port=right_leader_port,
                follower_port=right_follower_port,
            ),
        )
        return cls(arm_pairs=arm_pairs, **kwargs)

    @classmethod
    def single_arm_pair(
        cls,
        *,
        name: str = "main",
        leader_port: str = "/dev/ttyACM0",
        follower_port: str = "/dev/ttyACM1",
        **kwargs: Any,
    ) -> "TeleopSessionConfig":
        arm_pairs = (
            ArmPairConfig(name=name, leader_port=leader_port, follower_port=follower_port),
        )
        return cls(arm_pairs=arm_pairs, **kwargs)


def to_serializable(value: Any) -> Any:
    """Convert dataclasses and Paths into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_serializable(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    return value
