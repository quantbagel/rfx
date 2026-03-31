"""rfx.collection — LeRobot-native data collection for robots."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..robot.config import G1_CONFIG, GO2_CONFIG, SO101_CONFIG, CameraConfig, RobotConfig
from ._dataset import Dataset
from ._hub import from_hub, pull, push
from ._recorder import Recorder

if TYPE_CHECKING:
    from ..robot import Robot


_BUILTIN_CONFIGS: dict[str, RobotConfig] = {
    "so101": SO101_CONFIG,
    "go2": GO2_CONFIG,
    "g1": G1_CONFIG,
}


def _normalize_robot_key(robot_type: str) -> str:
    return robot_type.lower().replace("-", "").replace("_", "")


def _clone_config(config: RobotConfig) -> RobotConfig:
    return RobotConfig.from_dict(config.to_dict())


def _resolve_robot_config(
    robot_type: str,
    *,
    config: str | Path | None = None,
    state_dim: int | None = None,
) -> tuple[str, RobotConfig]:
    if config is not None:
        resolved = RobotConfig.from_yaml(config)
        normalized = _normalize_robot_key(robot_type)
    else:
        normalized = _normalize_robot_key(robot_type)
        if normalized not in _BUILTIN_CONFIGS:
            raise ValueError(
                f"Unknown robot type: '{robot_type}'. "
                f"Available: {', '.join(sorted(_BUILTIN_CONFIGS))}. "
                "Or pass --config path/to/robot.yaml."
            )
        resolved = _clone_config(_BUILTIN_CONFIGS[normalized])

    if state_dim is not None:
        resolved.state_dim = int(state_dim)

    return normalized, resolved


def _resolve_camera_configs(
    robot_config: RobotConfig,
    camera_names: Sequence[str],
    camera_ids: Sequence[int | str],
) -> list[CameraConfig]:
    cameras = [CameraConfig.from_dict(vars(cam)) for cam in robot_config.cameras]

    if camera_ids:
        if camera_names and len(camera_names) != len(camera_ids):
            raise ValueError("camera_names and camera_ids must have the same length when both set")
        if camera_names:
            return [
                CameraConfig(name=str(name), device_id=device_id)
                for name, device_id in zip(camera_names, camera_ids, strict=True)
            ]
        return [
            CameraConfig(name=f"cam{i}", device_id=device_id)
            for i, device_id in enumerate(camera_ids)
        ]

    if camera_names:
        if cameras and len(camera_names) != len(cameras):
            raise ValueError("camera_names must match configured camera count")
        if cameras:
            for cam, name in zip(cameras, camera_names, strict=True):
                cam.name = str(name)
            return cameras
        return [CameraConfig(name=str(name), device_id=i) for i, name in enumerate(camera_names)]

    return cameras


class _CameraRig:
    def __init__(self, camera_configs: Sequence[CameraConfig]) -> None:
        self._cameras: list[tuple[str, Any]] = []
        if not camera_configs:
            return

        from ..real.camera import Camera

        self._cameras = [
            (
                str(camera.name),
                Camera(
                    device_id=camera.device_id,
                    resolution=(camera.width, camera.height),
                    fps=camera.fps,
                ),
            )
            for camera in camera_configs
        ]

    @property
    def camera_names(self) -> list[str]:
        return [name for name, _camera in self._cameras]

    def capture(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for name, camera in self._cameras:
            frame = camera.capture()
            frames[name] = _tensor_to_numpy(frame, dtype=np.uint8)
        return frames

    def close(self) -> None:
        for _name, camera in self._cameras:
            try:
                camera.release()
            except Exception:
                pass


def _tensor_to_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _extract_state(obs: dict[str, Any]) -> np.ndarray:
    if "state" not in obs:
        raise KeyError("robot.observe() must return an observation dict containing 'state'")

    state = _tensor_to_numpy(obs["state"], dtype=np.float32)
    if state.ndim == 0:
        return state.reshape(1)
    if state.ndim >= 2:
        return np.asarray(state[0], dtype=np.float32)
    return np.asarray(state, dtype=np.float32)


def _extract_images(
    obs: dict[str, Any],
    camera_names: Sequence[str],
) -> dict[str, np.ndarray]:
    image_tensor = obs.get("images")
    if image_tensor is None:
        return {}

    images = _tensor_to_numpy(image_tensor, dtype=np.uint8)
    if images.ndim >= 5:
        images = images[0]

    if images.ndim == 3:
        name = camera_names[0] if camera_names else "cam0"
        return {name: np.asarray(images, dtype=np.uint8)}

    if images.ndim != 4:
        raise ValueError(f"Unsupported image tensor shape: {images.shape}")

    resolved_names = list(camera_names) or [f"cam{i}" for i in range(images.shape[0])]
    if len(resolved_names) < images.shape[0]:
        resolved_names.extend(f"cam{i}" for i in range(len(resolved_names), images.shape[0]))
    return {
        resolved_names[i]: np.asarray(images[i], dtype=np.uint8) for i in range(images.shape[0])
    }


def _create_robot(
    robot_key: str,
    robot_config: RobotConfig,
    *,
    mock: bool,
    port: str | None,
    camera_configs: Sequence[CameraConfig],
    robot_factory: Any | None,
) -> Robot:
    if robot_factory is not None:
        return robot_factory(robot_config, mock)

    if mock:
        from ..sim import MockRobot

        return MockRobot(
            state_dim=robot_config.state_dim,
            action_dim=robot_config.action_dim,
            max_state_dim=robot_config.max_state_dim,
            max_action_dim=robot_config.max_action_dim,
        )

    from ..real import RealRobot

    kwargs: dict[str, Any] = {}
    if port is not None:
        if robot_key == "so101":
            kwargs["port"] = port
        else:
            kwargs["ip_address"] = port
    if robot_key == "so101" and len(camera_configs) == 1:
        kwargs["camera_id"] = camera_configs[0].device_id

    return RealRobot(config=robot_config, robot_type=robot_key, **kwargs)


def _mock_action(iteration: int, robot_config: RobotConfig) -> Any:
    import torch

    phase = iteration / max(float(robot_config.control_freq_hz), 1.0)
    values = [
        0.25 * math.sin(phase + joint_idx * 0.2) for joint_idx in range(robot_config.action_dim)
    ]
    tensor = torch.zeros((1, robot_config.max_action_dim), dtype=torch.float32)
    tensor[0, : robot_config.action_dim] = torch.tensor(values, dtype=torch.float32)
    return tensor


def collect(
    robot_type: str,
    repo_id: str,
    *,
    output: str | Path = "datasets",
    episodes: int = 1,
    duration_s: float | None = None,
    task: str = "default",
    fps: int = 30,
    state_dim: int | None = None,
    camera_names: Sequence[str] = (),
    push_to_hub: bool = False,
    mcap: bool = False,
    config: str | Path | None = None,
    port: str | None = None,
    rate_hz: float | None = None,
    mock: bool = False,
    camera_ids: Sequence[int | str] = (),
    robot_factory: Any | None = None,
) -> Dataset:
    """Collect episodes from a robot into a LeRobot dataset.

    The simplest path from hardware to HuggingFace Hub.

    Example:
        dataset = rfx.collection.collect("so101", "my-org/demos", episodes=10)
        dataset.push()
    """
    robot_key, robot_config = _resolve_robot_config(
        robot_type,
        config=config,
        state_dim=state_dim,
    )
    resolved_camera_configs = _resolve_camera_configs(robot_config, camera_names, camera_ids)
    resolved_camera_names = [camera.name for camera in resolved_camera_configs]
    camera_shape = (480, 640, 3)
    if resolved_camera_configs:
        first_camera = resolved_camera_configs[0]
        camera_shape = (first_camera.height, first_camera.width, 3)

    recorder = Recorder.create(
        repo_id,
        root=output,
        fps=fps,
        robot_type=robot_key,
        state_dim=robot_config.state_dim,
        camera_names=resolved_camera_names,
        camera_shape=camera_shape,
        mcap=mcap,
    )

    robot = _create_robot(
        robot_key,
        robot_config,
        mock=mock,
        port=port,
        camera_configs=resolved_camera_configs,
        robot_factory=robot_factory,
    )
    camera_rig = (
        None
        if robot_key == "so101" and len(resolved_camera_configs) == 1 and not mock
        else _CameraRig(resolved_camera_configs)
    )

    sample_hz = float(rate_hz) if rate_hz is not None else float(fps)
    if sample_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    period_s = 1.0 / sample_hz

    try:
        for _ep in range(episodes):
            if hasattr(robot, "reset"):
                robot.reset()
            recorder.start_episode(task=task)
            frame_count = 0
            deadline = time.perf_counter() + float(duration_s) if duration_s is not None else None
            interrupted = False

            while True:
                tick_start = time.perf_counter()
                obs = robot.observe()
                state = _extract_state(obs)
                images = _extract_images(obs, resolved_camera_names)
                if not images and camera_rig is not None:
                    images = camera_rig.capture()

                recorder.add_frame(state=state, images=images or None)
                frame_count += 1

                if mock:
                    robot.act(_mock_action(frame_count, robot_config))

                if deadline is not None and time.perf_counter() >= deadline:
                    break

                sleep_s = period_s - (time.perf_counter() - tick_start)
                try:
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                except KeyboardInterrupt:
                    interrupted = True
                    break

                if deadline is None:
                    # Continue until Ctrl+C ends the episode.
                    continue

            if frame_count == 0:
                raise RuntimeError("record produced zero frames; check robot/camera connectivity")

            recorder.save_episode()

            if interrupted:
                break

        if push_to_hub:
            recorder.push()

        return recorder.dataset
    finally:
        if camera_rig is not None:
            camera_rig.close()
        if hasattr(robot, "disconnect"):
            try:
                robot.disconnect()
            except Exception:
                pass


def open_dataset(repo_id: str, *, root: str | Path = "datasets") -> Dataset:
    """Open an existing local LeRobot dataset."""
    return Dataset.open(repo_id, root=root)


__all__ = [
    "Dataset",
    "Recorder",
    "collect",
    "from_hub",
    "open_dataset",
    "pull",
    "push",
]
