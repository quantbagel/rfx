"""
rfx.teleop.lerobot_writer - Direct LeRobot package export.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from .recorder import RecordedEpisode


@dataclass(frozen=True)
class LeRobotExportConfig:
    """Configuration for direct LeRobot dataset exports."""

    repo_id: str
    root: Path | str = Path("lerobot_datasets")
    fps: int = 30
    robot_type: str = "so101_bimanual"
    use_videos: bool = True
    push_to_hub: bool = False

    def resolved_root(self) -> Path:
        return Path(self.root)


class LeRobotPackageWriter:
    """Write recorded teleop episodes directly with the LeRobot Python package."""

    def __init__(self, config: LeRobotExportConfig):
        self.config = config
        self.dataset_cls = _load_lerobot_dataset_class()

    def write_episode(self, episode: RecordedEpisode, *, task: str | None = None) -> dict[str, Any]:
        manifest = _load_json(episode.manifest_path)
        control_rows = _load_jsonl(episode.episode_dir / "control.jsonl")
        if not control_rows:
            raise ValueError("Episode has no control rows to export")

        features = _build_features(episode, control_rows)
        dataset = _create_dataset(
            self.dataset_cls,
            repo_id=self.config.repo_id,
            root=self.config.resolved_root(),
            fps=self.config.fps,
            robot_type=self.config.robot_type,
            features=features,
            use_videos=self.config.use_videos,
        )

        task_name = task or manifest.get("label") or "teleop"
        added = 0
        for row in control_rows:
            frame = _control_row_to_frame(episode.episode_dir, row)
            _dataset_add_frame(dataset, frame, task_name)
            added += 1

        _dataset_save_episode(dataset)
        if self.config.push_to_hub and hasattr(dataset, "push_to_hub"):
            dataset.push_to_hub()

        return {
            "repo_id": self.config.repo_id,
            "episode_id": episode.episode_id,
            "frames_added": added,
            "task": task_name,
            "root": str(self.config.resolved_root()),
        }


def _load_lerobot_dataset_class() -> Any:
    module_paths = (
        "lerobot.common.datasets.lerobot_dataset",
        "lerobot.datasets.lerobot_dataset",
    )
    for module_path in module_paths:
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            continue
        if hasattr(mod, "LeRobotDataset"):
            return mod.LeRobotDataset

    raise ImportError(
        "LeRobot package is unavailable. Install optional deps with: pip install -e '.[teleop-lerobot]'"
    )


def _create_dataset(
    dataset_cls: Any,
    *,
    repo_id: str,
    root: Path,
    fps: int,
    robot_type: str,
    features: dict[str, Any],
    use_videos: bool,
) -> Any:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    create = getattr(dataset_cls, "create", None)
    if create is None:
        raise AttributeError("LeRobotDataset.create is unavailable in this package version")

    attempts = [
        lambda: create(
            repo_id=repo_id,
            root=str(root),
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        ),
        lambda: create(
            repo_id,
            fps,
            root=str(root),
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        ),
        lambda: create(repo_id=repo_id, root=str(root), fps=fps, features=features),
        lambda: create(repo_id, fps, features=features, root=str(root)),
        lambda: create(repo_id=repo_id, root=str(root), fps=fps),
    ]

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:  # pragma: no cover - version compatibility branch
            last_error = exc

    raise RuntimeError("Unable to instantiate LeRobotDataset with known signatures") from last_error


def _dataset_add_frame(dataset: Any, frame: dict[str, Any], task_name: str) -> None:
    add_frame = getattr(dataset, "add_frame", None)
    if add_frame is None:
        raise AttributeError("Dataset instance has no add_frame method")

    attempts = [
        lambda: add_frame(frame, task=task_name),
        lambda: add_frame(frame),
        lambda: add_frame({**frame, "task": task_name}),
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            attempt()
            return
        except Exception as exc:  # pragma: no cover - compatibility fallback
            last_error = exc
    raise RuntimeError("Unable to add frame to LeRobot dataset") from last_error


def _dataset_save_episode(dataset: Any) -> None:
    save_episode = getattr(dataset, "save_episode", None)
    if save_episode is None:
        raise AttributeError("Dataset instance has no save_episode method")
    save_episode()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return cast(dict[str, Any], json.load(handle))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_features(episode: RecordedEpisode, control_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_row = control_rows[0]
    state_vec = _flatten_pairs(first_row.get("pairs", {}))
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (len(state_vec),)},
        "action": {"dtype": "float32", "shape": (len(state_vec),)},
    }

    camera_indices = first_row.get("camera_frame_indices", {})
    for camera_name, frame_idx in camera_indices.items():
        if frame_idx is None or int(frame_idx) < 0:
            continue
        frame_path = episode.episode_dir / "cameras" / camera_name / f"{int(frame_idx):08d}.npy"
        if not frame_path.exists():
            continue
        frame = np.load(frame_path)
        features[f"observation.images.{camera_name}"] = {
            "dtype": str(frame.dtype),
            "shape": tuple(int(v) for v in frame.shape),
        }

    return features


def _control_row_to_frame(episode_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    state = np.asarray(_flatten_pairs(row.get("pairs", {})), dtype=np.float32)
    frame: dict[str, Any] = {
        "observation.state": state,
        "action": state.copy(),
    }

    for camera_name, frame_idx in row.get("camera_frame_indices", {}).items():
        idx = int(frame_idx)
        if idx < 0:
            continue
        frame_path = episode_dir / "cameras" / camera_name / f"{idx:08d}.npy"
        if frame_path.exists():
            frame[f"observation.images.{camera_name}"] = np.load(frame_path)

    return frame


def _flatten_pairs(pairs: dict[str, Any]) -> list[float]:
    flattened: list[float] = []
    for pair_name in sorted(pairs.keys()):
        flattened.extend(float(v) for v in pairs[pair_name])
    return flattened
