"""
rfx.teleop.recorder - High-rate episode recorder with LeRobot-style metadata.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import TeleopSessionConfig, to_serializable


@dataclass(frozen=True)
class RecordedEpisode:
    """Finalized episode artifact summary."""

    episode_id: str
    episode_dir: Path
    manifest_path: Path
    control_steps: int
    camera_frames: dict[str, int]


class LeRobotRecorder:
    """
    Recorder that writes a LeRobot-oriented episode structure.

    Layout per episode:
    - `manifest.json`
    - `control.jsonl`
    - `cameras/<camera_name>/<frame_idx>.npy`
    - `cameras/<camera_name>/index.jsonl`
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._active_episode_id: str | None = None
        self._active_episode_dir: Path | None = None
        self._manifest_path: Path | None = None
        self._control_handle = None
        self._camera_index_handles: dict[str, Any] = {}
        self._camera_dirs: dict[str, Path] = {}
        self._control_steps = 0
        self._camera_counts: dict[str, int] = {}
        self._manifest: dict[str, Any] | None = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._active_episode_id is not None

    def start_episode(
        self,
        *,
        session_config: TeleopSessionConfig | Mapping[str, Any] | None = None,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        with self._lock:
            if self._active_episode_id is not None:
                raise RuntimeError("Episode recording is already active")

            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            episode_id = f"episode_{timestamp}"
            episode_dir = self.root_dir / episode_id
            episode_dir.mkdir(parents=True, exist_ok=False)

            control_path = episode_dir / "control.jsonl"
            manifest_path = episode_dir / "manifest.json"
            (episode_dir / "cameras").mkdir(parents=True, exist_ok=True)

            config_payload: Any = None
            if session_config is not None:
                config_payload = to_serializable(session_config)

            manifest: dict[str, Any] = {
                "schema": "lerobot-v1",
                "episode_id": episode_id,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "label": label,
                "session_config": config_payload,
                "metadata": to_serializable(dict(metadata or {})),
                "paths": {
                    "control": "control.jsonl",
                    "cameras": "cameras/",
                },
                "summary": {
                    "control_steps": 0,
                    "camera_frames": {},
                },
            }

            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)

            self._active_episode_id = episode_id
            self._active_episode_dir = episode_dir
            self._manifest_path = manifest_path
            self._control_handle = open(control_path, "a", encoding="utf-8")
            self._camera_index_handles = {}
            self._camera_dirs = {}
            self._control_steps = 0
            self._camera_counts = {}
            self._manifest = manifest
            return episode_id

    def append_control_step(
        self,
        *,
        timestamp_ns: int,
        dt_s: float,
        pair_positions: Mapping[str, Sequence[float]],
        camera_frame_indices: Mapping[str, int],
    ) -> None:
        with self._lock:
            if self._active_episode_id is None or self._control_handle is None:
                raise RuntimeError("No active episode")

            row = {
                "timestamp_ns": int(timestamp_ns),
                "dt_s": float(dt_s),
                "pairs": {
                    name: [float(v) for v in values] for name, values in pair_positions.items()
                },
                "camera_frame_indices": {
                    name: int(idx) for name, idx in camera_frame_indices.items()
                },
            }
            self._control_handle.write(json.dumps(row, sort_keys=True) + "\n")
            self._control_steps += 1

    def append_camera_frame(
        self,
        *,
        camera_name: str,
        frame_index: int,
        timestamp_ns: int,
        frame: np.ndarray,
    ) -> None:
        with self._lock:
            if self._active_episode_id is None or self._active_episode_dir is None:
                raise RuntimeError("No active episode")

            camera_dir = self._camera_dirs.get(camera_name)
            if camera_dir is None:
                camera_dir = self._active_episode_dir / "cameras" / camera_name
                camera_dir.mkdir(parents=True, exist_ok=True)
                self._camera_dirs[camera_name] = camera_dir

            index_handle = self._camera_index_handles.get(camera_name)
            if index_handle is None:
                index_path = camera_dir / "index.jsonl"
                index_handle = open(index_path, "a", encoding="utf-8")
                self._camera_index_handles[camera_name] = index_handle

            file_name = f"{int(frame_index):08d}.npy"
            frame_path = camera_dir / file_name
            np.save(frame_path, np.asarray(frame), allow_pickle=False)

            index_row = {
                "frame_index": int(frame_index),
                "timestamp_ns": int(timestamp_ns),
                "path": f"cameras/{camera_name}/{file_name}",
            }
            index_handle.write(json.dumps(index_row, sort_keys=True) + "\n")
            self._camera_counts[camera_name] = self._camera_counts.get(camera_name, 0) + 1

    def finalize_episode(self, *, loop_stats: Mapping[str, Any] | None = None) -> RecordedEpisode:
        with self._lock:
            if self._active_episode_id is None:
                raise RuntimeError("No active episode")
            if (
                self._manifest is None
                or self._manifest_path is None
                or self._active_episode_dir is None
            ):
                raise RuntimeError("Recorder is missing episode metadata")

            if self._control_handle is not None:
                self._control_handle.flush()
                self._control_handle.close()
                self._control_handle = None

            for handle in self._camera_index_handles.values():
                handle.flush()
                handle.close()
            self._camera_index_handles = {}

            self._manifest["summary"]["control_steps"] = int(self._control_steps)
            self._manifest["summary"]["camera_frames"] = dict(self._camera_counts)
            if loop_stats is not None:
                self._manifest["summary"]["loop_stats"] = to_serializable(dict(loop_stats))
            self._manifest["finalized_at_utc"] = datetime.now(UTC).isoformat()

            with open(self._manifest_path, "w", encoding="utf-8") as handle:
                json.dump(self._manifest, handle, indent=2, sort_keys=True)

            result = RecordedEpisode(
                episode_id=self._active_episode_id,
                episode_dir=self._active_episode_dir,
                manifest_path=self._manifest_path,
                control_steps=int(self._control_steps),
                camera_frames=dict(self._camera_counts),
            )

            self._active_episode_id = None
            self._active_episode_dir = None
            self._manifest_path = None
            self._manifest = None
            self._camera_dirs = {}
            self._control_steps = 0
            self._camera_counts = {}

            return result

    def export_episode_to_lerobot(
        self,
        episode: RecordedEpisode,
        *,
        repo_id: str,
        root: str | Path = Path("lerobot_datasets"),
        fps: int = 30,
        robot_type: str = "so101_bimanual",
        use_videos: bool = True,
        push_to_hub: bool = False,
        task: str | None = None,
    ) -> dict[str, Any]:
        """Export a finalized episode using the installed LeRobot package."""
        from .lerobot_writer import LeRobotExportConfig, LeRobotPackageWriter

        config = LeRobotExportConfig(
            repo_id=repo_id,
            root=root,
            fps=fps,
            robot_type=robot_type,
            use_videos=use_videos,
            push_to_hub=push_to_hub,
        )
        writer = LeRobotPackageWriter(config)
        return writer.write_episode(episode, task=task)

    def export_episode_to_mcap(
        self,
        episode: RecordedEpisode,
        *,
        output_dir: str | Path = Path("mcap_exports"),
        include_camera_frames: bool = True,
    ) -> dict[str, Any]:
        from .mcap_writer import McapEpisodeWriter, McapExportConfig

        config = McapExportConfig(
            output_dir=output_dir,
            include_camera_frames=include_camera_frames,
        )
        writer = McapEpisodeWriter(config)
        return writer.write_episode(episode)

    def close(self) -> None:
        with self._lock:
            if self._active_episode_id is None:
                return
        self.finalize_episode()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
