"""Tests for direct LeRobot package export writer."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from rfx.teleop import (
    LeRobotExportConfig,
    LeRobotPackageWriter,
    LeRobotRecorder,
    TeleopSessionConfig,
)


class _FakeLeRobotDataset:
    created: _FakeLeRobotDataset | None = None

    def __init__(self) -> None:
        self.frames: list[tuple[dict[str, object], str | None]] = []
        self.saved = False
        self.kwargs: dict[str, object] = {}

    @classmethod
    def create(cls, *args, **kwargs):
        instance = cls()
        instance.kwargs = dict(kwargs)
        cls.created = instance
        return instance

    def add_frame(self, frame, task=None):
        self.frames.append((frame, task))

    def save_episode(self):
        self.saved = True


def _install_fake_lerobot_modules(monkeypatch) -> None:
    package = types.ModuleType("lerobot")
    common = types.ModuleType("lerobot.common")
    datasets = types.ModuleType("lerobot.common.datasets")
    dataset_mod = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
    dataset_mod.LeRobotDataset = _FakeLeRobotDataset

    package.common = common
    common.datasets = datasets
    datasets.lerobot_dataset = dataset_mod

    monkeypatch.setitem(sys.modules, "lerobot", package)
    monkeypatch.setitem(sys.modules, "lerobot.common", common)
    monkeypatch.setitem(sys.modules, "lerobot.common.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.common.datasets.lerobot_dataset", dataset_mod)


def test_direct_lerobot_export_with_fake_package(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_FakeLeRobotDataset, "created", None)
    _install_fake_lerobot_modules(monkeypatch)

    recorder = LeRobotRecorder(tmp_path)
    config = TeleopSessionConfig.single_arm_pair(output_dir=tmp_path, cameras=())
    _ = recorder.start_episode(session_config=config, label="pick-place")
    recorder.append_control_step(
        timestamp_ns=1_000,
        dt_s=1.0 / 350.0,
        pair_positions={"main": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        camera_frame_indices={"cam0": 0},
    )
    recorder.append_camera_frame(
        camera_name="cam0",
        frame_index=0,
        timestamp_ns=2_000,
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    episode = recorder.finalize_episode()

    writer = LeRobotPackageWriter(
        LeRobotExportConfig(
            repo_id="local/test",
            root=tmp_path / "lerobot",
            fps=30,
            use_videos=False,
        )
    )
    summary = writer.write_episode(episode)

    assert summary["frames_added"] == 1
    created = _FakeLeRobotDataset.created
    assert created is not None
    assert created.saved is True
    assert len(created.frames) == 1
    frame, task = created.frames[0]
    assert task == "pick-place"
    assert "observation.state" in frame
    assert "observation.images.cam0" in frame
