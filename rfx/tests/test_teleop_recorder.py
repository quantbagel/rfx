"""Tests for LeRobot-style teleop recorder output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rfx.teleop import LeRobotRecorder, TeleopSessionConfig


def test_recorder_writes_control_and_camera_streams(tmp_path: Path) -> None:
    recorder = LeRobotRecorder(tmp_path)
    config = TeleopSessionConfig.single_arm_pair(output_dir=tmp_path, cameras=())

    episode_id = recorder.start_episode(session_config=config, label="unit-test")
    assert episode_id.startswith("episode_")

    recorder.append_control_step(
        timestamp_ns=1_000,
        dt_s=1.0 / 350.0,
        pair_positions={"main": [0.1, 0.2, 0.3]},
        camera_frame_indices={"cam0": 4},
    )
    recorder.append_camera_frame(
        camera_name="cam0",
        frame_index=4,
        timestamp_ns=2_000,
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    result = recorder.finalize_episode(loop_stats={"p99_jitter_s": 0.0004})

    assert result.control_steps == 1
    assert result.camera_frames == {"cam0": 1}
    assert result.manifest_path.exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "lerobot-v1"
    assert manifest["summary"]["control_steps"] == 1
    assert manifest["summary"]["camera_frames"]["cam0"] == 1

    control_path = result.episode_dir / "control.jsonl"
    assert control_path.exists()
    rows = control_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1

    frame_path = result.episode_dir / "cameras" / "cam0" / "00000004.npy"
    assert frame_path.exists()
