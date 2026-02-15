"""Tests for teleop runtime loop behavior without hardware dependencies."""

from __future__ import annotations

from pathlib import Path
import time

from rfx.teleop import BimanualSo101Session, LeRobotRecorder, TeleopSessionConfig


class _FakePair:
    def __init__(self, name: str) -> None:
        self.name = name
        self._step_count = 0

    def step(self):
        self._step_count += 1
        value = float(self._step_count)
        return [value] * 6

    def go_home(self) -> None:
        return None

    def disconnect(self) -> None:
        return None


def test_session_collects_timing_stats_and_records_episode(tmp_path: Path) -> None:
    config = TeleopSessionConfig.single_arm_pair(
        output_dir=tmp_path,
        rate_hz=200.0,
        cameras=(),
    )
    recorder = LeRobotRecorder(tmp_path)
    session = BimanualSo101Session(
        config=config,
        recorder=recorder,
        pair_factory=lambda pair_cfg: _FakePair(pair_cfg.name),
    )

    session.start()
    time.sleep(0.05)
    stats_before = session.timing_stats()
    assert stats_before.iterations > 0

    result = session.record_episode(duration_s=0.06, label="test")
    assert result.control_steps > 0
    assert result.manifest_path.exists()

    positions = session.latest_positions()
    assert "main" in positions
    assert len(positions["main"]) == 6

    stats_after = session.timing_stats()
    assert stats_after.p99_jitter_s >= 0.0
    session.stop()
