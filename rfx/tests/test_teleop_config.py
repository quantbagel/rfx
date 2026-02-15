"""Tests for teleop configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest

from rfx.teleop import ArmPairConfig, CameraStreamConfig, TeleopSessionConfig


def test_bimanual_config_defaults() -> None:
    config = TeleopSessionConfig.bimanual()
    assert len(config.arm_pairs) == 2
    assert config.rate_hz == 350.0
    assert config.period_s == pytest.approx(1.0 / 350.0)
    assert len(config.cameras) == 3


def test_single_arm_pair_config() -> None:
    config = TeleopSessionConfig.single_arm_pair(leader_port="/dev/a", follower_port="/dev/b")
    assert len(config.arm_pairs) == 1
    assert config.arm_pairs[0].leader_port == "/dev/a"
    assert config.arm_pairs[0].follower_port == "/dev/b"


def test_config_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError):
        TeleopSessionConfig(rate_hz=0)


def test_config_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError):
        TeleopSessionConfig(
            arm_pairs=(
                ArmPairConfig(name="arm", leader_port="/dev/a", follower_port="/dev/b"),
                ArmPairConfig(name="arm", leader_port="/dev/c", follower_port="/dev/d"),
            )
        )

    with pytest.raises(ValueError):
        TeleopSessionConfig(
            cameras=(
                CameraStreamConfig(name="cam", device_id=0),
                CameraStreamConfig(name="cam", device_id=1),
            )
        )


def test_output_dir_normalizes_to_path() -> None:
    config = TeleopSessionConfig(output_dir="tmp/out")
    assert isinstance(config.output_dir, Path)
    assert str(config.output_dir).endswith("tmp/out")
