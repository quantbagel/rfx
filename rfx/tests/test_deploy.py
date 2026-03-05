"""Tests for rfx.deploy module."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from rfx.deploy import (
    _load_policy_from_py,
    _resolve_robot_config,
    _WrapCallable,
    deploy,
)
from rfx.hub import LoadedPolicy
from rfx.robot.config import G1_CONFIG, GO2_CONFIG, SO101_CONFIG, RobotConfig

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loaded_policy(
    robot_config: RobotConfig | None = None,
    config: dict[str, Any] | None = None,
) -> LoadedPolicy:
    """Create a minimal LoadedPolicy stub for testing."""
    return LoadedPolicy(
        policy=MagicMock(),
        robot_config=robot_config,
        normalizer=None,
        config=config or {},
    )


# ---------------------------------------------------------------------------
# _resolve_robot_config
# ---------------------------------------------------------------------------


class TestResolveRobotConfig:
    def test_explicit_robot_so101(self) -> None:
        loaded = _make_loaded_policy()
        cfg = _resolve_robot_config(loaded, robot="so101")
        assert cfg.name == SO101_CONFIG.name

    def test_explicit_robot_go2(self) -> None:
        loaded = _make_loaded_policy()
        cfg = _resolve_robot_config(loaded, robot="go2")
        assert cfg.name == GO2_CONFIG.name

    def test_explicit_robot_g1(self) -> None:
        loaded = _make_loaded_policy()
        cfg = _resolve_robot_config(loaded, robot="g1")
        assert cfg.name == G1_CONFIG.name

    def test_robot_name_case_insensitive(self) -> None:
        loaded = _make_loaded_policy()
        cfg = _resolve_robot_config(loaded, robot="SO101")
        assert cfg.name == SO101_CONFIG.name

    def test_unknown_robot_raises(self) -> None:
        loaded = _make_loaded_policy()
        with pytest.raises(ValueError, match="Unknown robot type"):
            _resolve_robot_config(loaded, robot="nonexistent")

    def test_config_yaml_overrides_robot(self, tmp_path: Path) -> None:
        yaml_content = (
            "name: test-bot\n"
            "state_dim: 12\n"
            "action_dim: 6\n"
            "max_state_dim: 64\n"
            "max_action_dim: 64\n"
            "control_freq_hz: 50\n"
        )
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml_content)
        loaded = _make_loaded_policy()
        cfg = _resolve_robot_config(loaded, robot="so101", config=cfg_path)
        assert cfg.name == "test-bot"

    def test_auto_detect_from_policy(self) -> None:
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        cfg = _resolve_robot_config(loaded)
        assert cfg.name == SO101_CONFIG.name

    def test_no_robot_no_config_raises(self) -> None:
        loaded = _make_loaded_policy()
        with pytest.raises(ValueError, match="Cannot determine robot type"):
            _resolve_robot_config(loaded)


# ---------------------------------------------------------------------------
# _WrapCallable
# ---------------------------------------------------------------------------


class TestWrapCallable:
    def test_wraps_function(self) -> None:
        def my_fn(obs: Any) -> str:
            return "action"

        wrapped = _WrapCallable(my_fn)
        assert wrapped({"state": "obs"}) == "action"

    def test_policy_type(self) -> None:
        wrapped = _WrapCallable(lambda x: x)
        assert wrapped.policy_type == "python_function"

    def test_robot_config_is_none(self) -> None:
        wrapped = _WrapCallable(lambda x: x)
        assert wrapped.robot_config is None

    def test_config_is_empty_dict(self) -> None:
        wrapped = _WrapCallable(lambda x: x)
        assert wrapped.config == {}


# ---------------------------------------------------------------------------
# _load_policy_from_py
# ---------------------------------------------------------------------------


class TestLoadPolicyFromPy:
    def test_loads_decorated_policy(self, tmp_path: Path) -> None:
        py_file = tmp_path / "my_policy.py"
        py_file.write_text("import rfx\n@rfx.policy\ndef hold(obs):\n    return obs\n")
        fn = _load_policy_from_py(py_file)
        assert fn.__name__ == "hold"
        assert fn("test") == "test"

    def test_multiple_policies_raises(self, tmp_path: Path) -> None:
        py_file = tmp_path / "multi.py"
        py_file.write_text(
            "import rfx\n@rfx.policy\ndef a(obs): return obs\n@rfx.policy\ndef b(obs): return obs\n"
        )
        with pytest.raises(ValueError, match="Multiple @rfx.policy"):
            _load_policy_from_py(py_file)

    def test_no_policy_raises(self, tmp_path: Path) -> None:
        py_file = tmp_path / "empty.py"
        py_file.write_text("x = 42\n")
        with pytest.raises(ValueError, match="No @rfx.policy"):
            _load_policy_from_py(py_file)

    def test_fallback_to_policy_name(self, tmp_path: Path) -> None:
        py_file = tmp_path / "fallback.py"
        py_file.write_text("def policy(obs): return obs\n")
        fn = _load_policy_from_py(py_file)
        assert fn.__name__ == "policy"

    def test_invalid_path_raises(self, tmp_path: Path) -> None:
        py_file = tmp_path / "nonexistent.py"
        with pytest.raises((ValueError, FileNotFoundError)):
            _load_policy_from_py(py_file)


# ---------------------------------------------------------------------------
# deploy() integration (mock-based)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestDeployMock:
    def test_deploy_mock_with_duration(self, tmp_path: Path) -> None:
        """Deploy with --mock should run and return stats."""
        import torch

        # Create a minimal saved policy directory
        config = {
            "policy_type": "mlp",
            "architecture": {"type": "mlp", "hidden_dims": [64]},
            "state_dim": 12,
            "action_dim": 6,
        }
        policy_dir = tmp_path / "test-policy"
        policy_dir.mkdir()
        (policy_dir / "rfx_config.json").write_text(__import__("json").dumps(config))

        # Create a minimal model.safetensors (empty)
        # We'll monkeypatch load_policy instead
        mock_policy = MagicMock()
        mock_policy.return_value = torch.zeros(1, 64)
        mock_policy._is_torch_native = True

        loaded = LoadedPolicy(
            policy=mock_policy,
            robot_config=SO101_CONFIG,
            normalizer=None,
            config=config,
        )

        # Monkeypatch load_policy to return our mock
        import rfx.deploy as deploy_mod

        original_load = deploy_mod.load_policy
        deploy_mod.load_policy = lambda _src: loaded

        try:
            stats = deploy(
                str(policy_dir),
                robot="so101",
                mock=True,
                duration=0.5,
                rate_hz=20,
                warmup_s=0.0,
                verbose=False,
            )
            assert stats.iterations > 0
            assert stats.target_period_s == pytest.approx(1.0 / 20)
        finally:
            deploy_mod.load_policy = original_load

    def test_deploy_py_file_mock(self, tmp_path: Path) -> None:
        """Deploy a .py file with @rfx.policy should work with --mock."""

        py_file = tmp_path / "hold.py"
        py_file.write_text(
            "import torch\nimport rfx\n@rfx.policy\ndef hold(obs):\n    return torch.zeros(1, 64)\n"
        )

        stats = deploy(
            str(py_file),
            robot="so101",
            mock=True,
            duration=0.5,
            rate_hz=20,
            warmup_s=0.0,
            verbose=False,
        )
        assert stats.iterations > 0
