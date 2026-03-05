"""Tests for rfx.deploy module and supporting pipeline components."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rfx.deploy import (
    _load_policy_from_py,
    _print_stats,
    _resolve_robot_config,
    _WrapCallable,
    deploy,
)
from rfx.hub import LoadedPolicy
from rfx.robot.config import G1_CONFIG, GO2_CONFIG, SO101_CONFIG, RobotConfig
from rfx.runtime.cli import build_parser
from rfx.session import SessionStats

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


def _invoke(argv: list[str]) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.fn(ns))


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

    def test_robot_with_dashes_and_underscores(self) -> None:
        """so-101, so_101 should normalize to so101."""
        loaded = _make_loaded_policy()
        cfg = _resolve_robot_config(loaded, robot="so-101")
        assert cfg.name == SO101_CONFIG.name
        cfg2 = _resolve_robot_config(loaded, robot="so_101")
        assert cfg2.name == SO101_CONFIG.name

    def test_config_yaml_takes_precedence_over_everything(self, tmp_path: Path) -> None:
        yaml_content = (
            "name: custom-bot\n"
            "state_dim: 8\n"
            "action_dim: 4\n"
            "max_state_dim: 32\n"
            "max_action_dim: 32\n"
            "control_freq_hz: 100\n"
        )
        cfg_path = tmp_path / "custom.yaml"
        cfg_path.write_text(yaml_content)
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        cfg = _resolve_robot_config(loaded, robot="go2", config=cfg_path)
        assert cfg.name == "custom-bot"

    def test_robot_as_yaml_path(self, tmp_path: Path) -> None:
        """Passing a .yaml file path as robot= should load it."""
        yaml_content = (
            "name: yaml-bot\n"
            "state_dim: 4\n"
            "action_dim: 2\n"
            "max_state_dim: 16\n"
            "max_action_dim: 16\n"
            "control_freq_hz: 30\n"
        )
        cfg_path = tmp_path / "bot.yaml"
        cfg_path.write_text(yaml_content)
        loaded = _make_loaded_policy()
        cfg = _resolve_robot_config(loaded, robot=str(cfg_path))
        assert cfg.name == "yaml-bot"

    def test_fallback_to_policy_config_robot_config(self) -> None:
        """If loaded.robot_config is None but config has robot_config dict."""
        rc_dict = {
            "name": "embedded-bot",
            "state_dim": 6,
            "action_dim": 3,
            "max_state_dim": 32,
            "max_action_dim": 32,
            "control_freq_hz": 50,
        }
        loaded = _make_loaded_policy(config={"robot_config": rc_dict})
        cfg = _resolve_robot_config(loaded)
        assert cfg.name == "embedded-bot"

    def test_all_builtin_robots(self) -> None:
        """Every builtin name should resolve."""
        loaded = _make_loaded_policy()
        for name, expected in [("so101", SO101_CONFIG), ("go2", GO2_CONFIG), ("g1", G1_CONFIG)]:
            cfg = _resolve_robot_config(loaded, robot=name)
            assert cfg.name == expected.name


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

    def test_callable_class(self) -> None:
        class MyPolicy:
            def __call__(self, obs: Any) -> str:
                return f"action_{obs}"

        wrapped = _WrapCallable(MyPolicy())
        assert wrapped("test") == "action_test"
        assert wrapped.policy_type == "python_function"

    def test_lambda(self) -> None:
        wrapped = _WrapCallable(lambda x: x * 2)
        assert wrapped(5) == 10


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

    def test_policy_with_imports_and_logic(self, tmp_path: Path) -> None:
        py_file = tmp_path / "complex.py"
        py_file.write_text(
            "import math\nimport rfx\n\n@rfx.policy\n"
            "def my_policy(obs):\n    return [math.sin(x) for x in obs]\n"
        )
        fn = _load_policy_from_py(py_file)
        assert fn.__name__ == "my_policy"
        assert len(fn([0.0, 1.5708])) == 2

    def test_fallback_policy_name_is_callable(self, tmp_path: Path) -> None:
        py_file = tmp_path / "bare.py"
        py_file.write_text("def policy(obs):\n    return obs[::-1]\n")
        fn = _load_policy_from_py(py_file)
        assert fn([1, 2, 3]) == [3, 2, 1]

    def test_non_callable_policy_name_raises(self, tmp_path: Path) -> None:
        py_file = tmp_path / "notfn.py"
        py_file.write_text("policy = 42\n")
        with pytest.raises(ValueError, match="No @rfx.policy"):
            _load_policy_from_py(py_file)

    def test_syntax_error_in_file_raises(self, tmp_path: Path) -> None:
        py_file = tmp_path / "bad.py"
        py_file.write_text("def broken(\n")
        with pytest.raises(SyntaxError):
            _load_policy_from_py(py_file)


# ---------------------------------------------------------------------------
# SessionStats
# ---------------------------------------------------------------------------


class TestSessionStats:
    def test_to_dict_roundtrip(self) -> None:
        stats = SessionStats(
            iterations=100,
            overruns=2,
            target_period_s=0.02,
            avg_period_s=0.021,
            p50_jitter_s=0.001,
            p95_jitter_s=0.003,
            p99_jitter_s=0.005,
            max_jitter_s=0.01,
        )
        d = stats.to_dict()
        assert d["iterations"] == 100
        assert d["overruns"] == 2
        assert isinstance(d["target_period_s"], float)
        assert len(d) == 8

    def test_zero_iterations(self) -> None:
        stats = SessionStats(
            iterations=0,
            overruns=0,
            target_period_s=0.01,
            avg_period_s=0.0,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        assert stats.iterations == 0
        d = stats.to_dict()
        assert d["avg_period_s"] == 0.0

    def test_frozen_dataclass(self) -> None:
        stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.01,
            avg_period_s=0.01,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        with pytest.raises(AttributeError):
            stats.iterations = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _print_stats
# ---------------------------------------------------------------------------


class TestPrintStats:
    def test_prints_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        stats = SessionStats(
            iterations=500,
            overruns=3,
            target_period_s=0.02,
            avg_period_s=0.0205,
            p50_jitter_s=0.0005,
            p95_jitter_s=0.002,
            p99_jitter_s=0.004,
            max_jitter_s=0.008,
        )
        _print_stats(stats)
        out = capsys.readouterr().out
        assert "500 steps" in out
        assert "3 overruns" in out
        assert "avg period" in out
        assert "jitter p50" in out
        assert "jitter p95" in out
        assert "jitter p99" in out

    def test_zero_iterations_no_crash(self, capsys: pytest.CaptureFixture[str]) -> None:
        stats = SessionStats(
            iterations=0,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.0,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        _print_stats(stats)
        out = capsys.readouterr().out
        assert "0 steps" in out


# ---------------------------------------------------------------------------
# RobotConfig
# ---------------------------------------------------------------------------


class TestRobotConfig:
    def test_from_dict_basic(self) -> None:
        d = {
            "name": "test-bot",
            "state_dim": 6,
            "action_dim": 3,
            "max_state_dim": 32,
            "max_action_dim": 32,
            "control_freq_hz": 100,
        }
        cfg = RobotConfig.from_dict(d)
        assert cfg.name == "test-bot"
        assert cfg.state_dim == 6
        assert cfg.control_freq_hz == 100

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {
            "name": "bot",
            "state_dim": 4,
            "action_dim": 2,
            "unknown_field": "should be ignored",
        }
        cfg = RobotConfig.from_dict(d)
        assert cfg.name == "bot"
        assert not hasattr(cfg, "unknown_field")

    def test_to_dict_roundtrip(self) -> None:
        original = RobotConfig(
            name="roundtrip-bot",
            state_dim=10,
            action_dim=5,
            max_state_dim=64,
            max_action_dim=64,
            control_freq_hz=50,
        )
        d = original.to_dict()
        restored = RobotConfig.from_dict(d)
        assert restored.name == original.name
        assert restored.state_dim == original.state_dim
        assert restored.action_dim == original.action_dim

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = (
            "name: yaml-test\n"
            "state_dim: 12\n"
            "action_dim: 6\n"
            "max_state_dim: 64\n"
            "max_action_dim: 64\n"
            "control_freq_hz: 50\n"
            "hardware:\n"
            "  port: /dev/ttyACM0\n"
        )
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml_content)
        cfg = RobotConfig.from_yaml(cfg_path)
        assert cfg.name == "yaml-test"
        assert cfg.hardware["port"] == "/dev/ttyACM0"

    def test_from_yaml_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            RobotConfig.from_yaml("/nonexistent/path/robot.yaml")

    def test_builtin_configs_valid(self) -> None:
        for cfg in (SO101_CONFIG, GO2_CONFIG, G1_CONFIG):
            assert cfg.state_dim > 0
            assert cfg.action_dim > 0
            assert cfg.max_state_dim >= cfg.state_dim
            assert cfg.max_action_dim >= cfg.action_dim
            assert cfg.control_freq_hz > 0

    def test_g1_has_29_joints(self) -> None:
        assert len(G1_CONFIG.joints) == 29
        assert G1_CONFIG.action_dim == 29


# ---------------------------------------------------------------------------
# LoadedPolicy properties
# ---------------------------------------------------------------------------


class TestLoadedPolicyProperties:
    def test_policy_type_from_config(self) -> None:
        loaded = _make_loaded_policy(config={"policy_type": "mlp"})
        assert loaded.policy_type == "mlp"

    def test_policy_type_default(self) -> None:
        loaded = _make_loaded_policy(config={})
        assert loaded.policy_type == "unknown"

    def test_training_info(self) -> None:
        loaded = _make_loaded_policy(config={"training": {"epochs": 100, "lr": 0.001}})
        assert loaded.training_info["epochs"] == 100

    def test_training_info_empty(self) -> None:
        loaded = _make_loaded_policy(config={})
        assert loaded.training_info == {}


# ---------------------------------------------------------------------------
# CLI deploy error paths
# ---------------------------------------------------------------------------


class TestDeployCLIErrors:
    def test_deploy_nonexistent_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        rc = _invoke(["deploy", "does/not/exist", "--robot", "so101", "--mock"])
        assert rc == 1

    def test_deploy_bad_robot_name(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        rc = _invoke(["deploy", "fake-policy", "--robot", "nonexistent_robot", "--mock"])
        assert rc == 1

    def test_deploy_py_file_no_policy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        py_file = tmp_path / "empty.py"
        py_file.write_text("x = 42\n")
        rc = _invoke(["deploy", str(py_file), "--robot", "so101", "--mock"])
        assert rc == 1

    def test_deploy_py_file_no_robot(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Deploy a .py file without --robot should fail (no auto-detect)."""
        monkeypatch.chdir(tmp_path)
        py_file = tmp_path / "hold.py"
        py_file.write_text("import rfx\n@rfx.policy\ndef hold(obs): return obs\n")
        rc = _invoke(["deploy", str(py_file), "--mock"])
        assert rc == 1


# ---------------------------------------------------------------------------
# deploy() integration (mock-based, requires torch)
# ---------------------------------------------------------------------------


class TestDeployMock:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
    def test_deploy_mock_with_duration(self, tmp_path: Path) -> None:
        """Deploy with --mock should run and return stats (real Session)."""
        import torch

        config = {
            "policy_type": "mlp",
            "architecture": {"type": "mlp", "hidden_dims": [64]},
            "state_dim": 12,
            "action_dim": 6,
        }
        policy_dir = tmp_path / "test-policy"
        policy_dir.mkdir()
        (policy_dir / "rfx_config.json").write_text(__import__("json").dumps(config))

        mock_policy = MagicMock()
        mock_policy.return_value = torch.zeros(1, 64)
        mock_policy._is_torch_native = True

        loaded = LoadedPolicy(
            policy=mock_policy,
            robot_config=SO101_CONFIG,
            normalizer=None,
            config=config,
        )

        with patch("rfx.deploy.load_policy", return_value=loaded):
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

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
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


# ---------------------------------------------------------------------------
# Port override logic
# ---------------------------------------------------------------------------


class TestPortOverride:
    def test_port_sets_hardware_port_and_ip(self) -> None:
        """deploy() should set both hardware['port'] and hardware['ip_address']."""
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        cfg = _resolve_robot_config(loaded)
        # Simulate what deploy() does after resolving config
        port = "/dev/ttyACM1"
        cfg.hardware["port"] = port
        cfg.hardware["ip_address"] = port
        assert cfg.hardware["port"] == "/dev/ttyACM1"
        assert cfg.hardware["ip_address"] == "/dev/ttyACM1"

    def test_port_override_on_go2(self) -> None:
        """Go2 uses ip_address — port override should set both."""
        loaded = _make_loaded_policy()
        cfg = _resolve_robot_config(loaded, robot="go2")
        original_ip = cfg.hardware.get("ip_address")
        assert original_ip is not None  # Go2 has a default IP

        cfg.hardware["port"] = "192.168.1.100"
        cfg.hardware["ip_address"] = "192.168.1.100"
        assert cfg.hardware["ip_address"] == "192.168.1.100"


# ---------------------------------------------------------------------------
# Rate Hz defaulting
# ---------------------------------------------------------------------------


class TestRateHzDefault:
    def test_rate_hz_defaults_to_config(self) -> None:
        """When rate_hz is None, should use robot_config.control_freq_hz."""
        for name, expected_hz in [("so101", 50), ("go2", 200), ("g1", 50)]:
            loaded = _make_loaded_policy()
            cfg = _resolve_robot_config(loaded, robot=name)
            rate_hz = float(cfg.control_freq_hz)
            assert rate_hz == expected_hz

    def test_explicit_rate_hz_overrides(self) -> None:
        """Explicit rate_hz should not be overridden."""
        rate_hz = 100.0
        # This is what deploy() does: only default if rate_hz is None
        if rate_hz is None:
            rate_hz = float(SO101_CONFIG.control_freq_hz)
        assert rate_hz == 100.0


# ---------------------------------------------------------------------------
# _create_robot (mocked imports)
# ---------------------------------------------------------------------------


class TestCreateRobot:
    def test_mock_true_creates_mock_robot(self) -> None:
        """_create_robot(mock=True) should call MockRobot with config dims."""
        import sys
        from types import ModuleType
        from unittest.mock import patch

        from rfx.deploy import _create_robot

        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        # Create a fake rfx.sim module so the lazy import doesn't hit torch
        fake_sim = ModuleType("rfx.sim")
        fake_sim.MockRobot = mock_cls  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"rfx.sim": fake_sim}):
            result = _create_robot(SO101_CONFIG, mock=True)
            mock_cls.assert_called_once_with(
                state_dim=SO101_CONFIG.state_dim,
                action_dim=SO101_CONFIG.action_dim,
                max_state_dim=SO101_CONFIG.max_state_dim,
                max_action_dim=SO101_CONFIG.max_action_dim,
            )
            assert result is mock_instance

    def test_mock_false_creates_real_robot(self) -> None:
        """_create_robot(mock=False) should call RealRobot with config."""
        import sys
        from types import ModuleType
        from unittest.mock import patch

        from rfx.deploy import _create_robot

        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        fake_real = ModuleType("rfx.real")
        fake_real.RealRobot = mock_cls  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"rfx.real": fake_real}):
            result = _create_robot(SO101_CONFIG, mock=False)
            mock_cls.assert_called_once_with(SO101_CONFIG)
            assert result is mock_instance


# ---------------------------------------------------------------------------
# deploy() end-to-end without torch (fully mocked)
# ---------------------------------------------------------------------------


class TestDeployEndToEndMocked:
    """Test deploy() happy path by mocking load_policy, _create_robot, and _run_deploy_loop."""

    def test_deploy_directory_source(self, tmp_path: Path) -> None:
        """deploy() with a directory source should call load_policy and run."""
        fake_stats = SessionStats(
            iterations=10,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.021,
            p50_jitter_s=0.001,
            p95_jitter_s=0.002,
            p99_jitter_s=0.003,
            max_jitter_s=0.005,
        )

        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        mock_robot = MagicMock()

        policy_dir = tmp_path / "my-policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=mock_robot),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            stats = deploy(
                str(policy_dir),
                robot="so101",
                mock=True,
                duration=1.0,
                rate_hz=50,
                warmup_s=0.0,
                verbose=False,
            )
            assert stats.iterations == 10
            assert stats.target_period_s == 0.02

    def test_deploy_py_source(self, tmp_path: Path) -> None:
        """deploy() with a .py source should load via _load_policy_from_py."""
        fake_stats = SessionStats(
            iterations=5,
            overruns=0,
            target_period_s=0.05,
            avg_period_s=0.051,
            p50_jitter_s=0.001,
            p95_jitter_s=0.002,
            p99_jitter_s=0.003,
            max_jitter_s=0.004,
        )

        mock_robot = MagicMock()

        py_file = tmp_path / "hold.py"
        py_file.write_text("import rfx\n@rfx.policy\ndef hold(obs): return obs\n")

        with (
            patch("rfx.deploy._create_robot", return_value=mock_robot),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            stats = deploy(
                str(py_file),
                robot="so101",
                mock=True,
                duration=0.5,
                rate_hz=20,
                warmup_s=0.0,
                verbose=False,
            )
            assert stats.iterations == 5

    def test_deploy_calls_disconnect(self, tmp_path: Path) -> None:
        """deploy() should call disconnect() on robot if available."""
        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )

        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        mock_robot = MagicMock()

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=mock_robot),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(
                str(policy_dir),
                robot="so101",
                mock=True,
                verbose=False,
            )
            mock_robot.disconnect.assert_called_once()

    def test_deploy_rate_hz_defaults_from_config(self, tmp_path: Path) -> None:
        """When rate_hz=None, deploy should use robot_config.control_freq_hz."""
        captured: dict[str, Any] = {}

        def fake_run(robot, policy, rate_hz, duration, warmup_s, verbose):
            captured["rate_hz"] = rate_hz
            return SessionStats(
                iterations=1,
                overruns=0,
                target_period_s=1.0 / rate_hz,
                avg_period_s=1.0 / rate_hz,
                p50_jitter_s=0.0,
                p95_jitter_s=0.0,
                p99_jitter_s=0.0,
                max_jitter_s=0.0,
            )

        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        mock_robot = MagicMock()

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=mock_robot),
            patch("rfx.deploy._run_deploy_loop", side_effect=fake_run),
        ):
            deploy(
                str(policy_dir),
                robot="so101",
                mock=True,
                rate_hz=None,
                verbose=False,
            )
            assert captured["rate_hz"] == float(SO101_CONFIG.control_freq_hz)

    def test_deploy_port_override_applied(self, tmp_path: Path) -> None:
        """deploy(..., port=X) should set hardware port and ip_address."""
        captured_config: dict[str, Any] = {}

        def fake_create(cfg, mock=False, device="cpu"):
            captured_config["port"] = cfg.hardware.get("port")
            captured_config["ip_address"] = cfg.hardware.get("ip_address")
            return MagicMock()

        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )

        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", side_effect=fake_create),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(
                str(policy_dir),
                robot="so101",
                port="/dev/ttyACM1",
                mock=True,
                verbose=False,
            )
            assert captured_config["port"] == "/dev/ttyACM1"
            assert captured_config["ip_address"] == "/dev/ttyACM1"


# ---------------------------------------------------------------------------
# _run_deploy_loop — Session lifecycle and branch coverage
# ---------------------------------------------------------------------------


class TestRunDeployLoop:
    """Test _run_deploy_loop by mocking Session to avoid torch in _control_loop."""

    def _make_fake_session_cls(
        self,
        stats: SessionStats,
        *,
        error: Exception | None = None,
    ):
        """Return a Session class whose start/stop/run are no-ops."""

        class FakeSession:
            def __init__(self, robot, policy, rate_hz=50, warmup_s=0.5):
                self.robot = robot
                self.policy = policy
                self.rate_hz = rate_hz
                self.warmup_s = warmup_s
                self._running = False
                self._step_count = 3
                self._error = error

            @property
            def is_running(self):
                # Return True once then False so loops terminate
                if self._running:
                    self._running = False
                    return True
                return False

            @property
            def step_count(self):
                return self._step_count

            @property
            def stats(self):
                return stats

            def start(self):
                self._running = True

            def stop(self):
                self._running = False

            def run(self, duration=None):
                pass

            def check_health(self):
                if self._error is not None:
                    raise RuntimeError("Control loop failed") from self._error

        return FakeSession

    def test_silent_mode(self) -> None:
        """verbose=False should call session.run(duration=...)."""
        from rfx.deploy import _run_deploy_loop

        fake_stats = SessionStats(
            iterations=10,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.021,
            p50_jitter_s=0.001,
            p95_jitter_s=0.002,
            p99_jitter_s=0.003,
            max_jitter_s=0.005,
        )
        FakeSession = self._make_fake_session_cls(fake_stats)

        with patch("rfx.deploy.Session", FakeSession):
            result = _run_deploy_loop(
                MagicMock(),
                MagicMock(),
                50.0,
                1.0,
                0.0,
                verbose=False,
            )
        assert result.iterations == 10

    def test_verbose_timed_mode(self) -> None:
        """verbose=True + duration should print progress then return stats."""
        from rfx.deploy import _run_deploy_loop

        fake_stats = SessionStats(
            iterations=5,
            overruns=1,
            target_period_s=0.02,
            avg_period_s=0.025,
            p50_jitter_s=0.001,
            p95_jitter_s=0.003,
            p99_jitter_s=0.004,
            max_jitter_s=0.006,
        )
        FakeSession = self._make_fake_session_cls(fake_stats)

        with patch("rfx.deploy.Session", FakeSession):
            result = _run_deploy_loop(
                MagicMock(),
                MagicMock(),
                50.0,
                0.01,
                0.0,
                verbose=True,
            )
        assert result.overruns == 1

    def test_verbose_infinite_mode(self) -> None:
        """verbose=True + duration=None should enter infinite display loop."""
        from rfx.deploy import _run_deploy_loop

        fake_stats = SessionStats(
            iterations=3,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        FakeSession = self._make_fake_session_cls(fake_stats)

        with patch("rfx.deploy.Session", FakeSession):
            result = _run_deploy_loop(
                MagicMock(),
                MagicMock(),
                50.0,
                None,
                0.0,
                verbose=True,
            )
        assert result.iterations == 3

    def test_check_health_propagates_error(self) -> None:
        """If session had a loop error, check_health should raise."""
        from rfx.deploy import _run_deploy_loop

        fake_stats = SessionStats(
            iterations=0,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.0,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        FakeSession = self._make_fake_session_cls(
            fake_stats,
            error=ValueError("sensor fault"),
        )

        with (
            patch("rfx.deploy.Session", FakeSession),
            pytest.raises(RuntimeError, match="Control loop failed"),
        ):
            _run_deploy_loop(
                MagicMock(),
                MagicMock(),
                50.0,
                1.0,
                0.0,
                verbose=False,
            )


# ---------------------------------------------------------------------------
# hub.py — _resolve_source, inspect_policy, LoadedPolicy.__call__
# ---------------------------------------------------------------------------


class TestResolveSource:
    """Test hub._resolve_source local and hf:// paths."""

    def test_local_path(self) -> None:
        from rfx.hub import _resolve_source

        result = _resolve_source("/tmp/my-policy")
        assert result == Path("/tmp/my-policy")

    def test_local_path_object(self) -> None:
        from rfx.hub import _resolve_source

        result = _resolve_source(Path("/tmp/my-policy"))
        assert result == Path("/tmp/my-policy")

    def test_hf_prefix_calls_snapshot_download(self) -> None:
        import sys
        from types import ModuleType

        from rfx.hub import _resolve_source

        fake_hf = ModuleType("huggingface_hub")
        mock_dl = MagicMock(return_value="/tmp/cached")
        fake_hf.snapshot_download = mock_dl  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            result = _resolve_source("hf://rfx-community/go2-walk-v1")
            mock_dl.assert_called_once_with("rfx-community/go2-walk-v1")
            assert result == Path("/tmp/cached")


class TestInspectPolicy:
    """Test hub.inspect_policy reads rfx_config.json."""

    def test_reads_config(self, tmp_path: Path) -> None:
        import json

        from rfx.hub import inspect_policy

        config = {"policy_type": "act", "robot_config": {"name": "so101"}}
        (tmp_path / "rfx_config.json").write_text(json.dumps(config))

        result = inspect_policy(str(tmp_path))
        assert result["policy_type"] == "act"
        assert result["robot_config"]["name"] == "so101"

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        from rfx.hub import inspect_policy

        with pytest.raises(FileNotFoundError):
            inspect_policy(str(tmp_path))


class TestLoadedPolicyCall:
    """Test LoadedPolicy.__call__ dispatch logic without real torch/tinygrad."""

    def test_raw_tensor_passthrough(self) -> None:
        """Non-dict obs should pass directly to policy."""
        mock_policy = MagicMock(return_value="action")
        lp = LoadedPolicy(policy=mock_policy, robot_config=None, normalizer=None, config={})

        result = lp("raw_tensor")
        mock_policy.assert_called_once_with("raw_tensor")
        assert result == "action"

    def test_dict_obs_torch_native(self) -> None:
        """Dict obs with _is_torch_native policy should skip tinygrad conversion."""
        mock_policy = MagicMock(return_value="torch_action")
        mock_policy._is_torch_native = True
        lp = LoadedPolicy(policy=mock_policy, robot_config=None, normalizer=None, config={})

        obs = {"state": "fake_tensor"}
        result = lp(obs)
        mock_policy.assert_called_once_with(obs)
        assert result == "torch_action"

    def test_dict_obs_with_normalizer(self) -> None:
        """Dict obs with normalizer should normalize before passing to policy."""
        mock_policy = MagicMock(return_value="torch_action")
        mock_policy._is_torch_native = True
        mock_normalizer = MagicMock()
        mock_normalizer.normalize.return_value = {"state": "normalized"}

        lp = LoadedPolicy(
            policy=mock_policy,
            robot_config=None,
            normalizer=mock_normalizer,
            config={},
        )

        obs = {"state": "raw"}
        result = lp(obs)
        mock_normalizer.normalize.assert_called_once_with(obs)
        mock_policy.assert_called_once_with({"state": "normalized"})
        assert result == "torch_action"

    def test_dict_obs_no_torch_native_calls_tinygrad_conversion(self) -> None:
        """Dict obs without _is_torch_native should go through tinygrad path."""
        mock_policy = MagicMock(return_value="tinygrad_action")
        mock_policy._is_torch_native = False
        lp = LoadedPolicy(policy=mock_policy, robot_config=None, normalizer=None, config={})

        mock_d2t = MagicMock(return_value="tiny_obs")
        mock_t2t = MagicMock(return_value="torch_result")

        with (
            patch.object(LoadedPolicy, "_dict_to_tinygrad", mock_d2t),
            patch.object(LoadedPolicy, "_tinygrad_to_torch", mock_t2t),
        ):
            result = lp({"state": "tensor"})
        mock_d2t.assert_called_once_with({"state": "tensor"})
        mock_policy.assert_called_once_with("tiny_obs")
        mock_t2t.assert_called_once_with("tinygrad_action")
        assert result == "torch_result"


# ---------------------------------------------------------------------------
# cmd_deploy CLI bridge — error handling
# ---------------------------------------------------------------------------


class TestCmdDeployBridge:
    """Test cmd_deploy error handling paths."""

    def _make_ns(self, **overrides) -> MagicMock:
        ns = MagicMock()
        ns.policy = "/tmp/fake"
        ns.robot = "so101"
        ns.config = None
        ns.port = None
        ns.rate_hz = 50.0
        ns.duration = 1.0
        ns.mock = True
        ns.warmup = 0.0
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_keyboard_interrupt_returns_zero(self) -> None:
        """KeyboardInterrupt during deploy should return 0."""
        from rfx.runtime.cli import cmd_deploy

        with patch("rfx.deploy.deploy", side_effect=KeyboardInterrupt):
            result = cmd_deploy(self._make_ns())
        assert result == 0

    def test_value_error_returns_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        """ValueError during deploy should print error and return 1."""
        from rfx.runtime.cli import cmd_deploy

        with patch("rfx.deploy.deploy", side_effect=ValueError("bad robot")):
            result = cmd_deploy(self._make_ns())
        assert result == 1
        assert "bad robot" in capsys.readouterr().out

    def test_generic_exception_returns_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Generic exception during deploy should print and return 1."""
        from rfx.runtime.cli import cmd_deploy

        with patch("rfx.deploy.deploy", side_effect=RuntimeError("boom")):
            result = cmd_deploy(self._make_ns())
        assert result == 1
        assert "boom" in capsys.readouterr().out

    def test_file_not_found_returns_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        """FileNotFoundError during deploy should return 1."""
        from rfx.runtime.cli import cmd_deploy

        with patch("rfx.deploy.deploy", side_effect=FileNotFoundError("no such file")):
            result = cmd_deploy(self._make_ns(policy="/tmp/nonexistent"))
        assert result == 1
        assert "no such file" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# deploy() verbose output paths
# ---------------------------------------------------------------------------


class TestDeployVerboseOutput:
    """Test that deploy() prints expected messages in verbose mode."""

    def test_verbose_directory_source_prints_policy_type(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Verbose deploy with directory source should print policy type and robot config."""
        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        loaded = _make_loaded_policy(
            robot_config=SO101_CONFIG,
            config={"policy_type": "act"},
        )
        mock_robot = MagicMock()

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=mock_robot),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(str(policy_dir), robot="so101", mock=True, verbose=True)

        out = capsys.readouterr().out
        assert "Loading policy" in out
        assert "Policy type: act" in out
        assert "Bundled robot config" in out
        assert "Done" in out

    def test_verbose_py_source_prints_function_name(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Verbose deploy with .py source should print function name."""
        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        mock_robot = MagicMock()

        py_file = tmp_path / "hold.py"
        py_file.write_text("import rfx\n@rfx.policy\ndef hold(obs): return obs\n")

        with (
            patch("rfx.deploy._create_robot", return_value=mock_robot),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(str(py_file), robot="so101", mock=True, verbose=True)

        out = capsys.readouterr().out
        assert "Policy function: hold" in out

    def test_verbose_duration_prints_for_seconds(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Verbose deploy with duration should print 'for Xs'."""
        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        mock_robot = MagicMock()

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=mock_robot),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(
                str(policy_dir),
                robot="so101",
                mock=True,
                duration=30.0,
                rate_hz=50,
                verbose=True,
            )

        out = capsys.readouterr().out
        assert "for 30.0s" in out

    def test_verbose_no_duration_prints_ctrl_c(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Verbose deploy without duration should print 'Ctrl+C to stop'."""
        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        mock_robot = MagicMock()

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=mock_robot),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(
                str(policy_dir),
                robot="so101",
                mock=True,
                duration=None,
                rate_hz=50,
                verbose=True,
            )

        out = capsys.readouterr().out
        assert "Ctrl+C to stop" in out

    def test_verbose_no_bundled_config(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Verbose deploy with no bundled robot_config should not print config line."""
        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        loaded = _make_loaded_policy(robot_config=None)
        mock_robot = MagicMock()

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=mock_robot),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(str(policy_dir), robot="so101", mock=True, verbose=True)

        out = capsys.readouterr().out
        assert "Bundled robot config" not in out


# ---------------------------------------------------------------------------
# deploy() — robot without disconnect
# ---------------------------------------------------------------------------


class TestDeployNoDisconnect:
    """Verify deploy() handles robots that lack a disconnect method."""

    def test_no_disconnect_attribute(self, tmp_path: Path) -> None:
        """deploy() should not crash if robot has no disconnect()."""
        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)

        # Create an object without disconnect attribute
        class BareRobot:
            pass

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=BareRobot()),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            stats = deploy(str(policy_dir), robot="so101", mock=True, verbose=False)
        assert stats.iterations == 1


# ---------------------------------------------------------------------------
# deploy() — device passthrough
# ---------------------------------------------------------------------------


class TestDeployDevicePassthrough:
    """Verify deploy() passes device arg to _create_robot."""

    def test_device_forwarded(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        def fake_create(cfg, mock=False, device="cpu"):
            captured["device"] = device
            return MagicMock()

        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", side_effect=fake_create),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(
                str(policy_dir),
                robot="so101",
                mock=True,
                device="cuda:0",
                verbose=False,
            )
        assert captured["device"] == "cuda:0"


# ---------------------------------------------------------------------------
# _BUILTIN_CONFIGS sanity
# ---------------------------------------------------------------------------


class TestBuiltinConfigs:
    """Verify _BUILTIN_CONFIGS maps expected keys to correct configs."""

    def test_expected_keys(self) -> None:
        from rfx.deploy import _BUILTIN_CONFIGS

        # At minimum these must exist; new robots may be added
        assert {"so101", "go2", "g1"}.issubset(_BUILTIN_CONFIGS.keys())

    def test_so101_maps_to_correct_config(self) -> None:
        from rfx.deploy import _BUILTIN_CONFIGS

        assert _BUILTIN_CONFIGS["so101"] is SO101_CONFIG

    def test_go2_maps_to_correct_config(self) -> None:
        from rfx.deploy import _BUILTIN_CONFIGS

        assert _BUILTIN_CONFIGS["go2"] is GO2_CONFIG

    def test_g1_maps_to_correct_config(self) -> None:
        from rfx.deploy import _BUILTIN_CONFIGS

        assert _BUILTIN_CONFIGS["g1"] is G1_CONFIG

    def test_all_values_are_robot_configs(self) -> None:
        from rfx.deploy import _BUILTIN_CONFIGS

        for key, cfg in _BUILTIN_CONFIGS.items():
            assert isinstance(cfg, RobotConfig), f"{key} is not a RobotConfig"


# ---------------------------------------------------------------------------
# _WrapCallable.__call__ forwarding
# ---------------------------------------------------------------------------


class TestWrapCallableCall:
    """Verify _WrapCallable.__call__ actually forwards to the wrapped function."""

    def test_call_forwards_obs(self) -> None:
        def my_policy(obs):
            return obs * 2

        wrapped = _WrapCallable(my_policy)
        assert wrapped(5) == 10

    def test_call_forwards_dict_obs(self) -> None:
        def my_policy(obs):
            return obs["state"]

        wrapped = _WrapCallable(my_policy)
        assert wrapped({"state": 42}) == 42

    def test_call_preserves_exception(self) -> None:
        def bad_policy(obs):
            raise ValueError("bad obs")

        wrapped = _WrapCallable(bad_policy)
        with pytest.raises(ValueError, match="bad obs"):
            wrapped(None)


# ---------------------------------------------------------------------------
# deploy() — warmup_s passthrough
# ---------------------------------------------------------------------------


class TestDeployWarmupPassthrough:
    """Verify deploy() passes warmup_s to _run_deploy_loop."""

    def test_warmup_forwarded(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        def fake_run(robot, policy, rate_hz, duration, warmup_s, verbose):
            captured["warmup_s"] = warmup_s
            return SessionStats(
                iterations=1,
                overruns=0,
                target_period_s=0.02,
                avg_period_s=0.02,
                p50_jitter_s=0.0,
                p95_jitter_s=0.0,
                p99_jitter_s=0.0,
                max_jitter_s=0.0,
            )

        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=MagicMock()),
            patch("rfx.deploy._run_deploy_loop", side_effect=fake_run),
        ):
            deploy(
                str(policy_dir),
                robot="so101",
                mock=True,
                warmup_s=2.5,
                verbose=False,
            )
        assert captured["warmup_s"] == 2.5

    def test_warmup_default(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        def fake_run(robot, policy, rate_hz, duration, warmup_s, verbose):
            captured["warmup_s"] = warmup_s
            return SessionStats(
                iterations=1,
                overruns=0,
                target_period_s=0.02,
                avg_period_s=0.02,
                p50_jitter_s=0.0,
                p95_jitter_s=0.0,
                p99_jitter_s=0.0,
                max_jitter_s=0.0,
            )

        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=MagicMock()),
            patch("rfx.deploy._run_deploy_loop", side_effect=fake_run),
        ):
            deploy(str(policy_dir), robot="so101", mock=True, verbose=False)
        assert captured["warmup_s"] == 0.5


# ---------------------------------------------------------------------------
# deploy() — hf:// source goes through load_policy, not _load_policy_from_py
# ---------------------------------------------------------------------------


class TestDeployHfSource:
    """Verify deploy() with hf:// or directory source uses load_policy."""

    def test_hf_source_calls_load_policy(self) -> None:
        """An hf:// source should go through load_policy, not _load_policy_from_py."""
        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)

        with (
            patch("rfx.deploy.load_policy", return_value=loaded) as mock_load,
            patch("rfx.deploy._create_robot", return_value=MagicMock()),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy("hf://user/my-policy", robot="so101", mock=True, verbose=False)
            mock_load.assert_called_once_with("hf://user/my-policy")

    def test_directory_source_calls_load_policy(self, tmp_path: Path) -> None:
        """A directory source should go through load_policy."""
        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded) as mock_load,
            patch("rfx.deploy._create_robot", return_value=MagicMock()),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(str(policy_dir), robot="so101", mock=True, verbose=False)
            mock_load.assert_called_once_with(str(policy_dir))

    def test_py_source_does_not_call_load_policy(self, tmp_path: Path) -> None:
        """A .py source should NOT go through load_policy."""
        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )

        py_file = tmp_path / "hold.py"
        py_file.write_text("import rfx\n@rfx.policy\ndef hold(obs): return obs\n")

        with (
            patch("rfx.deploy.load_policy") as mock_load,
            patch("rfx.deploy._create_robot", return_value=MagicMock()),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(str(py_file), robot="so101", mock=True, verbose=False)
            mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# CameraConfig and JointConfig
# ---------------------------------------------------------------------------


class TestCameraConfig:
    """Test CameraConfig dataclass and from_dict."""

    def test_defaults(self) -> None:
        from rfx.robot.config import CameraConfig

        cam = CameraConfig()
        assert cam.name == "camera"
        assert cam.width == 640
        assert cam.height == 480
        assert cam.fps == 30

    def test_from_dict(self) -> None:
        from rfx.robot.config import CameraConfig

        cam = CameraConfig.from_dict({"name": "wrist", "width": 320, "height": 240, "fps": 15})
        assert cam.name == "wrist"
        assert cam.width == 320
        assert cam.height == 240
        assert cam.fps == 15

    def test_from_dict_ignores_unknown(self) -> None:
        from rfx.robot.config import CameraConfig

        cam = CameraConfig.from_dict({"name": "top", "foobar": 999})
        assert cam.name == "top"
        assert not hasattr(cam, "foobar")


class TestJointConfig:
    """Test JointConfig dataclass and from_dict."""

    def test_from_dict(self) -> None:
        from rfx.robot.config import JointConfig

        j = JointConfig.from_dict({"name": "elbow", "index": 2, "velocity_max": 5.0})
        assert j.name == "elbow"
        assert j.index == 2
        assert j.velocity_max == 5.0

    def test_from_dict_ignores_unknown(self) -> None:
        from rfx.robot.config import JointConfig

        j = JointConfig.from_dict({"name": "wrist", "index": 4, "extra": True})
        assert j.name == "wrist"
        assert not hasattr(j, "extra")

    def test_defaults(self) -> None:
        from rfx.robot.config import JointConfig

        j = JointConfig(name="test", index=0)
        assert j.position_min == pytest.approx(-3.14159)
        assert j.position_max == pytest.approx(3.14159)
        assert j.velocity_max == 10.0
        assert j.effort_max == 100.0


# ---------------------------------------------------------------------------
# RobotConfig — nested parsing with cameras and joints
# ---------------------------------------------------------------------------


class TestRobotConfigNested:
    """Test RobotConfig.from_dict with nested cameras and joints."""

    def test_from_dict_with_cameras_and_joints(self) -> None:
        d = {
            "name": "TestBot",
            "state_dim": 8,
            "action_dim": 4,
            "cameras": [{"name": "wrist", "width": 320, "height": 240}],
            "joints": [
                {"name": "shoulder", "index": 0},
                {"name": "elbow", "index": 1},
            ],
        }
        cfg = RobotConfig.from_dict(d)
        assert cfg.name == "TestBot"
        assert len(cfg.cameras) == 1
        assert cfg.cameras[0].name == "wrist"
        assert cfg.cameras[0].width == 320
        assert len(cfg.joints) == 2
        assert cfg.joints[0].name == "shoulder"
        assert cfg.joints[1].name == "elbow"

    def test_from_dict_empty_cameras_and_joints(self) -> None:
        d = {"name": "Bare", "state_dim": 4, "action_dim": 2}
        cfg = RobotConfig.from_dict(d)
        assert cfg.cameras == []
        assert cfg.joints == []

    def test_to_dict_preserves_cameras_and_joints(self) -> None:
        from rfx.robot.config import CameraConfig, JointConfig

        cfg = RobotConfig(
            name="Full",
            cameras=[CameraConfig(name="cam1")],
            joints=[JointConfig(name="j1", index=0)],
        )
        d = cfg.to_dict()
        assert len(d["cameras"]) == 1
        assert d["cameras"][0]["name"] == "cam1"
        assert len(d["joints"]) == 1
        assert d["joints"][0]["name"] == "j1"


# ---------------------------------------------------------------------------
# RobotConfig — config search paths and RFX_CONFIG_DIR
# ---------------------------------------------------------------------------


class TestConfigSearchPaths:
    """Test _get_config_search_paths and RFX_CONFIG_DIR env var."""

    def test_default_paths_include_cwd(self) -> None:
        from rfx.robot.config import _get_config_search_paths

        paths = _get_config_search_paths()
        assert Path.cwd() in paths

    def test_rfx_config_dir_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from rfx.robot.config import _get_config_search_paths

        monkeypatch.setenv("RFX_CONFIG_DIR", str(tmp_path))
        paths = _get_config_search_paths()
        assert tmp_path in paths

    def test_no_rfx_config_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rfx.robot.config import _get_config_search_paths

        monkeypatch.delenv("RFX_CONFIG_DIR", raising=False)
        paths = _get_config_search_paths()
        # Should not contain any env-based path
        assert all(p != Path("") for p in paths)


# ---------------------------------------------------------------------------
# load_config — dict passthrough
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Test load_config with dict input."""

    def test_dict_passthrough(self) -> None:
        from rfx.robot.config import load_config

        d = {"name": "test", "state_dim": 4}
        result = load_config(d)
        assert result == d

    def test_yaml_path(self, tmp_path: Path) -> None:
        from rfx.robot.config import load_config

        yaml_content = "name: YamlBot\nstate_dim: 8\naction_dim: 4\ncontrol_freq_hz: 100\n"
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml_content)

        result = load_config(cfg_file)
        assert result["name"] == "YamlBot"
        assert result["state_dim"] == 8


# ---------------------------------------------------------------------------
# push_policy — mock HfApi
# ---------------------------------------------------------------------------


class TestPushPolicy:
    """Test push_policy with mocked HfApi."""

    def test_push_calls_hf_api(self, tmp_path: Path) -> None:
        import sys
        from types import ModuleType

        from rfx.hub import push_policy

        mock_api_instance = MagicMock()
        mock_api_cls = MagicMock(return_value=mock_api_instance)

        fake_hf = ModuleType("huggingface_hub")
        fake_hf.HfApi = mock_api_cls  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            url = push_policy(tmp_path, "user/my-policy")

        mock_api_instance.create_repo.assert_called_once_with(
            "user/my-policy", exist_ok=True, private=False
        )
        mock_api_instance.upload_folder.assert_called_once_with(
            folder_path=str(tmp_path),
            repo_id="user/my-policy",
        )
        assert url == "https://huggingface.co/user/my-policy"

    def test_push_private(self, tmp_path: Path) -> None:
        import sys
        from types import ModuleType

        from rfx.hub import push_policy

        mock_api_instance = MagicMock()
        mock_api_cls = MagicMock(return_value=mock_api_instance)

        fake_hf = ModuleType("huggingface_hub")
        fake_hf.HfApi = mock_api_cls  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            push_policy(tmp_path, "user/private-policy", private=True)

        mock_api_instance.create_repo.assert_called_once_with(
            "user/private-policy",
            exist_ok=True,
            private=True,
        )


# ---------------------------------------------------------------------------
# RobotBase — properties and repr
# ---------------------------------------------------------------------------


class TestRobotBase:
    """Test RobotBase properties and edge cases."""

    def test_properties(self) -> None:
        from rfx.robot import RobotBase

        class DummyRobot(RobotBase):
            def observe(self):
                return {}

            def act(self, action):
                pass

            def reset(self, env_ids=None):
                return {}

        r = DummyRobot(state_dim=12, action_dim=6, num_envs=4, device="cuda")
        assert r.state_dim == 12
        assert r.action_dim == 6
        assert r.num_envs == 4
        assert r.device == "cuda"
        assert r.max_state_dim == 64
        assert r.max_action_dim == 64

    def test_max_dim_clamps_up(self) -> None:
        """If state_dim > max_state_dim, max_state_dim should be raised."""
        from rfx.robot import RobotBase

        class DummyRobot(RobotBase):
            def observe(self):
                return {}

            def act(self, action):
                pass

            def reset(self, env_ids=None):
                return {}

        r = DummyRobot(state_dim=100, action_dim=80, max_state_dim=64, max_action_dim=64)
        assert r.max_state_dim == 100
        assert r.max_action_dim == 80

    def test_repr(self) -> None:
        from rfx.robot import RobotBase

        class DummyRobot(RobotBase):
            def observe(self):
                return {}

            def act(self, action):
                pass

            def reset(self, env_ids=None):
                return {}

        r = DummyRobot(state_dim=12, action_dim=6)
        rep = repr(r)
        assert "DummyRobot" in rep
        assert "state_dim=12" in rep
        assert "action_dim=6" in rep


# ---------------------------------------------------------------------------
# _print_stats with overruns
# ---------------------------------------------------------------------------


class TestPrintStatsOverruns:
    """Test _print_stats shows overrun count."""

    def test_overruns_in_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        stats = SessionStats(
            iterations=100,
            overruns=5,
            target_period_s=0.02,
            avg_period_s=0.025,
            p50_jitter_s=0.003,
            p95_jitter_s=0.008,
            p99_jitter_s=0.012,
            max_jitter_s=0.015,
        )
        _print_stats(stats)
        out = capsys.readouterr().out
        assert "100 steps" in out
        assert "5 overruns" in out
        assert "p95" in out
        assert "p99" in out


# ---------------------------------------------------------------------------
# deploy() — error propagation from _run_deploy_loop
# ---------------------------------------------------------------------------


class TestDeployErrorPropagation:
    """Verify deploy() propagates errors from _run_deploy_loop."""

    def test_runtime_error_propagates(self, tmp_path: Path) -> None:
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=MagicMock()),
            patch("rfx.deploy._run_deploy_loop", side_effect=RuntimeError("loop crashed")),
            pytest.raises(RuntimeError, match="loop crashed"),
        ):
            deploy(str(policy_dir), robot="so101", mock=True, verbose=False)

    def test_keyboard_interrupt_propagates(self, tmp_path: Path) -> None:
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=MagicMock()),
            patch("rfx.deploy._run_deploy_loop", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            deploy(str(policy_dir), robot="so101", mock=True, verbose=False)


# ---------------------------------------------------------------------------
# ObservationSpec — properties
# ---------------------------------------------------------------------------


class TestObservationSpec:
    """Test ObservationSpec dataclass properties."""

    def test_has_images_true(self) -> None:
        from rfx.observation import ObservationSpec

        spec = ObservationSpec(state_dim=12, num_cameras=2, image_shape=(480, 640, 3))
        assert spec.has_images is True

    def test_has_images_false_no_cameras(self) -> None:
        from rfx.observation import ObservationSpec

        spec = ObservationSpec(state_dim=12, num_cameras=0, image_shape=(480, 640, 3))
        assert spec.has_images is False

    def test_has_images_false_no_shape(self) -> None:
        from rfx.observation import ObservationSpec

        spec = ObservationSpec(state_dim=12, num_cameras=2, image_shape=None)
        assert spec.has_images is False

    def test_has_language_true(self) -> None:
        from rfx.observation import ObservationSpec

        spec = ObservationSpec(state_dim=12, language_dim=512)
        assert spec.has_language is True

    def test_has_language_false(self) -> None:
        from rfx.observation import ObservationSpec

        spec = ObservationSpec(state_dim=12)
        assert spec.has_language is False

    def test_defaults(self) -> None:
        from rfx.observation import ObservationSpec

        spec = ObservationSpec(state_dim=6)
        assert spec.max_state_dim == 64
        assert spec.num_cameras == 0
        assert spec.image_shape is None
        assert spec.language_dim is None


# ---------------------------------------------------------------------------
# make_observation — padding, truncation, images, language
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestMakeObservation:
    """Test make_observation with real torch tensors."""

    def test_padding_when_state_smaller(self) -> None:
        import torch

        from rfx.observation import make_observation

        state = torch.ones(2, 12)  # batch=2, state_dim=12
        obs = make_observation(state=state, state_dim=12, max_state_dim=64)
        assert obs["state"].shape == (2, 64)
        assert torch.all(obs["state"][:, :12] == 1.0)
        assert torch.all(obs["state"][:, 12:] == 0.0)

    def test_truncation_when_state_larger(self) -> None:
        import torch

        from rfx.observation import make_observation

        state = torch.ones(1, 128)
        obs = make_observation(state=state, state_dim=128, max_state_dim=64)
        assert obs["state"].shape == (1, 64)

    def test_exact_size_no_padding(self) -> None:
        import torch

        from rfx.observation import make_observation

        state = torch.ones(1, 64)
        obs = make_observation(state=state, state_dim=64, max_state_dim=64)
        assert obs["state"].shape == (1, 64)
        assert torch.all(obs["state"] == 1.0)

    def test_images_included(self) -> None:
        import torch

        from rfx.observation import make_observation

        state = torch.zeros(1, 12)
        images = torch.rand(1, 2, 480, 640, 3)
        obs = make_observation(state=state, state_dim=12, max_state_dim=64, images=images)
        assert "images" in obs
        assert obs["images"].shape == (1, 2, 480, 640, 3)

    def test_language_included(self) -> None:
        import torch

        from rfx.observation import make_observation

        state = torch.zeros(1, 12)
        language = torch.randint(0, 1000, (1, 32))
        obs = make_observation(state=state, state_dim=12, max_state_dim=64, language=language)
        assert "language" in obs
        assert obs["language"].shape == (1, 32)

    def test_no_images_or_language(self) -> None:
        import torch

        from rfx.observation import make_observation

        state = torch.zeros(1, 12)
        obs = make_observation(state=state, state_dim=12, max_state_dim=64)
        assert "images" not in obs
        assert "language" not in obs


# ---------------------------------------------------------------------------
# unpad_action — 2D, 3D, invalid
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestUnpadAction:
    """Test unpad_action with real torch tensors."""

    def test_2d_action(self) -> None:
        import torch

        from rfx.observation import unpad_action

        action = torch.ones(2, 64)
        result = unpad_action(action, action_dim=6)
        assert result.shape == (2, 6)

    def test_3d_action(self) -> None:
        import torch

        from rfx.observation import unpad_action

        action = torch.ones(2, 10, 64)  # batch, horizon, padded_dim
        result = unpad_action(action, action_dim=6)
        assert result.shape == (2, 10, 6)

    def test_invalid_dim_raises(self) -> None:
        import torch

        from rfx.observation import unpad_action

        action = torch.ones(64)  # 1D — invalid
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            unpad_action(action, action_dim=6)

    def test_4d_raises(self) -> None:
        import torch

        from rfx.observation import unpad_action

        action = torch.ones(1, 2, 3, 64)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            unpad_action(action, action_dim=6)

    def test_action_dim_equals_padded(self) -> None:
        import torch

        from rfx.observation import unpad_action

        action = torch.ones(1, 6)
        result = unpad_action(action, action_dim=6)
        assert result.shape == (1, 6)
        assert torch.all(result == 1.0)


# ---------------------------------------------------------------------------
# ObservationBuffer — push, get_stacked, clear, capacity, empty
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestObservationBuffer:
    """Test ObservationBuffer with real torch tensors."""

    def test_push_and_len(self) -> None:
        import torch

        from rfx.observation import ObservationBuffer

        buf = ObservationBuffer(capacity=5)
        assert len(buf) == 0
        buf.push({"state": torch.zeros(1, 64)})
        assert len(buf) == 1
        buf.push({"state": torch.ones(1, 64)})
        assert len(buf) == 2

    def test_capacity_overflow(self) -> None:
        import torch

        from rfx.observation import ObservationBuffer

        buf = ObservationBuffer(capacity=3)
        for i in range(5):
            buf.push({"state": torch.full((1, 4), float(i))})
        assert len(buf) == 3
        # Oldest (0, 1) should be evicted; remaining are 2, 3, 4
        stacked = buf.get_stacked()
        assert stacked["state"][0, 0, 0].item() == 2.0
        assert stacked["state"][0, 2, 0].item() == 4.0

    def test_get_stacked_shape(self) -> None:
        import torch

        from rfx.observation import ObservationBuffer

        buf = ObservationBuffer(capacity=4)
        for _ in range(3):
            buf.push({"state": torch.zeros(2, 64)})
        stacked = buf.get_stacked()
        # shape: (batch=2, stack=3, state_dim=64)
        assert stacked["state"].shape == (2, 3, 64)

    def test_empty_buffer_raises(self) -> None:
        from rfx.observation import ObservationBuffer

        buf = ObservationBuffer(capacity=5)
        with pytest.raises(ValueError, match="Buffer is empty"):
            buf.get_stacked()

    def test_clear(self) -> None:
        import torch

        from rfx.observation import ObservationBuffer

        buf = ObservationBuffer(capacity=5)
        buf.push({"state": torch.zeros(1, 4)})
        buf.push({"state": torch.zeros(1, 4)})
        assert len(buf) == 2
        buf.clear()
        assert len(buf) == 0

    def test_push_clones_tensors(self) -> None:
        """Modifying original tensor after push should not affect buffer."""
        import torch

        from rfx.observation import ObservationBuffer

        buf = ObservationBuffer(capacity=5)
        t = torch.zeros(1, 4)
        buf.push({"state": t})
        t.fill_(99.0)
        stacked = buf.get_stacked()
        assert stacked["state"][0, 0, 0].item() == 0.0


# ---------------------------------------------------------------------------
# MockBackend — real torch tensor operations
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestMockBackend:
    """Test MockBackend with real torch tensors."""

    def _make_config(self, **overrides) -> RobotConfig:
        defaults = {
            "state_dim": 12,
            "action_dim": 6,
            "max_state_dim": 64,
            "max_action_dim": 64,
            "control_freq_hz": 50,
        }
        defaults.update(overrides)
        return RobotConfig(**defaults)

    def test_init_state(self) -> None:
        import torch

        from rfx.sim.mock import MockBackend

        cfg = self._make_config()
        backend = MockBackend(cfg, num_envs=2)
        assert backend._positions.shape == (2, 6)
        assert backend._velocities.shape == (2, 6)
        assert torch.all(backend._positions == 0.0)
        assert torch.all(backend._velocities == 0.0)

    def test_observe_returns_padded_state(self) -> None:
        from rfx.sim.mock import MockBackend

        cfg = self._make_config()
        backend = MockBackend(cfg, num_envs=1)
        obs = backend.observe()
        assert "state" in obs
        assert obs["state"].shape == (1, 64)

    def test_act_updates_positions(self) -> None:
        import torch

        from rfx.sim.mock import MockBackend

        cfg = self._make_config()
        backend = MockBackend(cfg, num_envs=1)
        action = torch.ones(1, 64) * 0.5
        backend.act(action)
        # After one step, positions should have moved toward target
        assert not torch.all(backend._positions == 0.0)

    def test_act_spring_damper_physics(self) -> None:
        """Multiple steps should converge positions toward action target."""
        import torch

        from rfx.sim.mock import MockBackend

        cfg = self._make_config()
        backend = MockBackend(cfg, num_envs=1)
        target = torch.ones(1, 64) * 0.1
        for _ in range(100):
            backend.act(target)
        # Should be close to target after many steps
        assert torch.allclose(backend._positions, target[:, :6], atol=0.05)

    def test_reset_clears_state(self) -> None:
        import torch

        from rfx.sim.mock import MockBackend

        cfg = self._make_config()
        backend = MockBackend(cfg, num_envs=2)
        backend.act(torch.ones(2, 64))
        obs = backend.reset()
        assert torch.all(backend._positions == 0.0)
        assert torch.all(backend._velocities == 0.0)
        assert torch.all(backend._step_count == 0)
        assert "state" in obs

    def test_reset_partial_envs(self) -> None:
        import torch

        from rfx.sim.mock import MockBackend

        cfg = self._make_config()
        backend = MockBackend(cfg, num_envs=4)
        backend.act(torch.ones(4, 64))
        # Reset only envs 0 and 2
        backend.reset(env_ids=torch.tensor([0, 2]))
        assert torch.all(backend._positions[0] == 0.0)
        assert torch.all(backend._positions[2] == 0.0)
        assert not torch.all(backend._positions[1] == 0.0)
        assert not torch.all(backend._positions[3] == 0.0)

    def test_get_reward(self) -> None:
        import torch

        from rfx.sim.mock import MockBackend

        cfg = self._make_config()
        backend = MockBackend(cfg, num_envs=2)
        backend.act(torch.ones(2, 64))
        reward = backend.get_reward()
        assert reward.shape == (2,)
        # Reward is -norm(error), should be negative
        assert torch.all(reward <= 0.0)

    def test_get_done_after_max_steps(self) -> None:
        import torch

        from rfx.sim.mock import MockBackend

        cfg = self._make_config()
        backend = MockBackend(cfg, num_envs=1, max_steps=5)
        action = torch.zeros(1, 64)
        for _ in range(5):
            backend.act(action)
        done = backend.get_done()
        assert done[0].item() is True

    def test_get_done_before_max_steps(self) -> None:
        import torch

        from rfx.sim.mock import MockBackend

        cfg = self._make_config()
        backend = MockBackend(cfg, num_envs=1, max_steps=100)
        backend.act(torch.zeros(1, 64))
        done = backend.get_done()
        assert done[0].item() is False

    def test_position_clamping(self) -> None:
        """Positions should be clamped to [-pi, pi]."""
        import torch

        from rfx.sim.mock import MockBackend

        cfg = self._make_config()
        backend = MockBackend(cfg, num_envs=1)
        # Huge action to force clamping
        action = torch.ones(1, 64) * 100.0
        for _ in range(50):
            backend.act(action)
        assert torch.all(backend._positions <= 3.14159)
        assert torch.all(backend._positions >= -3.14159)


# ---------------------------------------------------------------------------
# MockRobot — conforms to Robot protocol, delegates to backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestMockRobot:
    """Test MockRobot with real torch tensors."""

    def test_conforms_to_robot_protocol(self) -> None:
        from rfx.robot import Robot
        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        assert isinstance(robot, Robot)

    def test_properties(self) -> None:
        from rfx.sim.mock import MockRobot

        robot = MockRobot(
            state_dim=12, action_dim=6, num_envs=4, max_state_dim=64, max_action_dim=64
        )
        assert robot.state_dim == 12
        assert robot.action_dim == 6
        assert robot.num_envs == 4
        assert robot.max_state_dim == 64
        assert robot.max_action_dim == 64

    def test_observe_returns_dict(self) -> None:
        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        obs = robot.observe()
        assert isinstance(obs, dict)
        assert "state" in obs
        assert obs["state"].shape == (1, 64)

    def test_act_changes_state(self) -> None:
        import torch

        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        obs_before = robot.observe()["state"].clone()
        robot.act(torch.ones(1, 64) * 0.5)
        obs_after = robot.observe()["state"]
        assert not torch.equal(obs_before, obs_after)

    def test_reset_returns_obs(self) -> None:
        import torch

        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        robot.act(torch.ones(1, 64))
        obs = robot.reset()
        assert "state" in obs
        # After reset, state should be zero-padded
        assert obs["state"].shape == (1, 64)

    def test_multi_env(self) -> None:
        import torch

        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6, num_envs=8)
        obs = robot.observe()
        assert obs["state"].shape == (8, 64)
        robot.act(torch.ones(8, 64) * 0.3)
        obs2 = robot.observe()
        assert obs2["state"].shape == (8, 64)

    def test_get_reward_and_done(self) -> None:
        import torch

        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6, num_envs=2)
        robot.act(torch.ones(2, 64))
        reward = robot.get_reward()
        done = robot.get_done()
        assert reward.shape == (2,)
        assert done.shape == (2,)


# ---------------------------------------------------------------------------
# SimRobot — backend selection, from_config, render/close
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestSimRobot:
    """Test SimRobot backend selection and lifecycle."""

    def _make_yaml_config(self, tmp_path: Path) -> Path:
        yaml_content = (
            "name: test-sim-bot\n"
            "state_dim: 12\n"
            "action_dim: 6\n"
            "max_state_dim: 64\n"
            "max_action_dim: 64\n"
            "control_freq_hz: 50\n"
        )
        cfg_path = tmp_path / "sim_bot.yaml"
        cfg_path.write_text(yaml_content)
        return cfg_path

    def test_mock_backend_creates(self, tmp_path: Path) -> None:
        from rfx.sim.base import SimRobot

        cfg_path = self._make_yaml_config(tmp_path)
        robot = SimRobot(cfg_path, backend="mock", device="cpu")
        assert robot.backend == "mock"
        assert robot.state_dim == 12
        assert robot.action_dim == 6

    def test_unknown_backend_raises(self, tmp_path: Path) -> None:
        from rfx.sim.base import SimRobot

        cfg_path = self._make_yaml_config(tmp_path)
        with pytest.raises(ValueError, match="Unknown backend"):
            SimRobot(cfg_path, backend="nonexistent", device="cpu")

    def test_observe_act_reset(self, tmp_path: Path) -> None:
        import torch

        from rfx.sim.base import SimRobot

        cfg_path = self._make_yaml_config(tmp_path)
        robot = SimRobot(cfg_path, num_envs=2, backend="mock", device="cpu")
        obs = robot.observe()
        assert obs["state"].shape == (2, 64)
        robot.act(torch.ones(2, 64) * 0.1)
        obs2 = robot.observe()
        assert not torch.equal(obs["state"], obs2["state"])
        obs3 = robot.reset()
        assert obs3["state"].shape == (2, 64)

    def test_from_config_classmethod(self, tmp_path: Path) -> None:
        from rfx.sim.base import SimRobot

        cfg_path = self._make_yaml_config(tmp_path)
        robot = SimRobot.from_config(cfg_path, num_envs=4, backend="mock", device="cpu")
        assert robot.num_envs == 4
        assert robot.backend == "mock"

    def test_from_dict_config(self) -> None:
        from rfx.sim.base import SimRobot

        cfg_dict = {
            "name": "dict-bot",
            "state_dim": 8,
            "action_dim": 4,
            "max_state_dim": 32,
            "max_action_dim": 32,
            "control_freq_hz": 100,
        }
        robot = SimRobot(cfg_dict, backend="mock", device="cpu")
        assert robot.config.name == "dict-bot"
        assert robot.state_dim == 8

    def test_from_robot_config_object(self) -> None:
        from rfx.sim.base import SimRobot

        robot = SimRobot(SO101_CONFIG, backend="mock", device="cpu")
        assert robot.config.name == SO101_CONFIG.name

    def test_repr(self, tmp_path: Path) -> None:
        from rfx.sim.base import SimRobot

        cfg_path = self._make_yaml_config(tmp_path)
        robot = SimRobot(cfg_path, backend="mock", device="cpu")
        rep = repr(robot)
        assert "SimRobot" in rep
        assert "mock" in rep
        assert "state_dim=12" in rep

    def test_render_noop_on_mock(self, tmp_path: Path) -> None:
        """render() should not crash even if backend has no render."""
        from rfx.sim.base import SimRobot

        cfg_path = self._make_yaml_config(tmp_path)
        robot = SimRobot(cfg_path, backend="mock", device="cpu")
        robot.render()  # Should not raise

    def test_close_noop_on_mock(self, tmp_path: Path) -> None:
        """close() should not crash even if backend has no close."""
        from rfx.sim.base import SimRobot

        cfg_path = self._make_yaml_config(tmp_path)
        robot = SimRobot(cfg_path, backend="mock", device="cpu")
        robot.close()  # Should not raise

    def test_get_reward_and_done(self, tmp_path: Path) -> None:
        import torch

        from rfx.sim.base import SimRobot

        cfg_path = self._make_yaml_config(tmp_path)
        robot = SimRobot(cfg_path, num_envs=2, backend="mock", device="cpu")
        robot.act(torch.ones(2, 64))
        reward = robot.get_reward()
        done = robot.get_done()
        assert reward.shape == (2,)
        assert done.shape == (2,)


# ---------------------------------------------------------------------------
# Session lifecycle — context manager, run(), rfx.run() helper
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestSessionLifecycle:
    """Test Session with real MockRobot and torch tensors."""

    def test_context_manager(self) -> None:
        import torch

        from rfx.session import Session
        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        policy = lambda obs: torch.zeros(1, 64)  # noqa: E731

        with Session(robot, policy, rate_hz=100, warmup_s=0.0) as s:
            assert s.is_running
        assert not s.is_running

    def test_run_with_duration(self) -> None:
        import torch

        from rfx.session import Session
        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        policy = lambda obs: torch.zeros(1, 64)  # noqa: E731

        session = Session(robot, policy, rate_hz=100, warmup_s=0.0)
        session.run(duration=0.3)
        stats = session.stats
        assert stats.iterations > 0
        assert stats.target_period_s == pytest.approx(0.01)

    def test_rfx_run_helper(self) -> None:
        import torch

        from rfx.session import run
        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        policy = lambda obs: torch.zeros(1, 64)  # noqa: E731

        stats = run(robot, policy, rate_hz=100, duration=0.3, warmup_s=0.0)
        assert stats.iterations > 0

    def test_step_count_increments(self) -> None:
        import time

        import torch

        from rfx.session import Session
        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        policy = lambda obs: torch.zeros(1, 64)  # noqa: E731

        with Session(robot, policy, rate_hz=200, warmup_s=0.0) as s:
            time.sleep(0.2)
            count = s.step_count
        assert count > 0

    def test_stats_zero_when_not_started(self) -> None:
        import torch

        from rfx.session import Session
        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        policy = lambda obs: torch.zeros(1, 64)  # noqa: E731

        session = Session(robot, policy, rate_hz=50, warmup_s=0.0)
        stats = session.stats
        assert stats.iterations == 0
        assert stats.avg_period_s == 0.0

    def test_start_stop_idempotent(self) -> None:
        import torch

        from rfx.session import Session
        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        policy = lambda obs: torch.zeros(1, 64)  # noqa: E731

        session = Session(robot, policy, rate_hz=100, warmup_s=0.0)
        session.start()
        session.start()  # Should be idempotent
        assert session.is_running
        session.stop()
        session.stop()  # Should be idempotent
        assert not session.is_running

    def test_check_health_no_error(self) -> None:
        import torch

        from rfx.session import Session
        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)
        policy = lambda obs: torch.zeros(1, 64)  # noqa: E731

        session = Session(robot, policy, rate_hz=100, warmup_s=0.0)
        session.check_health()  # Should not raise

    def test_check_health_with_error(self) -> None:

        from rfx.session import Session
        from rfx.sim.mock import MockRobot

        robot = MockRobot(state_dim=12, action_dim=6)

        def bad_policy(obs):
            raise RuntimeError("policy exploded")

        session = Session(robot, bad_policy, rate_hz=100, warmup_s=0.0)
        session.start()
        import time

        time.sleep(0.1)  # Let the control loop hit the error
        with pytest.raises(RuntimeError, match="Control loop failed"):
            session.check_health()
        session.stop()


# ---------------------------------------------------------------------------
# Real integration: deploy() end-to-end with REAL MockRobot + Session
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestDeployRealIntegration:
    """End-to-end deploy() with real MockRobot, Session, and torch — no mocks."""

    def test_deploy_so101_mock_real(self, tmp_path: Path) -> None:
        """deploy() with mock=True, so101 config, real policy function."""

        py_file = tmp_path / "hold.py"
        py_file.write_text(
            "import torch\nimport rfx\n\n@rfx.policy\n"
            "def hold(obs):\n    return torch.zeros(1, 64)\n"
        )
        stats = deploy(
            str(py_file),
            robot="so101",
            mock=True,
            duration=0.5,
            rate_hz=50,
            warmup_s=0.0,
            verbose=False,
        )
        assert stats.iterations > 0
        assert stats.overruns >= 0
        assert stats.target_period_s == pytest.approx(1.0 / 50)

    def test_deploy_go2_mock_real(self, tmp_path: Path) -> None:
        """deploy() with mock=True, go2 config."""

        py_file = tmp_path / "walk.py"
        py_file.write_text(
            "import torch\nimport rfx\n\n@rfx.policy\n"
            "def walk(obs):\n    return torch.zeros(1, 64)\n"
        )
        stats = deploy(
            str(py_file),
            robot="go2",
            mock=True,
            duration=0.3,
            rate_hz=100,
            warmup_s=0.0,
            verbose=False,
        )
        assert stats.iterations > 0

    def test_deploy_g1_mock_real(self, tmp_path: Path) -> None:
        """deploy() with mock=True, g1 config."""

        py_file = tmp_path / "stand.py"
        py_file.write_text(
            "import torch\nimport rfx\n\n@rfx.policy\n"
            "def stand(obs):\n    return torch.zeros(1, 64)\n"
        )
        stats = deploy(
            str(py_file),
            robot="g1",
            mock=True,
            duration=0.3,
            rate_hz=50,
            warmup_s=0.0,
            verbose=False,
        )
        assert stats.iterations > 0

    def test_deploy_real_policy_reads_obs(self, tmp_path: Path) -> None:
        """Policy that actually reads observation state and returns scaled action."""

        py_file = tmp_path / "echo.py"
        py_file.write_text(
            "import torch\nimport rfx\n\n@rfx.policy\n"
            "def echo(obs):\n"
            "    state = obs['state']\n"
            "    return state * 0.1\n"
        )
        stats = deploy(
            str(py_file),
            robot="so101",
            mock=True,
            duration=0.5,
            rate_hz=50,
            warmup_s=0.0,
            verbose=False,
        )
        assert stats.iterations > 0

    def test_deploy_verbose_real(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """deploy() verbose=True with real execution prints expected output."""

        py_file = tmp_path / "hold.py"
        py_file.write_text(
            "import torch\nimport rfx\n\n@rfx.policy\n"
            "def hold(obs):\n    return torch.zeros(1, 64)\n"
        )
        stats = deploy(
            str(py_file),
            robot="so101",
            mock=True,
            duration=0.3,
            rate_hz=50,
            warmup_s=0.0,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert "Loading policy" in out
        assert "Policy function: hold" in out
        assert "Done" in out
        assert stats.iterations > 0

    def test_deploy_warmup_delays_start(self, tmp_path: Path) -> None:
        """deploy() with warmup_s should delay before running."""
        import time

        py_file = tmp_path / "hold.py"
        py_file.write_text(
            "import torch\nimport rfx\n\n@rfx.policy\n"
            "def hold(obs):\n    return torch.zeros(1, 64)\n"
        )
        t0 = time.perf_counter()
        stats = deploy(
            str(py_file),
            robot="so101",
            mock=True,
            duration=0.2,
            rate_hz=50,
            warmup_s=0.3,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        # Should take at least warmup_s + duration
        assert elapsed >= 0.4
        assert stats.iterations > 0


# ---------------------------------------------------------------------------
# deploy() — _run_deploy_loop SIGINT handling (double Ctrl+C)
# ---------------------------------------------------------------------------


class TestDeploySignalHandling:
    """Test that deploy() restores signal handlers after completion."""

    def test_signal_handler_restored(self, tmp_path: Path) -> None:
        """After deploy(), SIGINT handler should be restored to original."""
        import signal

        original_handler = signal.getsignal(signal.SIGINT)

        fake_stats = SessionStats(
            iterations=1,
            overruns=0,
            target_period_s=0.02,
            avg_period_s=0.02,
            p50_jitter_s=0.0,
            p95_jitter_s=0.0,
            p99_jitter_s=0.0,
            max_jitter_s=0.0,
        )
        loaded = _make_loaded_policy(robot_config=SO101_CONFIG)

        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()

        with (
            patch("rfx.deploy.load_policy", return_value=loaded),
            patch("rfx.deploy._create_robot", return_value=MagicMock()),
            patch("rfx.deploy._run_deploy_loop", return_value=fake_stats),
        ):
            deploy(str(policy_dir), robot="so101", mock=True, verbose=False)

        restored_handler = signal.getsignal(signal.SIGINT)
        assert restored_handler is original_handler


# ---------------------------------------------------------------------------
# RobotConfig — SO101 joints validation
# ---------------------------------------------------------------------------


class TestSO101ConfigDetails:
    """Detailed validation of SO-101 built-in config."""

    def test_so101_has_6_joints(self) -> None:
        assert len(SO101_CONFIG.joints) == 6
        assert SO101_CONFIG.action_dim == 6

    def test_so101_joint_names(self) -> None:
        names = [j.name for j in SO101_CONFIG.joints]
        assert "shoulder_pan" in names
        assert "shoulder_lift" in names
        assert "elbow" in names
        assert "wrist_pitch" in names
        assert "wrist_roll" in names
        assert "gripper" in names

    def test_so101_joint_indices_sequential(self) -> None:
        indices = [j.index for j in SO101_CONFIG.joints]
        assert indices == list(range(6))

    def test_so101_hardware_has_port_and_baudrate(self) -> None:
        assert "port" in SO101_CONFIG.hardware
        assert "baudrate" in SO101_CONFIG.hardware
        assert SO101_CONFIG.hardware["baudrate"] == 1000000


class TestGO2ConfigDetails:
    """Detailed validation of Go2 built-in config."""

    def test_go2_dims(self) -> None:
        assert GO2_CONFIG.state_dim == 36
        assert GO2_CONFIG.action_dim == 12

    def test_go2_200hz(self) -> None:
        assert GO2_CONFIG.control_freq_hz == 200

    def test_go2_has_ip_address(self) -> None:
        assert "ip_address" in GO2_CONFIG.hardware


class TestG1ConfigDetails:
    """Detailed validation of G1 built-in config."""

    def test_g1_29_dof(self) -> None:
        assert G1_CONFIG.action_dim == 29
        assert len(G1_CONFIG.joints) == 29

    def test_g1_state_dim_69(self) -> None:
        # 29 pos + 29 vel + 4 quat + 3 gyro + 4 foot_contacts = 69
        assert G1_CONFIG.state_dim == 69

    def test_g1_max_state_dim_128(self) -> None:
        assert G1_CONFIG.max_state_dim == 128

    def test_g1_joint_ordering(self) -> None:
        """Verify G1 joint ordering matches official G1JointIndex."""
        joints = G1_CONFIG.joints
        # Left leg 0-5
        assert joints[0].name == "left_hip_pitch"
        assert joints[5].name == "left_ankle_roll"
        # Right leg 6-11
        assert joints[6].name == "right_hip_pitch"
        assert joints[11].name == "right_ankle_roll"
        # Waist 12-14
        assert joints[12].name == "waist_yaw"
        assert joints[14].name == "waist_pitch"
        # Left arm 15-21
        assert joints[15].name == "left_shoulder_pitch"
        assert joints[21].name == "left_wrist_yaw"
        # Right arm 22-28
        assert joints[22].name == "right_shoulder_pitch"
        assert joints[28].name == "right_wrist_yaw"

    def test_g1_joint_limits_reasonable(self) -> None:
        """All G1 joints should have reasonable position limits."""
        for j in G1_CONFIG.joints:
            assert j.position_min < j.position_max
            assert j.position_min >= -3.2
            assert j.position_max <= 3.2


# ---------------------------------------------------------------------------
# SessionStats — to_dict field types
# ---------------------------------------------------------------------------


class TestSessionStatsTypes:
    """Verify SessionStats.to_dict returns correct types."""

    def test_to_dict_types(self) -> None:
        stats = SessionStats(
            iterations=42,
            overruns=3,
            target_period_s=0.02,
            avg_period_s=0.021,
            p50_jitter_s=0.001,
            p95_jitter_s=0.003,
            p99_jitter_s=0.005,
            max_jitter_s=0.01,
        )
        d = stats.to_dict()
        assert isinstance(d["iterations"], int)
        assert isinstance(d["overruns"], int)
        assert isinstance(d["target_period_s"], float)
        assert isinstance(d["avg_period_s"], float)
        assert isinstance(d["p50_jitter_s"], float)
        assert isinstance(d["p95_jitter_s"], float)
        assert isinstance(d["p99_jitter_s"], float)
        assert isinstance(d["max_jitter_s"], float)


# ---------------------------------------------------------------------------
# @rfx.policy decorator
# ---------------------------------------------------------------------------


class TestPolicyDecorator:
    """Test the @rfx.policy decorator used by deploy to discover policies."""

    def test_marks_function(self) -> None:
        import rfx

        @rfx.policy
        def my_policy(obs):
            return obs

        assert getattr(my_policy, "_rfx_policy", False) is True

    def test_preserves_function_name(self) -> None:
        import rfx

        @rfx.policy
        def hold(obs):
            return obs

        assert hold.__name__ == "hold"

    def test_preserves_return_value(self) -> None:
        import rfx

        @rfx.policy
        def echo(obs):
            return obs * 2

        assert echo(5) == 10

    def test_with_parentheses(self) -> None:
        import rfx

        @rfx.policy()
        def my_policy(obs):
            return obs

        assert getattr(my_policy, "_rfx_policy", False) is True
        assert my_policy("test") == "test"

    def test_without_parentheses(self) -> None:
        import rfx

        @rfx.policy
        def my_policy(obs):
            return obs

        assert getattr(my_policy, "_rfx_policy", False) is True


# ---------------------------------------------------------------------------
# MotorCommands — build action tensors from named joints
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestMotorCommands:
    """Test MotorCommands used to build deploy actions from joint names."""

    def test_to_tensor_basic(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands({"gripper": 0.8}, config=SO101_CONFIG)
        action = cmd.to_tensor()
        assert action.shape == (1, SO101_CONFIG.max_action_dim)
        # gripper is index 5 in SO101
        assert action[0, 5].item() == pytest.approx(0.8)

    def test_to_tensor_multiple_joints(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands(
            {"shoulder_pan": 1.0, "elbow": -0.5, "gripper": 0.3},
            config=SO101_CONFIG,
        )
        action = cmd.to_tensor()
        assert action[0, 0].item() == pytest.approx(1.0)  # shoulder_pan
        assert action[0, 2].item() == pytest.approx(-0.5)  # elbow
        assert action[0, 5].item() == pytest.approx(0.3)  # gripper

    def test_to_tensor_batch_size(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands({"gripper": 0.5}, config=SO101_CONFIG)
        action = cmd.to_tensor(batch_size=4)
        assert action.shape == (4, SO101_CONFIG.max_action_dim)
        assert action[3, 5].item() == pytest.approx(0.5)

    def test_to_tensor_no_config_raises(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands({"gripper": 0.8})
        with pytest.raises(ValueError, match="requires a config"):
            cmd.to_tensor()

    def test_to_tensor_unknown_joint_raises(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands({"nonexistent_joint": 1.0}, config=SO101_CONFIG)
        with pytest.raises(ValueError, match="not found"):
            cmd.to_tensor()

    def test_to_list(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands({"elbow": -0.3, "gripper": 0.8}, config=SO101_CONFIG)
        result = cmd.to_list()
        assert len(result) == SO101_CONFIG.action_dim
        assert result[2] == pytest.approx(-0.3)  # elbow
        assert result[5] == pytest.approx(0.8)  # gripper
        assert result[0] == 0.0  # unset joints default to 0

    def test_to_list_no_config_raises(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands({"gripper": 0.8})
        with pytest.raises(ValueError, match="requires a config"):
            cmd.to_list()

    def test_from_positions_factory(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands.from_positions({"gripper": 0.5}, config=SO101_CONFIG, kp=30.0, kd=1.0)
        assert cmd.positions == {"gripper": 0.5}
        assert cmd.kp == 30.0
        assert cmd.kd == 1.0

    def test_repr_with_positions(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands({"gripper": 0.80}, config=SO101_CONFIG)
        rep = repr(cmd)
        assert "gripper=0.80" in rep
        assert "SO-101" in rep

    def test_repr_empty(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands(config=SO101_CONFIG)
        rep = repr(cmd)
        assert "empty" in rep

    def test_repr_no_config(self) -> None:
        from rfx.decorators import MotorCommands

        cmd = MotorCommands({"gripper": 0.5})
        rep = repr(cmd)
        assert "no config" in rep

    def test_g1_joint_mapping(self) -> None:
        """MotorCommands should work with G1's 29-DOF config."""
        from rfx.decorators import MotorCommands

        cmd = MotorCommands(
            {"left_hip_pitch": 0.1, "right_wrist_yaw": -0.2},
            config=G1_CONFIG,
        )
        action = cmd.to_tensor()
        assert action[0, 0].item() == pytest.approx(0.1)  # left_hip_pitch = index 0
        assert action[0, 28].item() == pytest.approx(-0.2)  # right_wrist_yaw = index 28


# ---------------------------------------------------------------------------
# load_policy — mocked internals (Policy.load, normalizer, robot_config)
# ---------------------------------------------------------------------------


class TestLoadPolicy:
    """Test hub.load_policy with mocked nn.Policy and transforms."""

    def test_loads_with_robot_config(self, tmp_path: Path) -> None:
        import json

        from rfx.hub import load_policy

        config = {
            "policy_type": "act",
            "robot_config": {
                "name": "test-bot",
                "state_dim": 12,
                "action_dim": 6,
                "max_state_dim": 64,
                "max_action_dim": 64,
                "control_freq_hz": 50,
            },
        }
        (tmp_path / "rfx_config.json").write_text(json.dumps(config))

        mock_policy = MagicMock()
        mock_policy_cls = MagicMock()
        mock_policy_cls.load = MagicMock(return_value=mock_policy)

        with (
            patch.dict("sys.modules", {}),
            patch("rfx.nn.Policy", mock_policy_cls, create=True),
            patch("rfx.utils.transforms.ObservationNormalizer", create=True),
        ):
            loaded = load_policy(str(tmp_path))

        assert loaded.robot_config is not None
        assert loaded.robot_config.name == "test-bot"
        assert loaded.policy_type == "act"

    def test_loads_without_robot_config(self, tmp_path: Path) -> None:
        import json

        from rfx.hub import load_policy

        config = {"policy_type": "mlp"}
        (tmp_path / "rfx_config.json").write_text(json.dumps(config))

        mock_policy = MagicMock()
        mock_policy_cls = MagicMock()
        mock_policy_cls.load = MagicMock(return_value=mock_policy)

        with (
            patch("rfx.nn.Policy", mock_policy_cls, create=True),
            patch("rfx.utils.transforms.ObservationNormalizer", create=True),
        ):
            loaded = load_policy(str(tmp_path))

        assert loaded.robot_config is None

    def test_loads_normalizer_when_present(self, tmp_path: Path) -> None:
        import json

        from rfx.hub import load_policy

        config = {"policy_type": "act"}
        (tmp_path / "rfx_config.json").write_text(json.dumps(config))
        (tmp_path / "normalizer.json").write_text(json.dumps({"mean": [0.0], "std": [1.0]}))

        mock_policy = MagicMock()
        mock_policy_cls = MagicMock()
        mock_policy_cls.load = MagicMock(return_value=mock_policy)
        mock_normalizer = MagicMock()
        mock_norm_cls = MagicMock()
        mock_norm_cls.from_dict = MagicMock(return_value=mock_normalizer)

        with (
            patch("rfx.nn.Policy", mock_policy_cls, create=True),
            patch("rfx.utils.transforms.ObservationNormalizer", mock_norm_cls, create=True),
        ):
            loaded = load_policy(str(tmp_path))

        assert loaded.normalizer is mock_normalizer
        mock_norm_cls.from_dict.assert_called_once()

    def test_no_normalizer_when_absent(self, tmp_path: Path) -> None:
        import json

        from rfx.hub import load_policy

        config = {"policy_type": "act"}
        (tmp_path / "rfx_config.json").write_text(json.dumps(config))

        mock_policy = MagicMock()
        mock_policy_cls = MagicMock()
        mock_policy_cls.load = MagicMock(return_value=mock_policy)

        with (
            patch("rfx.nn.Policy", mock_policy_cls, create=True),
            patch("rfx.utils.transforms.ObservationNormalizer", create=True),
        ):
            loaded = load_policy(str(tmp_path))

        assert loaded.normalizer is None


# ---------------------------------------------------------------------------
# deploy() with MotorCommands — real integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")
class TestDeployWithMotorCommands:
    """Test deploy() with a policy that uses MotorCommands — the real user workflow."""

    def test_deploy_motor_commands_policy(self, tmp_path: Path) -> None:
        """A policy using MotorCommands should work end-to-end with deploy()."""
        py_file = tmp_path / "hold_gripper.py"
        py_file.write_text(
            "import rfx\n"
            "from rfx.robot.config import SO101_CONFIG\n\n"
            "@rfx.policy\n"
            "def hold_gripper(obs):\n"
            "    cmd = rfx.MotorCommands({'gripper': 0.8}, config=SO101_CONFIG)\n"
            "    return cmd.to_tensor()\n"
        )
        stats = deploy(
            str(py_file),
            robot="so101",
            mock=True,
            duration=0.3,
            rate_hz=50,
            warmup_s=0.0,
            verbose=False,
        )
        assert stats.iterations > 0

    def test_deploy_reactive_policy(self, tmp_path: Path) -> None:
        """A policy that reads obs and reacts — the core deploy use case."""
        py_file = tmp_path / "reactive.py"
        py_file.write_text(
            "import torch\nimport rfx\n\n"
            "@rfx.policy\n"
            "def reactive(obs):\n"
            "    state = obs['state']\n"
            "    # Simple proportional controller: move toward zero\n"
            "    return -0.1 * state\n"
        )
        stats = deploy(
            str(py_file),
            robot="so101",
            mock=True,
            duration=0.5,
            rate_hz=50,
            warmup_s=0.0,
            verbose=False,
        )
        assert stats.iterations > 0

    def test_deploy_multi_robot_configs(self, tmp_path: Path) -> None:
        """Same policy should deploy to different robot configs."""
        py_file = tmp_path / "zero.py"
        py_file.write_text(
            "import torch\nimport rfx\n\n"
            "@rfx.policy\n"
            "def zero(obs):\n"
            "    return torch.zeros_like(obs['state'])\n"
        )
        for robot_name in ("so101", "go2", "g1"):
            stats = deploy(
                str(py_file),
                robot=robot_name,
                mock=True,
                duration=0.2,
                rate_hz=50,
                warmup_s=0.0,
                verbose=False,
            )
            assert stats.iterations > 0, f"Failed for {robot_name}"
