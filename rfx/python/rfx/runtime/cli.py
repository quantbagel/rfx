"""
rfx CLI - Three commands. That's it.

    rfx record   - Record demonstrations from hardware
    rfx train    - Train a policy from recorded data
    rfx deploy   - Deploy a trained policy to a robot

    rfx doctor   - Check your setup
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------


def cmd_deploy(args: argparse.Namespace) -> int:
    """Deploy a trained policy to a robot."""
    from rfx.deploy import deploy

    try:
        deploy(
            policy_source=args.policy,
            robot=args.robot,
            config=args.config,
            port=args.port,
            rate_hz=args.rate_hz,
            duration=args.duration,
            mock=args.mock,
            warmup_s=args.warmup,
            verbose=True,
        )
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as exc:
        print(f"[rfx] Error: {exc}")
        return 1
    except ValueError as exc:
        print(f"[rfx] Error: {exc}")
        return 1
    except Exception as exc:
        print(f"[rfx] Deploy failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def cmd_record(args: argparse.Namespace) -> int:
    """Record demonstrations from a robot."""
    from rfx.collection._cli import run_collection

    try:
        result = run_collection(args)
        print(f"[rfx] Recorded {result['episodes']} episodes, {result['total_frames']} frames")
        print(f"[rfx] Saved to {result['root']}")
    except KeyboardInterrupt:
        print("\n[rfx] Recording stopped.")
    except Exception as exc:
        print(f"[rfx] Record failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    """Train a policy from collected data."""
    from rfx.workflow.registry import (
        create_run_record,
        generate_run_id,
        materialize_refs,
        snapshot_config,
    )
    from rfx.workflow.stages import execute_stage

    run_id = generate_run_id("train")
    config_data = snapshot_config(args.config)
    input_refs = materialize_refs(list(args.input))
    output_refs = materialize_refs(list(args.output))

    metadata: dict[str, Any] = {}
    if args.data:
        input_refs = materialize_refs([args.data]) + input_refs
        metadata["data"] = args.data

    try:
        result = execute_stage(
            stage="train",
            run_id=run_id,
            root=Path.cwd(),
            config_snapshot_data=config_data,
            input_refs=input_refs,
            output_refs=output_refs,
            metadata=metadata,
        )
        if result.generated_outputs:
            output_refs = output_refs + materialize_refs(result.generated_outputs)

        create_run_record(
            run_id=run_id,
            stage="train",
            status=result.status,
            invocation_argv=list(sys.argv) if sys.argv else ["rfx"],
            config_snapshot_data=config_data,
            input_refs=input_refs,
            output_refs=output_refs,
            metadata=result.metadata,
            reports=result.reports,
            artifacts=result.artifacts,
        )
    except Exception as exc:
        print(f"[rfx] Train failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"[rfx] train run_id={run_id} status={result.status}")
    if result.message:
        print(f"[rfx] {result.message}")
    for artifact in result.artifacts:
        print(f"[rfx] artifact: {artifact.get('ref')}")
    return 0 if result.status == "succeeded" else 1


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Check system setup."""
    import importlib.util
    import os
    import platform
    import shutil

    all_ok = True

    def _ok(label: str, detail: str = "") -> None:
        suffix = f"  {detail}" if detail else ""
        print(f"  {label:36s}  ok{suffix}")

    def _warn(label: str, detail: str = "") -> None:
        suffix = f"  {detail}" if detail else ""
        print(f"  {label:36s}  not available{suffix}")

    def _fail(label: str, detail: str = "") -> None:
        nonlocal all_ok
        all_ok = False
        suffix = f"  ({detail})" if detail else ""
        print(f"  {label:36s}  missing{suffix}")

    # -- header ---------------------------------------------------------------
    print("[rfx] doctor\n")

    # -- version info ---------------------------------------------------------
    print("  Version")
    try:
        from rfx import __version__

        print(f"    rfx-sdk          {__version__}")
    except Exception:
        print("    rfx-sdk          unknown")
    print(f"    Python           {platform.python_version()}")
    print(f"    Platform         {platform.platform()}")
    print()

    # -- required tools -------------------------------------------------------
    print("  Required tools")
    for name in ("python3", "cargo", "uv"):
        path = shutil.which(name)
        if path:
            _ok(name, path)
        else:
            _fail(name)
    print()

    # -- Rust extension -------------------------------------------------------
    print("  Rust extension")
    try:
        from rfx import _rfx  # noqa: F401

        _ok("rfx._rfx", f"v{_rfx.__version__}")
    except ImportError:
        _fail("rfx._rfx", "run: maturin develop")
    print()

    # -- core Python deps -----------------------------------------------------
    print("  Core Python packages")
    core_imports = [
        ("tinygrad", "tinygrad"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
    ]
    for module, label in core_imports:
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "")
            _ok(label, ver)
        except ImportError:
            _fail(label)
    print()

    # -- optional Python deps -------------------------------------------------
    print("  Optional packages")
    optional_imports = [
        ("torch", "PyTorch"),
        ("lerobot", "LeRobot"),
        ("cv2", "OpenCV"),
        ("anthropic", "Anthropic SDK"),
        ("openai", "OpenAI SDK"),
        ("mujoco", "MuJoCo"),
    ]
    for module, label in optional_imports:
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "")
            _ok(label, ver)
        except ImportError:
            _warn(label)
    print()

    # -- Zenoh transport ------------------------------------------------------
    print("  Transport")
    try:
        from rfx import _rfx as _ext

        if hasattr(_ext, "Topic"):
            _ok("Zenoh (via Rust extension)")
        else:
            _warn("Zenoh", "Rust extension missing Topic binding")
    except ImportError:
        _warn("Zenoh", "Rust extension not built")
    print()

    # -- robot configs --------------------------------------------------------
    print("  Robot configs")
    # configs/ lives at rfx/configs/ relative to the repo root.
    # __file__ is rfx/python/rfx/runtime/cli.py → walk up to rfx/, then into configs/
    configs_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs"
    if configs_dir.is_dir():
        for cfg_file in sorted(configs_dir.glob("*.yaml")):
            try:
                from rfx.robot.config import load_config

                load_config(str(cfg_file))
                _ok(cfg_file.name)
            except Exception as exc:
                _fail(cfg_file.name, str(exc))
    else:
        _warn("configs directory", str(configs_dir))
    print()

    # -- simulation backends --------------------------------------------------
    print("  Simulation backends")
    sim_backends = [
        ("rfx.sim.mock", "MockRobot", "torch"),
        ("rfx.sim.genesis", "Genesis (GPU)", "genesis"),
        ("rfx.sim.mjx", "MJX (JAX)", "mujoco"),
    ]
    for module, label, dep in sim_backends:
        if importlib.util.find_spec(dep) is None:
            _warn(label, f"requires {dep}")
        else:
            try:
                __import__(module)
                _ok(label)
            except Exception:
                _warn(label)
    print()

    # -- rfxJIT backends ------------------------------------------------------
    print("  rfxJIT backends")
    try:
        from rfxJIT.runtime.executor import available_backends

        backends = available_backends()
        for name, avail in backends.items():
            if avail:
                _ok(name)
            elif name == "cpu":
                _fail(name)
            else:
                _warn(name)
    except ImportError:
        # rfxJIT lives at repo root and may not be on sys.path in all contexts.
        # Try adding the repo root (two levels above rfx/python/rfx/runtime/).
        import sys

        repo_root = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from rfxJIT.runtime.executor import available_backends

            backends = available_backends()
            for name, avail in backends.items():
                if avail:
                    _ok(name)
                elif name == "cpu":
                    _fail(name)
                else:
                    _warn(name)
        except ImportError:
            _warn("rfxJIT", "not found")
    print()

    # -- hardware discovery ---------------------------------------------------
    print("  Hardware")
    try:
        from rfx.robot.discovery import discover_ports

        ports = discover_ports()
        if ports:
            for p in ports:
                port_name = p.get("port", str(p)) if isinstance(p, dict) else str(p)
                robot_type = p.get("robot_type", "unknown") if isinstance(p, dict) else ""
                _ok(port_name, robot_type)
        else:
            print("    No serial devices found")
    except Exception:
        _warn("Serial port detection")

    # Check Zenoh env vars
    zenoh_connect = os.environ.get("RFX_ZENOH_CONNECT", "")
    if zenoh_connect:
        _ok("RFX_ZENOH_CONNECT", zenoh_connect)
    else:
        print("    RFX_ZENOH_CONNECT              not set (using defaults)")
    print()

    # -- summary --------------------------------------------------------------
    if all_ok:
        print("[rfx] All good. Ready to go.")
    else:
        print("[rfx] Some required items missing. See above.")

    return 0


# ---------------------------------------------------------------------------
# runs (lightweight registry query)
# ---------------------------------------------------------------------------


def cmd_runs_list(args: argparse.Namespace) -> int:
    from rfx.workflow.registry import list_runs

    runs = list_runs(stage=args.stage, status=args.status, limit=args.limit)
    if not runs:
        print("No runs found.")
        return 0
    for run in runs:
        print(
            f"{run.get('run_id')}  {run.get('stage')}  {run.get('status')}  {run.get('finished_at')}"
        )
    return 0


def cmd_runs_show(args: argparse.Namespace) -> int:
    from rfx.workflow.registry import load_run

    run = load_run(args.run_id)
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rfx",
        description="rfx - The PyTorch for Robots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  rfx record --robot so101 --repo-id my-org/demos --episodes 10
  rfx train  --data demos/ --config train.yaml
  rfx deploy runs/my-policy --robot so101
  rfx deploy hf://user/my-policy --robot go2 --duration 60
  rfx deploy runs/my-policy --mock
""",
    )
    p.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    sp = p.add_subparsers(dest="cmd", required=True)

    # --- deploy ---
    s = sp.add_parser(
        "deploy",
        help="deploy a trained policy to a robot",
        description="Load a policy and run it on hardware (or mock). "
        "This is the main command — train, then deploy.",
    )
    s.add_argument(
        "policy",
        help='path to saved policy directory, or "hf://org/repo" for HuggingFace Hub',
    )
    s.add_argument(
        "--robot",
        "-r",
        default=None,
        help="robot type (so101, go2, g1) or path to YAML config",
    )
    s.add_argument("--config", default=None, help="path to robot YAML config (overrides --robot)")
    s.add_argument("--port", "-p", default=None, help="serial port or IP address override")
    s.add_argument(
        "--rate-hz", type=float, default=None, help="control loop Hz (default: from robot config)"
    )
    s.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="run time in seconds (default: infinite, Ctrl+C to stop)",
    )
    s.add_argument("--mock", action="store_true", help="use MockRobot (no hardware needed)")
    s.add_argument("--warmup", type=float, default=0.5, help="warmup sleep after reset (seconds)")
    s.set_defaults(fn=cmd_deploy)

    # --- record ---
    s = sp.add_parser(
        "record",
        help="record demonstrations from a robot",
        description="Collect teleoperation demos into a LeRobot dataset.",
    )
    s.add_argument("--robot", required=True, help="robot type (e.g. so101)")
    s.add_argument("--repo-id", required=True, help="HuggingFace dataset repo ID")
    s.add_argument("--output", "-o", default="datasets", help="output root directory")
    s.add_argument("--episodes", "-n", type=int, default=1, help="number of episodes to collect")
    s.add_argument(
        "--duration", "-d", type=float, default=None, help="duration per episode in seconds"
    )
    s.add_argument("--task", default="default", help="task label for episodes")
    s.add_argument("--fps", type=int, default=30, help="recording frame rate")
    s.add_argument("--push", action="store_true", help="push to Hub after collection")
    s.add_argument("--mcap", action="store_true", help="also log MCAP sidecar")
    s.add_argument("--state-dim", type=int, default=6, help="state dimension")
    s.set_defaults(fn=cmd_record)

    # --- train ---
    s = sp.add_parser(
        "train",
        help="train a policy from collected data",
        description="Run a training stage and register the resulting artifact.",
    )
    s.add_argument("--data", default=None, help="path to training data directory or dataset")
    s.add_argument("--config", default=None, help="training config file")
    s.add_argument(
        "--input", action="append", default=[], help="additional input refs (repeatable)"
    )
    s.add_argument(
        "--output", action="append", default=[], help="additional output refs (repeatable)"
    )
    s.set_defaults(fn=cmd_train)

    # --- doctor ---
    s = sp.add_parser("doctor", help="check system setup and dependencies")
    s.set_defaults(fn=cmd_doctor)

    # --- runs ---
    s = sp.add_parser("runs", help="query the run registry")
    runs_sp = s.add_subparsers(dest="runs_cmd", required=True)

    rs = runs_sp.add_parser("list", help="list past runs")
    rs.add_argument("--stage", default=None)
    rs.add_argument("--status", default=None)
    rs.add_argument("--limit", type=int, default=20)
    rs.set_defaults(fn=cmd_runs_list)

    rs = runs_sp.add_parser("show", help="show a run record")
    rs.add_argument("run_id")
    rs.set_defaults(fn=cmd_runs_show)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.fn(args))


if __name__ == "__main__":
    main()
