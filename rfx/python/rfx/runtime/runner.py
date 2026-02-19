from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from .launch import load_launch_file
from .node import Node, NodeContext
from .packages import discover_packages, resolve_node_entry
from .registry import load_registry, write_registry


def run_node(
    package: str,
    node: str,
    name: str | None,
    backend: str,
    params: dict[str, Any],
    rate_hz: float,
    max_steps: int | None = None,
) -> int:
    packages = discover_packages()
    node_obj, _pkg = resolve_node_entry(packages, package, node)

    node_name = name or f"{package}.{node}"
    ctx = NodeContext(name=node_name, package=package, params=params, backend=backend)
    instance = (
        node_obj(ctx)
        if isinstance(node_obj, type) and issubclass(node_obj, Node)
        else node_obj(ctx)
    )
    if not isinstance(instance, Node):
        raise TypeError(f"Node entry must produce Node, got: {type(instance)}")

    reg = load_registry()
    pubs = set(reg.get("topics", {}).get("publish", []))
    subs = set(reg.get("topics", {}).get("subscribe", []))
    pubs.update(getattr(instance, "publish_topics", ()) or ())
    subs.update(getattr(instance, "subscribe_topics", ()) or ())
    reg["topics"] = {"publish": sorted(pubs), "subscribe": sorted(subs)}
    write_registry(reg)

    steps = instance.run(rate_hz=rate_hz, max_steps=max_steps)
    return steps


def launch(spec_path: str | Path) -> int:
    spec = load_launch_file(spec_path)
    profile_env = {}
    if spec.profile in spec.profiles:
        profile_env = {str(k): str(v) for k, v in spec.profiles[spec.profile].items()}

    procs: list[subprocess.Popen] = []
    reg = {"launch": spec.name, "nodes": [], "topics": {"publish": [], "subscribe": []}}
    write_registry(reg)
    try:
        for n in spec.nodes:
            node_name = n.name or f"{n.package}.{n.node}"
            env = os.environ.copy()
            env["RFX_BACKEND"] = spec.backend
            env.update(profile_env)
            params_json = __import__("json").dumps(n.params)
            cmd = [
                sys.executable,
                "-m",
                "rfx.runtime.runner",
                "--run-node",
                "--package",
                n.package,
                "--node",
                n.node,
                "--name",
                node_name,
                "--backend",
                spec.backend,
                "--params-json",
                params_json,
                "--rate-hz",
                str(n.rate_hz),
            ]
            if n.max_steps is not None:
                cmd.extend(["--max-steps", str(n.max_steps)])
            p = subprocess.Popen(cmd, env=env)
            procs.append(p)
            reg["nodes"].append(
                {
                    "name": node_name,
                    "package": n.package,
                    "node": n.node,
                    "pid": p.pid,
                    "backend": spec.backend,
                    "params": n.params,
                }
            )
        write_registry(reg)

        for p in procs:
            p.wait()
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=3.0)
                except Exception:
                    p.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="rfx runtime process runner")
    parser.add_argument("--run-node", action="store_true")
    parser.add_argument("--package")
    parser.add_argument("--node")
    parser.add_argument("--name")
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--params-json", default="{}")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--launch-file")
    args = parser.parse_args()

    if args.launch_file:
        raise SystemExit(launch(args.launch_file))

    if args.run_node:
        import json

        params = json.loads(args.params_json)
        steps = run_node(
            package=args.package,
            node=args.node,
            name=args.name,
            backend=args.backend,
            params=params,
            rate_hz=args.rate_hz,
            max_steps=args.max_steps,
        )
        print(f"[rfx.runtime] node completed steps={steps}")
        return

    parser.error("No action selected. Use --launch-file or --run-node.")


if __name__ == "__main__":
    main()
