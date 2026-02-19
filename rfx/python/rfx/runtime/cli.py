from __future__ import annotations

import argparse
import json
from pathlib import Path

from .packages import discover_packages
from .registry import load_registry
from .runner import launch, run_node

TEMPLATE_MANIFEST = """\
[package]
name = "{name}"
version = "0.1.0"
python_module = "src"

[nodes]
{name}_node = "{module}:{symbol}"
"""

TEMPLATE_NODE = """\
from __future__ import annotations

from rfx.runtime.node import Node, NodeContext


class {symbol}(Node):
    publish_topics = ("{name}/state",)
    subscribe_topics = ()

    def __init__(self, context: NodeContext):
        super().__init__(context)
        self.counter = 0

    def tick(self) -> bool:
        self.counter += 1
        self.publish("{name}/state", {{"counter": self.counter, "backend": self.ctx.backend}})
        return True
"""

TEMPLATE_LAUNCH = """\
name: demo
backend: mock
profile: default
profiles:
  default:
    RFX_BACKEND: mock
nodes:
  - package: {name}
    node: {name}_node
    name: {name}.main
    rate_hz: 20
    max_steps: 200
    params: {{}}
"""


def cmd_pkg_create(args: argparse.Namespace) -> int:
    root = Path.cwd() / "packages" / args.name
    src_mod = root / "src" / args.name.replace("-", "_")
    src_mod.mkdir(parents=True, exist_ok=True)

    module = f"{args.name.replace('-', '_')}.nodes"
    symbol = "MainNode"
    (root / "rfx_pkg.toml").write_text(
        TEMPLATE_MANIFEST.format(name=args.name, module=module, symbol=symbol)
    )
    (root / "src" / args.name.replace("-", "_") / "__init__.py").write_text("")
    (root / "src" / args.name.replace("-", "_") / "nodes.py").write_text(
        TEMPLATE_NODE.format(name=args.name, symbol=symbol)
    )
    (root / "launch.yaml").write_text(TEMPLATE_LAUNCH.format(name=args.name))
    print(f"[rfx] created package: {root}")
    print(f"[rfx] run with: rfx run {args.name} {args.name}_node")
    print(f"[rfx] launch with: rfx launch packages/{args.name}/launch.yaml")
    return 0


def cmd_pkg_list(_args: argparse.Namespace) -> int:
    pkgs = discover_packages()
    if not pkgs:
        print("No rfx packages found under ./packages")
        return 0
    for name, pkg in sorted(pkgs.items()):
        print(f"{name}\t{pkg.version}\t{pkg.root}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    params = json.loads(args.params_json or "{}")
    steps = run_node(
        package=args.package,
        node=args.node,
        name=args.name,
        backend=args.backend,
        params=params,
        rate_hz=args.rate_hz,
        max_steps=args.max_steps,
    )
    print(f"[rfx] node completed steps={steps}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    return launch(args.file)


def cmd_graph(_args: argparse.Namespace) -> int:
    reg = load_registry()
    print(f"launch: {reg.get('launch')}")
    for n in reg.get("nodes", []):
        print(
            f"- {n['name']} (pkg={n['package']} node={n['node']} pid={n['pid']} backend={n.get('backend', '')})"
        )
    return 0


def cmd_topic_list(_args: argparse.Namespace) -> int:
    reg = load_registry()
    pubs = set(reg.get("topics", {}).get("publish", []))
    subs = set(reg.get("topics", {}).get("subscribe", []))
    topics = sorted(pubs | subs)
    if not topics:
        print("No topics registered in runtime graph yet.")
        return 0
    for t in topics:
        print(t)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="rfx runtime CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("pkg-create", help="create a new runtime package")
    s.add_argument("name")
    s.set_defaults(fn=cmd_pkg_create)

    s = sp.add_parser("pkg-list", help="list discovered runtime packages")
    s.set_defaults(fn=cmd_pkg_list)

    s = sp.add_parser("run", help="run a package node")
    s.add_argument("package")
    s.add_argument("node")
    s.add_argument("--name")
    s.add_argument("--backend", default="mock")
    s.add_argument("--rate-hz", type=float, default=50.0)
    s.add_argument("--max-steps", type=int, default=None)
    s.add_argument("--params-json", default="{}")
    s.set_defaults(fn=cmd_run)

    s = sp.add_parser("launch", help="run a launch file")
    s.add_argument("file")
    s.set_defaults(fn=cmd_launch)

    s = sp.add_parser("graph", help="show active launch graph")
    s.set_defaults(fn=cmd_graph)

    s = sp.add_parser("topic-list", help="list runtime topics")
    s.set_defaults(fn=cmd_topic_list)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.fn(args))


if __name__ == "__main__":
    main()
