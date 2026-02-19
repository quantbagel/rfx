from __future__ import annotations

import importlib
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RfxPackage:
    name: str
    root: Path
    module_path: Path
    nodes: dict[str, str]
    version: str = "0.1.0"


def _workspace_root() -> Path:
    return Path.cwd()


def discover_packages(root: Path | None = None) -> dict[str, RfxPackage]:
    root = root or _workspace_root()
    pkgs_dir = root / "packages"
    out: dict[str, RfxPackage] = {}
    if not pkgs_dir.exists():
        return out

    for manifest in pkgs_dir.glob("*/rfx_pkg.toml"):
        data = tomllib.loads(manifest.read_text())
        pkg = data.get("package", {})
        nodes = data.get("nodes", {})
        mod_rel = pkg.get("python_module", "src")
        name = str(pkg.get("name", manifest.parent.name))
        out[name] = RfxPackage(
            name=name,
            root=manifest.parent,
            module_path=manifest.parent / mod_rel,
            nodes={str(k): str(v) for k, v in nodes.items()},
            version=str(pkg.get("version", "0.1.0")),
        )
    return out


def resolve_node_entry(
    packages: dict[str, RfxPackage], package: str, node: str
) -> tuple[Any, RfxPackage]:
    if package not in packages:
        raise KeyError(f"Unknown package '{package}'. Available: {sorted(packages)}")
    pkg = packages[package]
    if node not in pkg.nodes:
        raise KeyError(
            f"Unknown node '{node}' in package '{package}'. Available: {sorted(pkg.nodes)}"
        )

    entry = pkg.nodes[node]
    module_name, _, symbol = entry.partition(":")
    if not module_name or not symbol:
        raise ValueError(f"Invalid node entry '{entry}'. Expected 'module.submodule:Symbol'")

    sys.path.insert(0, str(pkg.module_path))
    mod = importlib.import_module(module_name)
    obj = getattr(mod, symbol)
    return obj, pkg
