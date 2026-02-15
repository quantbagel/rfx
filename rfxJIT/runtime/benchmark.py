"""Baseline-vs-IR benchmark harness for phase 0/2 rfxJIT."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rfxJIT.kernels.lowering import lower_kernel_ir
from rfxJIT.kernels.optimize import make_redundant_affine_relu_kernel, optimize_kernel_ir
from rfxJIT.runtime.executor import (
    available_backends,
    execute_lowered_kernel,
    resolve_backend_name,
)
from rfxJIT.runtime.interpreter import execute_kernel
from rfxJIT.runtime.queue import KernelDispatchQueue


def baseline_affine_relu(x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Direct NumPy baseline equivalent to the test kernel."""

    return np.maximum((x * scale) + bias, 0.0)


def benchmark_affine_relu(
    *,
    size: int,
    iterations: int,
    warmup: int,
    seed: int,
    backend: str = "cpu",
) -> tuple[dict[str, float | int], str]:
    """Benchmark baseline, phase 0 IR, phase 1 lowered, and phase 2 optimized paths."""
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if size <= 0:
        raise ValueError("size must be > 0")

    resolved_backend = resolve_backend_name(backend)

    rng = np.random.default_rng(seed)
    shape = (size,)
    x = rng.standard_normal(size=shape, dtype=np.float32)
    scale = rng.standard_normal(size=shape, dtype=np.float32)
    bias = rng.standard_normal(size=shape, dtype=np.float32)

    kernel = make_redundant_affine_relu_kernel(shape=shape)
    optimized_kernel = optimize_kernel_ir(kernel)
    lowered_kernel = lower_kernel_ir(kernel)
    lowered_optimized_kernel = lower_kernel_ir(optimized_kernel)

    with KernelDispatchQueue(autostart=True, backend=resolved_backend) as dispatch:
        for _ in range(warmup):
            baseline_affine_relu(x, scale, bias)
            execute_kernel(kernel, {"x": x, "scale": scale, "bias": bias})
            execute_lowered_kernel(
                lowered_kernel,
                {"x": x, "scale": scale, "bias": bias},
                backend=resolved_backend,
            )
            execute_lowered_kernel(
                lowered_optimized_kernel,
                {"x": x, "scale": scale, "bias": bias},
                backend=resolved_backend,
            )
            dispatch.submit(lowered_kernel, {"x": x, "scale": scale, "bias": bias}).result()

    baseline_start = time.perf_counter()
    for _ in range(iterations):
        y_baseline = baseline_affine_relu(x, scale, bias)
    baseline_time = time.perf_counter() - baseline_start

    ir_start = time.perf_counter()
    for _ in range(iterations):
        y_ir = execute_kernel(kernel, {"x": x, "scale": scale, "bias": bias})
    ir_time = time.perf_counter() - ir_start

    lowered_start = time.perf_counter()
    for _ in range(iterations):
        y_lowered = execute_lowered_kernel(
            lowered_kernel,
            {"x": x, "scale": scale, "bias": bias},
            backend=resolved_backend,
        )
    lowered_time = time.perf_counter() - lowered_start

    lowered_optimized_start = time.perf_counter()
    for _ in range(iterations):
        y_lowered_optimized = execute_lowered_kernel(
            lowered_optimized_kernel,
            {"x": x, "scale": scale, "bias": bias},
            backend=resolved_backend,
        )
    lowered_optimized_time = time.perf_counter() - lowered_optimized_start

    with KernelDispatchQueue(autostart=True, backend=resolved_backend) as dispatch:
        queue_start = time.perf_counter()
        for _ in range(iterations):
            y_queue = dispatch.submit(
                lowered_kernel,
                {"x": x, "scale": scale, "bias": bias},
            ).result()
        queue_time = time.perf_counter() - queue_start

    if not np.allclose(y_baseline, y_ir, atol=1e-6):
        raise RuntimeError("IR output mismatch against baseline")
    if not np.allclose(y_baseline, y_lowered, atol=1e-6):
        raise RuntimeError("Lowered output mismatch against baseline")
    if not np.allclose(y_baseline, y_lowered_optimized, atol=1e-6):
        raise RuntimeError("Optimized lowered output mismatch against baseline")
    if not np.allclose(y_baseline, y_queue, atol=1e-6):
        raise RuntimeError("Queue output mismatch against baseline")

    metrics = {
        "ops_before": len(kernel.ops),
        "ops_after": len(optimized_kernel.ops),
        "baseline_total_s": baseline_time,
        "ir_total_s": ir_time,
        "lowered_total_s": lowered_time,
        "lowered_optimized_total_s": lowered_optimized_time,
        "queue_total_s": queue_time,
        "baseline_per_iter_ms": (baseline_time / iterations) * 1000.0,
        "ir_per_iter_ms": (ir_time / iterations) * 1000.0,
        "lowered_per_iter_ms": (lowered_time / iterations) * 1000.0,
        "lowered_optimized_per_iter_ms": (lowered_optimized_time / iterations) * 1000.0,
        "queue_per_iter_ms": (queue_time / iterations) * 1000.0,
        "slowdown_x": ir_time / baseline_time if baseline_time > 0 else float("inf"),
        "lowered_slowdown_x": lowered_time / baseline_time if baseline_time > 0 else float("inf"),
        "lowered_optimized_slowdown_x": (
            lowered_optimized_time / baseline_time if baseline_time > 0 else float("inf")
        ),
        "queue_slowdown_x": queue_time / baseline_time if baseline_time > 0 else float("inf"),
    }
    return metrics, resolved_backend


def build_benchmark_report(
    *,
    size: int,
    iterations: int,
    warmup: int,
    seed: int,
    backend: str,
) -> dict[str, Any]:
    """Run the benchmark and return a structured report for humans and CI tooling."""
    metrics, resolved_backend = benchmark_affine_relu(
        size=size,
        iterations=iterations,
        warmup=warmup,
        seed=seed,
        backend=backend,
    )
    return {
        "schema_version": 1,
        "name": "rfxjit_affine_relu_phase02",
        "metadata": {
            "timestamp_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "available_backends": available_backends(),
        },
        "parameters": {
            "size": size,
            "iterations": iterations,
            "warmup": warmup,
            "seed": seed,
            "requested_backend": backend,
            "resolved_backend": resolved_backend,
        },
        "metrics": metrics,
    }


def write_benchmark_report(report: dict[str, Any], output_path: str) -> None:
    """Persist a benchmark report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark phase 0/2 rfxJIT affine+relu kernel")
    parser.add_argument("--size", type=int, default=65536, help="Number of elements in the kernel")
    parser.add_argument("--iterations", type=int, default=200, help="Timing iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--backend", type=str, default="cpu", help="Execution backend")
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write structured benchmark JSON output",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = build_benchmark_report(
        size=args.size,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        backend=args.backend,
    )
    params = report["parameters"]
    metrics = report["metrics"]

    if args.json_out:
        write_benchmark_report(report, args.json_out)

    print("rfxJIT phase0/2 benchmark")
    print(f"size={params['size']} iterations={params['iterations']}")
    print(f"backend={params['resolved_backend']} requested_backend={params['requested_backend']}")
    print(f"ops_before={metrics['ops_before']} ops_after={metrics['ops_after']}")
    print(f"baseline_per_iter_ms={metrics['baseline_per_iter_ms']:.4f}")
    print(f"ir_per_iter_ms={metrics['ir_per_iter_ms']:.4f}")
    print(f"slowdown_x={metrics['slowdown_x']:.2f}")
    print(f"lowered_per_iter_ms={metrics['lowered_per_iter_ms']:.4f}")
    print(f"lowered_slowdown_x={metrics['lowered_slowdown_x']:.2f}")
    print(f"lowered_optimized_per_iter_ms={metrics['lowered_optimized_per_iter_ms']:.4f}")
    print(f"lowered_optimized_slowdown_x={metrics['lowered_optimized_slowdown_x']:.2f}")
    print(f"queue_per_iter_ms={metrics['queue_per_iter_ms']:.4f}")
    print(f"queue_slowdown_x={metrics['queue_slowdown_x']:.2f}")
    if args.json_out:
        print(f"json_out={args.json_out}")


if __name__ == "__main__":
    main()
