"""Baseline-vs-IR benchmark harness for phase 0/2 rfxJIT."""

from __future__ import annotations

import argparse
import time

import numpy as np

from rfxJIT.kernels.lowering import lower_kernel_ir
from rfxJIT.kernels.optimize import make_redundant_affine_relu_kernel, optimize_kernel_ir
from rfxJIT.runtime.executor import execute_lowered_kernel
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
) -> dict[str, float]:
    """Benchmark baseline, phase 0 IR, phase 1 lowered, and phase 2 optimized paths."""
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if size <= 0:
        raise ValueError("size must be > 0")

    rng = np.random.default_rng(seed)
    shape = (size,)
    x = rng.standard_normal(size=shape, dtype=np.float32)
    scale = rng.standard_normal(size=shape, dtype=np.float32)
    bias = rng.standard_normal(size=shape, dtype=np.float32)

    kernel = make_redundant_affine_relu_kernel(shape=shape)
    optimized_kernel = optimize_kernel_ir(kernel)
    lowered_kernel = lower_kernel_ir(kernel)
    lowered_optimized_kernel = lower_kernel_ir(optimized_kernel)

    with KernelDispatchQueue(autostart=True) as dispatch:
        for _ in range(warmup):
            baseline_affine_relu(x, scale, bias)
            execute_kernel(kernel, {"x": x, "scale": scale, "bias": bias})
            execute_lowered_kernel(lowered_kernel, {"x": x, "scale": scale, "bias": bias})
            execute_lowered_kernel(
                lowered_optimized_kernel,
                {"x": x, "scale": scale, "bias": bias},
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
        y_lowered = execute_lowered_kernel(lowered_kernel, {"x": x, "scale": scale, "bias": bias})
    lowered_time = time.perf_counter() - lowered_start

    lowered_optimized_start = time.perf_counter()
    for _ in range(iterations):
        y_lowered_optimized = execute_lowered_kernel(
            lowered_optimized_kernel,
            {"x": x, "scale": scale, "bias": bias},
        )
    lowered_optimized_time = time.perf_counter() - lowered_optimized_start

    with KernelDispatchQueue(autostart=True) as dispatch:
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

    return {
        "size": float(size),
        "iterations": float(iterations),
        "ops_before": float(len(kernel.ops)),
        "ops_after": float(len(optimized_kernel.ops)),
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark phase 0/2 rfxJIT affine+relu kernel")
    parser.add_argument("--size", type=int, default=65536, help="Number of elements in the kernel")
    parser.add_argument("--iterations", type=int, default=200, help="Timing iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    results = benchmark_affine_relu(
        size=args.size,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
    )

    print("rfxJIT phase0/2 benchmark")
    print(f"size={int(results['size'])} iterations={int(results['iterations'])}")
    print(f"ops_before={int(results['ops_before'])} ops_after={int(results['ops_after'])}")
    print(f"baseline_per_iter_ms={results['baseline_per_iter_ms']:.4f}")
    print(f"ir_per_iter_ms={results['ir_per_iter_ms']:.4f}")
    print(f"slowdown_x={results['slowdown_x']:.2f}")
    print(f"lowered_per_iter_ms={results['lowered_per_iter_ms']:.4f}")
    print(f"lowered_slowdown_x={results['lowered_slowdown_x']:.2f}")
    print(f"lowered_optimized_per_iter_ms={results['lowered_optimized_per_iter_ms']:.4f}")
    print(f"lowered_optimized_slowdown_x={results['lowered_optimized_slowdown_x']:.2f}")
    print(f"queue_per_iter_ms={results['queue_per_iter_ms']:.4f}")
    print(f"queue_slowdown_x={results['queue_slowdown_x']:.2f}")


if __name__ == "__main__":
    main()
