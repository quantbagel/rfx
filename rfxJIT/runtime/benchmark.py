"""Baseline-vs-IR benchmark harness for phase 0 rfxJIT."""

from __future__ import annotations

import argparse
import time

import numpy as np

from rfxJIT.kernels.ir import make_affine_relu_kernel
from rfxJIT.runtime.interpreter import execute_kernel


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
    """Benchmark the baseline path against the reference IR interpreter."""

    rng = np.random.default_rng(seed)
    shape = (size,)
    x = rng.standard_normal(size=shape, dtype=np.float32)
    scale = rng.standard_normal(size=shape, dtype=np.float32)
    bias = rng.standard_normal(size=shape, dtype=np.float32)

    kernel = make_affine_relu_kernel(shape=shape)

    for _ in range(warmup):
        baseline_affine_relu(x, scale, bias)
        execute_kernel(kernel, {"x": x, "scale": scale, "bias": bias})

    baseline_start = time.perf_counter()
    for _ in range(iterations):
        y_baseline = baseline_affine_relu(x, scale, bias)
    baseline_time = time.perf_counter() - baseline_start

    ir_start = time.perf_counter()
    for _ in range(iterations):
        y_ir = execute_kernel(kernel, {"x": x, "scale": scale, "bias": bias})
    ir_time = time.perf_counter() - ir_start

    if not np.allclose(y_baseline, y_ir, atol=1e-6):
        raise RuntimeError("IR output mismatch against baseline")

    return {
        "size": float(size),
        "iterations": float(iterations),
        "baseline_total_s": baseline_time,
        "ir_total_s": ir_time,
        "baseline_per_iter_ms": (baseline_time / iterations) * 1000.0,
        "ir_per_iter_ms": (ir_time / iterations) * 1000.0,
        "slowdown_x": ir_time / baseline_time if baseline_time > 0 else float("inf"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark phase 0 rfxJIT affine+relu kernel")
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

    print("rfxJIT phase0 benchmark")
    print(f"size={int(results['size'])} iterations={int(results['iterations'])}")
    print(f"baseline_per_iter_ms={results['baseline_per_iter_ms']:.4f}")
    print(f"ir_per_iter_ms={results['ir_per_iter_ms']:.4f}")
    print(f"slowdown_x={results['slowdown_x']:.2f}")


if __name__ == "__main__":
    main()
