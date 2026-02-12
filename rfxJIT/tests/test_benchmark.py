"""Tests for benchmark report generation."""

from __future__ import annotations

import json

from rfxJIT.runtime.benchmark import (
    benchmark_affine_relu,
    build_benchmark_report,
    write_benchmark_report,
)


def test_benchmark_affine_relu_cpu_metrics() -> None:
    metrics, backend = benchmark_affine_relu(
        size=32,
        iterations=2,
        warmup=0,
        seed=7,
        backend="cpu",
    )

    assert backend == "cpu"
    assert metrics["ops_before"] >= metrics["ops_after"]
    assert metrics["baseline_per_iter_ms"] >= 0.0
    assert metrics["lowered_per_iter_ms"] >= 0.0
    assert metrics["queue_per_iter_ms"] >= 0.0


def test_write_benchmark_report_json(tmp_path) -> None:
    report = build_benchmark_report(
        size=32,
        iterations=2,
        warmup=0,
        seed=7,
        backend="cpu",
    )
    output_path = tmp_path / "rfxjit-benchmark.json"
    write_benchmark_report(report, str(output_path))

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["name"] == "rfxjit_affine_relu_phase02"
    assert payload["parameters"]["resolved_backend"] == "cpu"
    assert "slowdown_x" in payload["metrics"]
