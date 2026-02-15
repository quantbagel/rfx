"""
rfx.teleop.benchmark - Jitter benchmark helpers for control-loop acceptance gates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import TeleopSessionConfig
from .session import BimanualSo101Session, LoopTimingStats


@dataclass(frozen=True)
class JitterBenchmarkResult:
    """Summary result for a teleop loop jitter benchmark run."""

    rate_hz: float
    duration_s: float
    warmup_s: float
    stats: LoopTimingStats

    def to_dict(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "rate_hz": float(self.rate_hz),
            "duration_s": float(self.duration_s),
            "warmup_s": float(self.warmup_s),
        }
        payload.update(self.stats.to_dict())
        return payload


class _BenchmarkPair:
    def __init__(self, name: str) -> None:
        self.name = name
        self._tick = 0

    def step(self) -> list[float]:
        self._tick += 1
        value = float(self._tick)
        return [value] * 6

    def go_home(self) -> None:
        return None

    def disconnect(self) -> None:
        return None


def run_jitter_benchmark(
    *,
    rate_hz: float = 350.0,
    duration_s: float = 2.0,
    warmup_s: float = 0.5,
    max_timing_samples: int = 100_000,
) -> JitterBenchmarkResult:
    """
    Run a hardware-free jitter benchmark using fake arm pairs.
    """
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if warmup_s < 0:
        raise ValueError("warmup_s must be >= 0")

    config = TeleopSessionConfig.single_arm_pair(
        rate_hz=rate_hz,
        cameras=(),
        max_timing_samples=max_timing_samples,
    )
    session = BimanualSo101Session(
        config=config,
        pair_factory=lambda pair_cfg: _BenchmarkPair(pair_cfg.name),
    )

    try:
        session.start()
        if warmup_s > 0:
            time.sleep(warmup_s)
        session.reset_timing_stats()
        time.sleep(duration_s)
        stats = session.timing_stats()
    finally:
        session.stop()

    return JitterBenchmarkResult(
        rate_hz=rate_hz,
        duration_s=duration_s,
        warmup_s=warmup_s,
        stats=stats,
    )


def assert_jitter_budget(
    result: JitterBenchmarkResult,
    *,
    p99_budget_s: float = 0.0005,
) -> None:
    """Raise when p99 jitter exceeds the configured budget."""
    if result.stats.p99_jitter_s > p99_budget_s:
        raise RuntimeError(
            f"Jitter budget exceeded: p99={result.stats.p99_jitter_s * 1e3:.3f}ms "
            f"budget={p99_budget_s * 1e3:.3f}ms"
        )
