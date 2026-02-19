"""
rfx.session - Hardened inference-robot connection runtime.

Provides a Session class that connects a policy to a robot with lifecycle
management, rate control, jitter tracking, error handling, and clean shutdown.

Example:
    >>> rfx.run(robot, policy, rate_hz=50, duration=10.0)

    >>> with rfx.Session(robot, policy, rate_hz=50) as s:
    ...     s.run(duration=10.0)
    ...     print(s.stats)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

    from .robot import Robot


@dataclass(frozen=True)
class SessionStats:
    """Loop timing summary for jitter and overrun analysis."""

    iterations: int
    overruns: int
    target_period_s: float
    avg_period_s: float
    p50_jitter_s: float
    p95_jitter_s: float
    p99_jitter_s: float
    max_jitter_s: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "iterations": int(self.iterations),
            "overruns": int(self.overruns),
            "target_period_s": float(self.target_period_s),
            "avg_period_s": float(self.avg_period_s),
            "p50_jitter_s": float(self.p50_jitter_s),
            "p95_jitter_s": float(self.p95_jitter_s),
            "p99_jitter_s": float(self.p99_jitter_s),
            "max_jitter_s": float(self.max_jitter_s),
        }


_MAX_TIMING_SAMPLES = 10_000


class Session:
    """Hardened inference-robot connection with lifecycle management.

    Connects a policy callable to a robot with rate control, jitter tracking,
    error handling, and clean shutdown. Thread-based, no async forced on users.

    Args:
        robot: Any object satisfying the Robot protocol (observe/act/reset).
        policy: Callable that maps an observation dict to an action tensor.
        rate_hz: Target control loop frequency in Hz.
        warmup_s: Seconds to sleep after reset before starting the loop.

    Example:
        >>> with Session(robot, policy, rate_hz=50) as s:
        ...     s.run(duration=10.0)
        ...     print(s.stats)
    """

    def __init__(
        self,
        robot: Robot,
        policy: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        rate_hz: float = 50,
        warmup_s: float = 0.5,
    ) -> None:
        self._robot = robot
        self._policy = policy
        self._rate_hz = rate_hz
        self._warmup_s = warmup_s
        self._target_period = 1.0 / rate_hz

        self._control_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._stop_requested = threading.Event()
        self._loop_error: Exception | None = None

        self._lock = threading.Lock()
        self._period_samples: deque[float] = deque(maxlen=_MAX_TIMING_SAMPLES)
        self._jitter_samples: deque[float] = deque(maxlen=_MAX_TIMING_SAMPLES)
        self._iterations = 0
        self._overruns = 0

    @property
    def step_count(self) -> int:
        with self._lock:
            return self._iterations

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def stats(self) -> SessionStats:
        with self._lock:
            period_samples = np.asarray(tuple(self._period_samples), dtype=np.float64)
            jitter_samples = np.asarray(tuple(self._jitter_samples), dtype=np.float64)
            iterations = self._iterations
            overruns = self._overruns

        if period_samples.size == 0:
            return SessionStats(
                iterations=iterations,
                overruns=overruns,
                target_period_s=self._target_period,
                avg_period_s=0.0,
                p50_jitter_s=0.0,
                p95_jitter_s=0.0,
                p99_jitter_s=0.0,
                max_jitter_s=0.0,
            )

        return SessionStats(
            iterations=iterations,
            overruns=overruns,
            target_period_s=self._target_period,
            avg_period_s=float(np.mean(period_samples)),
            p50_jitter_s=float(np.percentile(jitter_samples, 50)),
            p95_jitter_s=float(np.percentile(jitter_samples, 95)),
            p99_jitter_s=float(np.percentile(jitter_samples, 99)),
            max_jitter_s=float(np.max(jitter_samples)),
        )

    def check_health(self) -> None:
        """Raise if the control thread encountered an error."""
        if self._loop_error is not None:
            raise RuntimeError("Control loop failed") from self._loop_error

    def start(self) -> None:
        """Reset robot and spawn the daemon control thread."""
        if self._running.is_set():
            return

        self._loop_error = None
        self._stop_requested.clear()
        self._robot.reset()

        if self._warmup_s > 0:
            time.sleep(self._warmup_s)

        self._running.set()
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="rfx-session-control",
            daemon=True,
        )
        self._control_thread.start()

    def stop(self) -> None:
        """Signal stop and join the control thread. Idempotent."""
        if not self._running.is_set() and self._control_thread is None:
            return

        self._stop_requested.set()
        self._running.clear()
        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None

    def run(self, duration: float | None = None) -> None:
        """Blocking run. ``None`` means infinite until Ctrl+C / stop().

        Args:
            duration: Maximum run time in seconds. None for infinite.
        """
        if not self._running.is_set():
            self.start()

        try:
            if duration is None:
                # Infinite — wait until stop() or Ctrl+C
                while self._running.is_set():
                    self.check_health()
                    time.sleep(0.05)
            else:
                deadline = time.perf_counter() + duration
                while self._running.is_set() and time.perf_counter() < deadline:
                    self.check_health()
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

        # Re-raise loop errors after stop
        self.check_health()

    def _control_loop(self) -> None:
        import torch

        target_period = self._target_period
        next_deadline = time.perf_counter()
        last_loop_start = next_deadline

        while self._running.is_set() and not self._stop_requested.is_set():
            loop_start = time.perf_counter()
            dt = loop_start - last_loop_start
            last_loop_start = loop_start

            try:
                obs = self._robot.observe()
                with torch.no_grad():
                    action = self._policy(obs)
                self._robot.act(action)
            except Exception as exc:
                self._loop_error = exc
                self._running.clear()
                break

            with self._lock:
                self._iterations += 1
                if dt > target_period:
                    self._overruns += 1
                jitter = abs(dt - target_period)
                self._period_samples.append(dt)
                self._jitter_samples.append(jitter)

            next_deadline += target_period
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                # Hybrid wait: coarse sleep + busy spin for last ~1.2ms
                spin_window_s = 0.0012
                if sleep_s > spin_window_s:
                    time.sleep(sleep_s - spin_window_s)
                while time.perf_counter() < next_deadline:
                    pass
            else:
                # Fell behind — reset deadline to avoid burst catch-up
                next_deadline = time.perf_counter()

    def __enter__(self) -> Session:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def run(
    robot: Robot,
    policy: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    rate_hz: float = 50,
    duration: float | None = None,
    warmup_s: float = 0.5,
) -> SessionStats:
    """One-liner to connect a policy to a robot and run inference.

    Args:
        robot: Any object satisfying the Robot protocol.
        policy: Callable mapping observation dict to action tensor.
        rate_hz: Target control loop frequency in Hz.
        duration: Run time in seconds. None for infinite (Ctrl+C to stop).
        warmup_s: Seconds to sleep after reset before starting the loop.

    Returns:
        SessionStats with timing and jitter information.
    """
    with Session(robot, policy, rate_hz=rate_hz, warmup_s=warmup_s) as s:
        s.run(duration=duration)
        return s.stats
