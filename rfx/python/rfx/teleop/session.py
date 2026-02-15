"""
rfx.teleop.session - High-rate SO-101 teleoperation runtime.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import threading
import time
from typing import Any, Protocol

import numpy as np

from .config import ArmPairConfig, CameraStreamConfig, TeleopSessionConfig
from .recorder import LeRobotRecorder, RecordedEpisode


@dataclass(frozen=True)
class LoopTimingStats:
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


class ArmPair(Protocol):
    """Protocol for leader/follower arm pairs used by the session loop."""

    name: str

    def step(self) -> Sequence[float]: ...

    def go_home(self) -> None: ...

    def disconnect(self) -> None: ...


class _So101ArmPair:
    """Default SO-101 leader/follower arm pair implementation."""

    def __init__(self, pair: ArmPairConfig) -> None:
        from ..config import SO101_CONFIG
        from ..real.so101 import So101Backend

        self.name = pair.name
        self._leader = So101Backend(config=SO101_CONFIG, port=pair.leader_port, is_leader=True)
        self._follower = So101Backend(config=SO101_CONFIG, port=pair.follower_port, is_leader=False)

    def step(self) -> Sequence[float]:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch is required for SO-101 teleoperation sessions") from exc

        positions = self._leader.read_positions()
        padded = torch.zeros(1, self._follower.config.max_action_dim, dtype=positions.dtype)
        padded[0, : positions.shape[0]] = positions
        self._follower.act(padded)
        return [float(v) for v in positions.tolist()]

    def go_home(self) -> None:
        self._follower.go_home()

    def disconnect(self) -> None:
        self._leader.disconnect()
        self._follower.disconnect()


class _CameraWorker:
    """Asynchronous camera reader that captures frames independently of control loop."""

    def __init__(
        self,
        config: CameraStreamConfig,
        frame_callback: Callable[[str, int, int, np.ndarray], None],
    ) -> None:
        self.config = config
        self._frame_callback = frame_callback

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_count = 0
        self._latest_frame_index = -1
        self._latest_timestamp_ns = 0
        self._error: Exception | None = None
        self._state_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.config.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def latest_frame_index(self) -> int:
        with self._state_lock:
            return self._latest_frame_index

    @property
    def latest_timestamp_ns(self) -> int:
        with self._state_lock:
            return self._latest_timestamp_ns

    @property
    def frame_count(self) -> int:
        with self._state_lock:
            return self._frame_count

    @property
    def error(self) -> Exception | None:
        return self._error

    def _run(self) -> None:
        try:
            from ..real.camera import Camera
        except Exception as exc:  # pragma: no cover - environment dependent
            self._error = exc
            return

        period_s = 1.0 / max(self.config.fps, 1)
        camera = Camera(
            device_id=self.config.device_id,
            resolution=(self.config.width, self.config.height),
            fps=self.config.fps,
        )

        frame_index = -1
        try:
            while self._running.is_set():
                tick_start = time.perf_counter()
                frame = camera.capture()
                frame_index += 1
                ts_ns = time.time_ns()
                frame_array = self._to_numpy(frame)

                with self._state_lock:
                    self._frame_count = frame_index + 1
                    self._latest_frame_index = frame_index
                    self._latest_timestamp_ns = ts_ns

                self._frame_callback(self.config.name, frame_index, ts_ns, frame_array)

                elapsed = time.perf_counter() - tick_start
                sleep_s = period_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._error = exc
        finally:
            camera.release()

    @staticmethod
    def _to_numpy(frame: Any) -> np.ndarray:
        value = frame
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)


class BimanualSo101Session:
    """Python-first high-rate teleop session with async cameras and recorder integration."""

    def __init__(
        self,
        config: TeleopSessionConfig,
        *,
        recorder: LeRobotRecorder | None = None,
        pair_factory: Callable[[ArmPairConfig], ArmPair] | None = None,
    ) -> None:
        self.config = config
        self.recorder = recorder or LeRobotRecorder(config.output_dir)
        self._pair_factory = pair_factory or _So101ArmPair

        self._pairs: list[ArmPair] = []
        self._camera_workers: list[_CameraWorker] = []
        self._control_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._state_lock = threading.Lock()
        self._record_lock = threading.Lock()
        self._active_episode_id: str | None = None
        self._latest_positions: dict[str, tuple[float, ...]] = {}
        self._latest_camera_indices: dict[str, int] = {
            camera.name: -1 for camera in self.config.cameras
        }
        self._last_timestamp_ns: int = 0
        self._loop_error: Exception | None = None

        self._period_samples = deque(maxlen=self.config.max_timing_samples)
        self._jitter_samples = deque(maxlen=self.config.max_timing_samples)
        self._iterations = 0
        self._overruns = 0

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def is_recording(self) -> bool:
        with self._record_lock:
            return self._active_episode_id is not None

    @classmethod
    def from_ports(
        cls,
        *,
        left_leader_port: str = "/dev/ttyACM0",
        left_follower_port: str = "/dev/ttyACM1",
        right_leader_port: str = "/dev/ttyACM2",
        right_follower_port: str = "/dev/ttyACM3",
        **kwargs: Any,
    ) -> "BimanualSo101Session":
        config = TeleopSessionConfig.bimanual(
            left_leader_port=left_leader_port,
            left_follower_port=left_follower_port,
            right_leader_port=right_leader_port,
            right_follower_port=right_follower_port,
            **kwargs,
        )
        return cls(config=config)

    @classmethod
    def from_single_pair(
        cls,
        *,
        leader_port: str = "/dev/ttyACM0",
        follower_port: str = "/dev/ttyACM1",
        **kwargs: Any,
    ) -> "BimanualSo101Session":
        config = TeleopSessionConfig.single_arm_pair(
            name="main",
            leader_port=leader_port,
            follower_port=follower_port,
            **kwargs,
        )
        return cls(config=config)

    def start(self) -> None:
        if self.is_running:
            return

        self._pairs = [self._pair_factory(pair_config) for pair_config in self.config.arm_pairs]
        self._loop_error = None

        self._camera_workers = [
            _CameraWorker(camera_config, frame_callback=self._on_camera_frame)
            for camera_config in self.config.cameras
        ]
        for worker in self._camera_workers:
            worker.start()

        self._running.set()
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="teleop-control-loop",
            daemon=True,
        )
        self._control_thread.start()

    def stop(self) -> None:
        if not self.is_running and self._control_thread is None:
            return

        if self.is_recording:
            self.stop_recording()

        self._running.clear()
        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None

        for worker in self._camera_workers:
            worker.stop()
        self._camera_workers = []

        for pair in self._pairs:
            pair.disconnect()
        self._pairs = []

    def go_home(self) -> None:
        for pair in self._pairs:
            pair.go_home()

    def latest_positions(self) -> dict[str, tuple[float, ...]]:
        with self._state_lock:
            return dict(self._latest_positions)

    def latest_camera_frame_indices(self) -> dict[str, int]:
        with self._state_lock:
            return dict(self._latest_camera_indices)

    def latest_timestamp_ns(self) -> int:
        with self._state_lock:
            return int(self._last_timestamp_ns)

    def start_recording(
        self,
        *,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        with self._record_lock:
            if self._active_episode_id is not None:
                raise RuntimeError("Recording is already active")
            episode_id = self.recorder.start_episode(
                session_config=self.config,
                label=label,
                metadata=metadata,
            )
            self._active_episode_id = episode_id
            return episode_id

    def stop_recording(self) -> RecordedEpisode:
        with self._record_lock:
            if self._active_episode_id is None:
                raise RuntimeError("Recording is not active")
            self._active_episode_id = None
        return self.recorder.finalize_episode(loop_stats=self.timing_stats().to_dict())

    def record_episode(
        self,
        *,
        duration_s: float,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RecordedEpisode:
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")

        if not self.is_running:
            self.start()

        self.start_recording(label=label, metadata=metadata)
        deadline = time.perf_counter() + duration_s
        while self.is_running and time.perf_counter() < deadline:
            time.sleep(0.01)
        return self.stop_recording()

    def timing_stats(self) -> LoopTimingStats:
        with self._state_lock:
            period_samples = np.asarray(tuple(self._period_samples), dtype=np.float64)
            jitter_samples = np.asarray(tuple(self._jitter_samples), dtype=np.float64)
            iterations = self._iterations
            overruns = self._overruns

        if period_samples.size == 0:
            return LoopTimingStats(
                iterations=iterations,
                overruns=overruns,
                target_period_s=self.config.period_s,
                avg_period_s=0.0,
                p50_jitter_s=0.0,
                p95_jitter_s=0.0,
                p99_jitter_s=0.0,
                max_jitter_s=0.0,
            )

        return LoopTimingStats(
            iterations=iterations,
            overruns=overruns,
            target_period_s=self.config.period_s,
            avg_period_s=float(np.mean(period_samples)),
            p50_jitter_s=float(np.percentile(jitter_samples, 50)),
            p95_jitter_s=float(np.percentile(jitter_samples, 95)),
            p99_jitter_s=float(np.percentile(jitter_samples, 99)),
            max_jitter_s=float(np.max(jitter_samples)),
        )

    def check_health(self) -> None:
        if self._loop_error is not None:
            raise RuntimeError("Control loop failed") from self._loop_error
        for worker in self._camera_workers:
            if worker.error is not None:
                raise RuntimeError(f"Camera worker failed: {worker.config.name}") from worker.error

    def _control_loop(self) -> None:
        target_period = self.config.period_s
        next_deadline = time.perf_counter()
        last_loop_start = next_deadline

        while self._running.is_set():
            loop_start = time.perf_counter()
            dt = loop_start - last_loop_start
            last_loop_start = loop_start

            try:
                pair_positions: dict[str, tuple[float, ...]] = {}
                for pair in self._pairs:
                    values = tuple(float(v) for v in pair.step())
                    pair_positions[pair.name] = values
            except Exception as exc:
                self._loop_error = exc
                self._running.clear()
                break

            timestamp_ns = time.time_ns()
            camera_indices = self.latest_camera_frame_indices()

            with self._state_lock:
                self._latest_positions = pair_positions
                self._last_timestamp_ns = timestamp_ns
                self._iterations += 1
                if dt > target_period:
                    self._overruns += 1
                jitter = abs(dt - target_period)
                self._period_samples.append(dt)
                self._jitter_samples.append(jitter)

            with self._record_lock:
                recording = self._active_episode_id is not None
            if recording:
                self.recorder.append_control_step(
                    timestamp_ns=timestamp_ns,
                    dt_s=dt,
                    pair_positions=pair_positions,
                    camera_frame_indices=camera_indices,
                )

            next_deadline += target_period
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_deadline = time.perf_counter()

    def _on_camera_frame(
        self,
        camera_name: str,
        frame_index: int,
        timestamp_ns: int,
        frame: np.ndarray,
    ) -> None:
        with self._state_lock:
            self._latest_camera_indices[camera_name] = int(frame_index)

        with self._record_lock:
            recording = self._active_episode_id is not None
        if recording:
            self.recorder.append_camera_frame(
                camera_name=camera_name,
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                frame=frame,
            )

    def __enter__(self) -> "BimanualSo101Session":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()
