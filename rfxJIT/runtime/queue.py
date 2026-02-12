"""Dispatch queue for lowered kernels."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Thread

import numpy as np

from rfxJIT.kernels.lowering import LoweredKernel
from rfxJIT.runtime.executor import execute_lowered_kernel


@dataclass(frozen=True)
class _DispatchRequest:
    kernel: LoweredKernel
    named_inputs: dict[str, np.ndarray]
    future: Future[np.ndarray]


class KernelDispatchQueue:
    """Single-worker dispatch queue for deterministic phase 1 execution."""

    _STOP = object()

    def __init__(self, *, maxsize: int = 0, autostart: bool = True, backend: str = "cpu"):
        self._queue: Queue[_DispatchRequest | object] = Queue(maxsize=maxsize)
        self._thread: Thread | None = None
        self._closed = False
        self._backend = backend
        if autostart:
            self.start()

    def __enter__(self) -> KernelDispatchQueue:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("KernelDispatchQueue is closed")
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = Thread(target=self._worker_loop, name="rfxjit-dispatch", daemon=True)
        self._thread.start()

    def submit(
        self,
        kernel: LoweredKernel,
        named_inputs: Mapping[str, np.ndarray],
    ) -> Future[np.ndarray]:
        if self._closed:
            raise RuntimeError("Cannot submit to a closed KernelDispatchQueue")
        self.start()

        future: Future[np.ndarray] = Future()
        request = _DispatchRequest(
            kernel=kernel,
            named_inputs=dict(named_inputs),
            future=future,
        )
        self._queue.put(request)
        return future

    def join(self) -> None:
        self._queue.join()

    def stop(self, *, timeout: float | None = 5.0) -> None:
        if self._closed:
            return
        self._closed = True

        self._queue.put(self._STOP)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return

                assert isinstance(item, _DispatchRequest)
                try:
                    result = execute_lowered_kernel(
                        item.kernel,
                        item.named_inputs,
                        backend=self._backend,
                    )
                except Exception as exc:
                    item.future.set_exception(exc)
                else:
                    item.future.set_result(result)
            finally:
                self._queue.task_done()
