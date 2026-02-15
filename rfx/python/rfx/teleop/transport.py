"""
rfx.teleop.transport - Zenoh-style in-process transport primitives.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
import threading
import time
from typing import Any


_BYTES_LIKE = (bytes, bytearray, memoryview)


@dataclass(frozen=True)
class TransportEnvelope:
    """Message envelope carrying routing and timing metadata."""

    key: str
    sequence: int
    timestamp_ns: int
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class Subscription:
    """Subscription cursor for keyed transport messages."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self._queue: deque[TransportEnvelope] = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._closed = False

    def push(self, envelope: TransportEnvelope) -> None:
        with self._cond:
            if self._closed:
                return
            self._queue.append(envelope)
            self._cond.notify()

    def recv(self, timeout_s: float | None = None) -> TransportEnvelope | None:
        with self._cond:
            if timeout_s is None:
                while not self._queue and not self._closed:
                    self._cond.wait()
            elif not self._queue and not self._closed:
                self._cond.wait(timeout=timeout_s)

            if self._queue:
                return self._queue.popleft()
            return None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._queue.clear()
            self._cond.notify_all()


class InprocTransport:
    """
    In-process keyed pub/sub transport with Zenoh-like key-expression matching.

    Pattern matching currently uses shell-style wildcards (`*`, `?`, `**`) via `fnmatch`.
    """

    def __init__(self) -> None:
        self._subscriptions: list[Subscription] = []
        self._lock = threading.Lock()
        self._seq = 0

    def subscribe(self, pattern: str) -> Subscription:
        sub = Subscription(pattern=pattern)
        with self._lock:
            self._subscriptions.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subscriptions = [s for s in self._subscriptions if s is not sub]
        sub.close()

    def publish(
        self,
        key: str,
        payload: Any,
        *,
        metadata: dict[str, Any] | None = None,
        timestamp_ns: int | None = None,
    ) -> TransportEnvelope:
        now_ns = int(timestamp_ns if timestamp_ns is not None else time.time_ns())
        with self._lock:
            self._seq += 1
            seq = self._seq
            subscribers = tuple(self._subscriptions)

        normalized_payload: Any
        if isinstance(payload, _BYTES_LIKE):
            # Preserve bytes-like payload views for zero-copy hot-path callers.
            normalized_payload = payload if isinstance(payload, memoryview) else memoryview(payload)
        else:
            normalized_payload = payload

        envelope = TransportEnvelope(
            key=key,
            sequence=seq,
            timestamp_ns=now_ns,
            payload=normalized_payload,
            metadata=dict(metadata or {}),
        )

        for sub in subscribers:
            if fnmatchcase(key, sub.pattern):
                sub.push(envelope)

        return envelope
