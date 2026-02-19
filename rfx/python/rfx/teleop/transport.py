"""
rfx.teleop.transport - Zenoh-style in-process transport primitives.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .config import TransportConfig

_BYTES_LIKE = (bytes, bytearray, memoryview)

try:
    from rfx._rfx import (
        Transport as _RustTransport,
    )
except Exception:  # pragma: no cover - optional native extension path
    _RustTransport = None


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

    def subscribe(self, pattern: str, capacity: int = 1024) -> Subscription:
        _ = capacity  # in-process queue currently uses unbounded deques
        sub = Subscription(pattern=pattern)
        with self._lock:
            self._subscriptions.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subscriptions = [s for s in self._subscriptions if s is not sub]
        sub.close()

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscriptions)

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


class RustSubscription:
    """Python adapter around the native Rust `TransportSubscription`."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.pattern = inner.pattern

    @property
    def id(self) -> int:
        return int(self._inner.id)

    def recv(self, timeout_s: float | None = None) -> TransportEnvelope | None:
        if timeout_s is None:
            env = self._inner.recv()
        else:
            env = self._inner.recv_timeout(timeout_s)
        return _from_rust_envelope(env)

    def try_recv(self) -> TransportEnvelope | None:
        return _from_rust_envelope(self._inner.try_recv())

    def __len__(self) -> int:
        return len(self._inner)

    def is_empty(self) -> bool:
        return self._inner.is_empty()


class RustTransport:
    """
    Native Rust-backed keyed transport with the same API shape as `InprocTransport`.
    """

    def __init__(self) -> None:
        if _RustTransport is None:
            raise RuntimeError(
                "Rust transport bindings are unavailable. Build extension with maturin develop."
            )
        self._inner = _RustTransport()

    def subscribe(self, pattern: str, capacity: int = 1024) -> RustSubscription:
        return RustSubscription(self._inner.subscribe(pattern, capacity))

    def unsubscribe(self, sub: RustSubscription | int) -> bool:
        if isinstance(sub, RustSubscription):
            return bool(self._inner.unsubscribe(sub.id))
        return bool(self._inner.unsubscribe(int(sub)))

    def publish(
        self,
        key: str,
        payload: Any,
        *,
        metadata: dict[str, Any] | None = None,
        timestamp_ns: int | None = None,
    ) -> TransportEnvelope:
        if timestamp_ns is not None:
            # Native Rust transport stamps wall-time itself.
            # Caller-provided timestamp is currently tracked in metadata.
            metadata = dict(metadata or {})
            metadata["_timestamp_ns"] = int(timestamp_ns)

        payload_bytes = _normalize_payload_bytes(payload)
        metadata_json = json.dumps(metadata, sort_keys=True) if metadata else None
        env = self._inner.publish(key, payload_bytes, metadata_json)
        return _from_rust_envelope(env)

    @property
    def subscriber_count(self) -> int:
        return int(self._inner.subscriber_count)


class ZenohTransport:
    """
    Zenoh-backed keyed transport with the same API shape as `RustTransport`.

    Requires the ``zenoh`` feature to be compiled into the native extension.
    """

    def __init__(
        self,
        *,
        connect: tuple[str, ...] | list[str] = (),
        listen: tuple[str, ...] | list[str] = (),
        shared_memory: bool = True,
        key_prefix: str = "",
    ) -> None:
        if _RustTransport is None:
            raise RuntimeError(
                "Rust transport bindings are unavailable. Build extension with maturin develop."
            )
        if not hasattr(_RustTransport, "zenoh_available") or not _RustTransport.zenoh_available():
            raise RuntimeError(
                "Zenoh support is not compiled in. "
                "Rebuild with: maturin develop --features 'extension-module,zenoh'"
            )
        self._inner = _RustTransport.zenoh(
            list(connect),
            list(listen),
            shared_memory,
            key_prefix,
        )

    def subscribe(self, pattern: str, capacity: int = 1024) -> RustSubscription:
        return RustSubscription(self._inner.subscribe(pattern, capacity))

    def unsubscribe(self, sub: RustSubscription | int) -> bool:
        if isinstance(sub, RustSubscription):
            return bool(self._inner.unsubscribe(sub.id))
        return bool(self._inner.unsubscribe(int(sub)))

    def publish(
        self,
        key: str,
        payload: Any,
        *,
        metadata: dict[str, Any] | None = None,
        timestamp_ns: int | None = None,
    ) -> TransportEnvelope:
        if timestamp_ns is not None:
            metadata = dict(metadata or {})
            metadata["_timestamp_ns"] = int(timestamp_ns)

        payload_bytes = _normalize_payload_bytes(payload)
        metadata_json = json.dumps(metadata, sort_keys=True) if metadata else None
        env = self._inner.publish(key, payload_bytes, metadata_json)
        return _from_rust_envelope(env)

    @property
    def subscriber_count(self) -> int:
        return int(self._inner.subscriber_count)


def zenoh_transport_available() -> bool:
    """Return whether Zenoh transport support is compiled and importable."""
    if _RustTransport is None:
        return False
    if not hasattr(_RustTransport, "zenoh_available"):
        return False
    return bool(_RustTransport.zenoh_available())


class TransportLike(Protocol):
    """Unified transport protocol consumed by the teleop session runtime."""

    def subscribe(self, pattern: str, capacity: int = 1024) -> Any: ...

    def unsubscribe(self, sub: Any) -> Any: ...

    def publish(
        self,
        key: str,
        payload: Any,
        *,
        metadata: dict[str, Any] | None = None,
        timestamp_ns: int | None = None,
    ) -> TransportEnvelope: ...

    @property
    def subscriber_count(self) -> int: ...


def rust_transport_available() -> bool:
    """Return whether Rust transport bindings are importable."""
    return _RustTransport is not None


def create_transport(config: TransportConfig) -> TransportLike:
    """
    Resolve a concrete transport backend from teleop TransportConfig.

    - `inproc`: Rust-backed transport when available and `zero_copy_hot_path=True`,
      otherwise pure-Python in-process transport.
    - `zenoh` / `dds`: reserved backend names; explicit runtime error until wired.
    """
    backend = str(config.backend).strip().lower()
    if backend == "inproc":
        if bool(config.zero_copy_hot_path) and rust_transport_available():
            return RustTransport()
        return InprocTransport()

    if backend == "zenoh":
        from .config import ZenohConfig

        zenoh_cfg = config.zenoh if config.zenoh is not None else ZenohConfig()
        return ZenohTransport(
            connect=zenoh_cfg.connect,
            listen=zenoh_cfg.listen,
            shared_memory=zenoh_cfg.shared_memory,
            key_prefix=zenoh_cfg.key_prefix,
        )

    if backend == "dds":
        raise NotImplementedError(
            "Transport backend 'dds' is declared but not wired yet. Use backend='inproc' for now."
        )

    raise ValueError(
        f"Unsupported transport backend {config.backend!r}. Expected one of: inproc, zenoh, dds."
    )


def _normalize_payload_bytes(payload: Any) -> bytes:
    if isinstance(payload, memoryview):
        return payload.tobytes()
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if payload is None:
        return b""
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _from_rust_envelope(env: Any) -> TransportEnvelope | None:
    if env is None:
        return None
    metadata: dict[str, Any]
    if env.metadata_json:
        try:
            metadata = json.loads(env.metadata_json)
        except Exception:
            metadata = {"_raw_metadata_json": env.metadata_json}
    else:
        metadata = {}
    return TransportEnvelope(
        key=env.key,
        sequence=int(env.sequence),
        timestamp_ns=int(env.timestamp_ns),
        payload=memoryview(bytes(env.payload)),
        metadata=metadata,
    )
