from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..teleop.transport import InprocTransport, RustTransport, rust_transport_available


@dataclass
class NodeContext:
    name: str
    package: str
    params: dict[str, Any] = field(default_factory=dict)
    backend: str = "mock"
    transport: Any = None


class Node:
    """
    Base node contract for launch/runtime execution.
    """

    publish_topics: tuple[str, ...] = ()
    subscribe_topics: tuple[str, ...] = ()

    def __init__(self, context: NodeContext):
        self.ctx = context
        self._running = False
        self.transport = context.transport or (
            RustTransport() if rust_transport_available() else InprocTransport()
        )

    def setup(self) -> None:
        pass

    def tick(self) -> bool:
        time.sleep(0.01)
        return True

    def shutdown(self) -> None:
        pass

    def publish(self, key: str, payload: Mapping[str, Any]) -> None:
        self.transport.publish(
            key=key,
            payload=json.dumps(dict(payload)),
            metadata={"node": self.ctx.name, "package": self.ctx.package},
        )

    def run(self, rate_hz: float = 50.0, max_steps: int | None = None) -> int:
        period = 1.0 / max(rate_hz, 1.0)
        self._running = True
        steps = 0
        self.setup()
        try:
            while self._running:
                t0 = time.perf_counter()
                keep_going = bool(self.tick())
                steps += 1
                if not keep_going:
                    break
                if max_steps is not None and steps >= max_steps:
                    break
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            self.shutdown()
            self._running = False
        return steps

    def stop(self) -> None:
        self._running = False
