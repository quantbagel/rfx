"""Tests for rfx.runtime.health - Watchdog, ReconnectPolicy, HealthMonitor."""

from __future__ import annotations

import threading
import time

from rfx.runtime.health import HealthMonitor, ReconnectPolicy, Watchdog
from rfx.teleop.transport import InprocTransport

# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


def test_watchdog_fires_on_timeout() -> None:
    """Watchdog fires on_timeout when not kicked within the timeout window."""
    fired = threading.Event()

    def on_timeout() -> None:
        fired.set()

    wd = Watchdog(timeout_s=0.1, on_timeout=on_timeout)
    try:
        # Don't kick; should fire within ~0.2s
        assert fired.wait(timeout=0.5), "Watchdog should have fired"
    finally:
        wd.stop()


def test_watchdog_does_not_fire_when_kicked() -> None:
    """Watchdog does NOT fire when kicked regularly."""
    fired = threading.Event()

    def on_timeout() -> None:
        fired.set()

    wd = Watchdog(timeout_s=0.4, on_timeout=on_timeout)
    try:
        # Keep a wide margin vs timeout to avoid scheduler-related flakes on CI.
        for _ in range(10):
            wd.kick()
            time.sleep(0.02)
        assert not fired.is_set(), "Watchdog should NOT have fired"
    finally:
        wd.stop()


def test_watchdog_stop_prevents_further_firing() -> None:
    """After stop(), the watchdog thread does not fire again."""
    count = 0
    lock = threading.Lock()

    def on_timeout() -> None:
        nonlocal count
        with lock:
            count += 1

    wd = Watchdog(timeout_s=0.05, on_timeout=on_timeout)
    time.sleep(0.15)  # let it fire at least once
    wd.stop()
    with lock:
        count_at_stop = count
    time.sleep(0.15)  # wait to confirm no more fires
    with lock:
        assert count == count_at_stop, "No more fires after stop()"


# ---------------------------------------------------------------------------
# ReconnectPolicy
# ---------------------------------------------------------------------------


def test_reconnect_policy_delays() -> None:
    """delays() yields correct exponential backoff sequence."""
    policy = ReconnectPolicy(max_retries=4, base_delay_s=1.0, backoff_factor=2.0, max_delay_s=10.0)
    delays = list(policy.delays())
    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_reconnect_policy_delays_capped_by_max() -> None:
    """Delays are capped at max_delay_s."""
    policy = ReconnectPolicy(max_retries=5, base_delay_s=1.0, backoff_factor=10.0, max_delay_s=5.0)
    delays = list(policy.delays())
    # 1.0, min(10.0, 5.0)=5.0, 5.0, 5.0, 5.0
    assert delays == [1.0, 5.0, 5.0, 5.0, 5.0]


def test_reconnect_policy_execute_success_first_try() -> None:
    """execute() returns True immediately if action succeeds on first attempt."""
    policy = ReconnectPolicy(max_retries=3, base_delay_s=0.01)
    attempts = []

    def action() -> bool:
        attempts.append(1)
        return True

    # Note: execute() still sleeps after each attempt including success
    # so we use very short base_delay_s
    result = policy.execute(action)
    assert result is True
    assert len(attempts) == 1


def test_reconnect_policy_execute_all_fail() -> None:
    """execute() returns False when all retries are exhausted."""
    policy = ReconnectPolicy(max_retries=3, base_delay_s=0.01, backoff_factor=1.0)

    def action() -> bool:
        return False

    result = policy.execute(action)
    assert result is False


def test_reconnect_policy_execute_success_on_second_try() -> None:
    """execute() returns True if action succeeds on a retry."""
    policy = ReconnectPolicy(max_retries=5, base_delay_s=0.01, backoff_factor=1.0)
    attempts = []

    def action() -> bool:
        attempts.append(1)
        return len(attempts) >= 2

    result = policy.execute(action)
    assert result is True
    assert len(attempts) == 2


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------


def test_health_monitor_publishes() -> None:
    """HealthMonitor publishes health on rfx/health/{node}."""
    import json

    transport = InprocTransport()
    sub = transport.subscribe("rfx/health/my_node")

    monitor = HealthMonitor(transport, "my_node", check_interval_s=0.05)
    try:
        env = sub.recv(timeout_s=0.5)
        assert env is not None
        data = json.loads(env.payload)
        assert data["node"] == "my_node"
        assert data["status"] == "ok"
        assert "timestamp_ns" in data
    finally:
        monitor.stop()
