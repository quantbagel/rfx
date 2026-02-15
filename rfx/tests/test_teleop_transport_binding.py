"""Optional tests for native Rust transport Python bindings."""

from __future__ import annotations

import importlib

import pytest


def test_native_transport_binding_if_available() -> None:
    rfx_ext = importlib.import_module("rfx._rfx")
    if not hasattr(rfx_ext, "Transport"):
        pytest.skip("native Transport binding unavailable in current extension build")

    transport = rfx_ext.Transport()
    sub = transport.subscribe("teleop/**", 16)
    env = transport.publish("teleop/left/state", b"abc", '{"robot":"so101"}')
    got = sub.recv_timeout(0.2)

    assert env.key == "teleop/left/state"
    assert got is not None
    assert bytes(got.payload) == b"abc"
