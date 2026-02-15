"""Tests for in-process teleop transport semantics."""

from __future__ import annotations

from rfx.teleop import InprocTransport


def test_publish_subscribe_with_key_patterns() -> None:
    transport = InprocTransport()
    sub_all = transport.subscribe("teleop/**")
    sub_left = transport.subscribe("teleop/left/*")

    env = transport.publish("teleop/left/state", {"q": [1, 2, 3]})
    got_all = sub_all.recv(timeout_s=0.1)
    got_left = sub_left.recv(timeout_s=0.1)

    assert got_all is not None
    assert got_left is not None
    assert got_all.sequence == env.sequence
    assert got_left.key == "teleop/left/state"


def test_zero_copy_bytes_payload_preserves_memoryview() -> None:
    transport = InprocTransport()
    sub = transport.subscribe("teleop/**")

    payload = bytearray(b"abc")
    env = transport.publish("teleop/frame/raw", payload)
    got = sub.recv(timeout_s=0.1)

    assert isinstance(env.payload, memoryview)
    assert got is not None
    assert isinstance(got.payload, memoryview)
    assert got.payload.obj is payload
