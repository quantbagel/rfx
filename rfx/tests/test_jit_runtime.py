"""Tests for rfxJIT integration points in the rfx package."""

from __future__ import annotations

import numpy as np
import pytest

from rfx.decorators import policy
from rfx.jit import (
    PolicyJitRuntime,
    available_backends,
    grad,
    rfx_jit_backend,
    rfx_jit_strict,
    value_and_grad,
)
from rfxJIT.runtime.tinyjit import jit_relu


def _affine_relu(x):
    return jit_relu((x * 2.0) + 1.0)


def _affine_relu_grad_expected(x: np.ndarray) -> np.ndarray:
    return (x > -0.5).astype(x.dtype) * 2.0


def test_policy_jit_runtime_defaults_to_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RFX_JIT", raising=False)

    @policy(jit=True)
    def fn(x):
        return x

    runtime = fn._rfx_jit_runtime  # type: ignore[attr-defined]
    assert isinstance(runtime, PolicyJitRuntime)
    assert fn._rfx_jit_backend == "fallback"  # type: ignore[attr-defined]
    assert runtime.has_rfx_jit is False


def test_policy_jit_runtime_matches_eager_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RFX_JIT", "1")

    @policy(jit=True)
    def fn(x):
        return _affine_relu(x)

    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    eager = _affine_relu(x)
    got = fn(x)
    np.testing.assert_allclose(got, eager, atol=1e-6)


def test_value_and_grad_respects_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RFX_JIT", raising=False)
    with pytest.raises(RuntimeError, match="Set RFX_JIT=1"):
        value_and_grad(_affine_relu, argnums=0)

    with pytest.raises(RuntimeError, match="Set RFX_JIT=1"):
        grad(_affine_relu, argnums=0)


def test_value_and_grad_matches_manual_derivative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RFX_JIT", "1")

    vag = value_and_grad(_affine_relu, argnums=0)
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    value, dx = vag(x)
    expected_value = _affine_relu(x)
    expected_dx = _affine_relu_grad_expected(x)

    np.testing.assert_allclose(value, expected_value, atol=1e-6)
    np.testing.assert_allclose(dx, expected_dx, atol=1e-6)


def test_rfx_jit_backend_env_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RFX_JIT_BACKEND", "cuda")
    assert rfx_jit_backend() == "cuda"

    monkeypatch.setenv("RFX_JIT_BACKEND", "invalid")
    assert rfx_jit_backend() == "auto"


def test_rfx_jit_strict_env_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RFX_JIT_STRICT", "1")
    assert rfx_jit_strict() is True

    monkeypatch.setenv("RFX_JIT_STRICT", "no")
    assert rfx_jit_strict() is False


def test_available_backends_shape() -> None:
    backends = available_backends()
    assert set(backends.keys()) == {"cpu", "cuda", "metal"}
    assert backends["cpu"] is True


class _AlwaysFailJit:
    compile_count = 0

    def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("forced jit failure")

    def close(self) -> None:
        return None


def test_policy_runtime_falls_back_on_jit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RFX_JIT_STRICT", raising=False)

    runtime = PolicyJitRuntime(lambda x: x + 1, fallback=lambda x: x + 2, name="fallback_test")
    runtime._rfx_jit = _AlwaysFailJit()  # type: ignore[attr-defined]

    x = np.array([1.0, 2.0], dtype=np.float32)
    got = runtime(x)
    np.testing.assert_allclose(got, x + 2, atol=1e-6)
    assert runtime.has_rfx_jit is False
    assert runtime.rfx_jit_error is not None


def test_policy_runtime_strict_raises_on_jit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RFX_JIT_STRICT", "1")

    runtime = PolicyJitRuntime(lambda x: x + 1, fallback=lambda x: x + 2, name="strict_test")
    runtime._rfx_jit = _AlwaysFailJit()  # type: ignore[attr-defined]

    x = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match="forced jit failure"):
        runtime(x)
