"""Tests for tinyJIT-style runtime wrapper."""

from __future__ import annotations

import numpy as np

from rfxJIT.runtime.tinyjit import TinyRfxJit, jit_relu


def affine_relu_expr(x, scale, bias):
    return jit_relu((x * scale) + bias)


def shift_relu_expr(x):
    return jit_relu((x * 2.0) + 1.0)


def test_tinyjit_caches_after_first_compile() -> None:
    jit = TinyRfxJit(affine_relu_expr, name="affine_relu")

    rng = np.random.default_rng(123)
    x = rng.standard_normal(size=(32,), dtype=np.float32)
    scale = rng.standard_normal(size=(32,), dtype=np.float32)
    bias = rng.standard_normal(size=(32,), dtype=np.float32)

    expected = np.maximum((x * scale) + bias, 0.0)
    first = jit(x, scale, bias)
    second = jit(x, scale, bias)

    assert jit.compile_count == 1
    assert np.allclose(first, expected, atol=1e-6)
    assert np.allclose(second, expected, atol=1e-6)


def test_tinyjit_recompiles_on_shape_change() -> None:
    jit = TinyRfxJit(affine_relu_expr)

    rng = np.random.default_rng(5)
    x0 = rng.standard_normal(size=(16,), dtype=np.float32)
    s0 = rng.standard_normal(size=(16,), dtype=np.float32)
    b0 = rng.standard_normal(size=(16,), dtype=np.float32)

    x1 = rng.standard_normal(size=(24,), dtype=np.float32)
    s1 = rng.standard_normal(size=(24,), dtype=np.float32)
    b1 = rng.standard_normal(size=(24,), dtype=np.float32)

    jit(x0, s0, b0)
    jit(x1, s1, b1)

    assert jit.compile_count == 2


def test_tinyjit_supports_scalar_constants() -> None:
    jit = TinyRfxJit(shift_relu_expr, name="shift_relu")

    x = np.array([-2.0, -0.1, 0.0, 0.5], dtype=np.float32)
    expected = np.maximum((x * 2.0) + 1.0, 0.0)
    result = jit(x)

    assert jit.compile_count == 1
    assert np.allclose(result, expected, atol=1e-6)


def test_tinyjit_queue_mode_executes() -> None:
    jit = TinyRfxJit(affine_relu_expr, use_queue=True)

    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    s = np.linspace(0.5, 1.5, 8, dtype=np.float32)
    b = np.linspace(-0.2, 0.2, 8, dtype=np.float32)
    expected = np.maximum((x * s) + b, 0.0)

    result = jit(x, s, b)
    jit.close()
    assert np.allclose(result, expected, atol=1e-6)
