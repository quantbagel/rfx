"""Tests for IR-based grad and value_and_grad transforms."""

from __future__ import annotations

import numpy as np

from rfxJIT.runtime.tinyjit import grad, jit_relu, value_and_grad


def affine_relu_expr(x, scale, bias):
    return jit_relu((x * scale) + bias)


def two_input_expr(x, y):
    return (x * y) + x


def _relu_mask(values: np.ndarray) -> np.ndarray:
    return (values > 0).astype(values.dtype)


def test_value_and_grad_matches_manual_affine_relu_derivatives() -> None:
    vag = value_and_grad(affine_relu_expr, argnums=(0, 1, 2))

    x = np.array([-2.0, -0.5, 0.25, 1.5], dtype=np.float32)
    scale = np.array([0.5, 1.0, -2.0, 3.0], dtype=np.float32)
    bias = np.array([0.1, -0.1, 0.2, -0.3], dtype=np.float32)

    value, (dx, dscale, dbias) = vag(x, scale, bias)
    pre = (x * scale) + bias
    mask = _relu_mask(pre)

    expected_value = np.maximum(pre, 0.0)
    expected_dx = mask * scale
    expected_dscale = mask * x
    expected_dbias = mask

    assert np.allclose(value, expected_value, atol=1e-6)
    assert np.allclose(dx, expected_dx, atol=1e-6)
    assert np.allclose(dscale, expected_dscale, atol=1e-6)
    assert np.allclose(dbias, expected_dbias, atol=1e-6)
    assert vag.compile_count == 1


def test_grad_single_arg_returns_array_and_caches() -> None:
    g = grad(two_input_expr, argnums=0)

    x = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    y = np.linspace(0.1, 2.0, 16, dtype=np.float32)

    first = g(x, y)
    second = g(x, y)

    expected = y + 1.0
    assert np.allclose(first, expected, atol=1e-6)
    assert np.allclose(second, expected, atol=1e-6)
    assert g.compile_count == 1


def test_grad_recompiles_on_signature_change() -> None:
    g = grad(two_input_expr, argnums=(0, 1))

    x0 = np.ones(8, dtype=np.float32)
    y0 = np.ones(8, dtype=np.float32)
    x1 = np.ones(12, dtype=np.float32)
    y1 = np.ones(12, dtype=np.float32)

    g(x0, y0)
    g(x1, y1)

    assert g.compile_count == 2
