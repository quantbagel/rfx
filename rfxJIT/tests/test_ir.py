"""Tests for the phase 0 rfxJIT kernel IR and interpreter."""

from __future__ import annotations

import numpy as np
import pytest

from rfxJIT.kernels.ir import KernelIR, KernelOp, OpCode, TensorSpec, make_affine_relu_kernel
from rfxJIT.runtime.interpreter import execute_kernel


def test_validate_rejects_unknown_input_reference() -> None:
    kernel = KernelIR(
        name="bad",
        shape=(8,),
        inputs=(TensorSpec("x", (8,)),),
        output=TensorSpec("y", (8,)),
        ops=[KernelOp(op=OpCode.ADD, out="y", inputs=("x", "missing"))],
    )

    with pytest.raises(ValueError, match="unknown input"):
        kernel.validate()


def test_affine_relu_kernel_matches_baseline() -> None:
    kernel = make_affine_relu_kernel(shape=(16,))

    x = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    scale = np.linspace(0.5, 1.5, 16, dtype=np.float32)
    bias = np.full(16, 0.1, dtype=np.float32)

    expected = np.maximum((x * scale) + bias, 0.0)
    result = execute_kernel(
        kernel,
        {"x": x, "scale": scale, "bias": bias},
    )

    assert np.allclose(result, expected, atol=1e-6)


def test_execute_requires_exact_input_set() -> None:
    kernel = make_affine_relu_kernel(shape=(4,))

    x = np.ones(4, dtype=np.float32)
    scale = np.ones(4, dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError, match="Missing required inputs"):
        execute_kernel(kernel, {"x": x, "scale": scale})

    with pytest.raises(ValueError, match="Unexpected inputs"):
        execute_kernel(kernel, {"x": x, "scale": scale, "bias": bias, "extra": x})
