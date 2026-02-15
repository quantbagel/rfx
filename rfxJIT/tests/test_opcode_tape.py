"""Tests for opcode tape serialization and execution."""

from __future__ import annotations

import numpy as np

from rfxJIT.kernels.ir import make_affine_relu_kernel
from rfxJIT.kernels.lowering import lower_kernel_ir
from rfxJIT.runtime.executor import compile_opcode_kernel, execute_compiled_kernel
from rfxJIT.runtime.opcode import OpcodeKernel, dump_opcode_tape
from rfxJIT.runtime.tinyjit import TinyRfxJit, jit_relu


def _make_lowered():
    kernel = make_affine_relu_kernel(shape=(16,))
    return lower_kernel_ir(kernel)


def test_opcode_kernel_roundtrip_lowered_kernel() -> None:
    lowered = _make_lowered()
    tape = OpcodeKernel.from_lowered(lowered)
    rebuilt = tape.to_lowered()

    assert rebuilt.name == lowered.name
    assert rebuilt.shape == lowered.shape
    assert rebuilt.dtype == lowered.dtype
    assert rebuilt.input_specs == lowered.input_specs
    assert rebuilt.value_names == lowered.value_names
    assert rebuilt.ops == lowered.ops


def test_opcode_kernel_dict_roundtrip() -> None:
    lowered = _make_lowered()
    payload = dump_opcode_tape(lowered)
    tape = OpcodeKernel.from_dict(payload)

    assert tape.to_dict() == payload


def test_compile_opcode_kernel_executes_like_numpy() -> None:
    lowered = _make_lowered()
    tape = OpcodeKernel.from_lowered(lowered)
    compiled = compile_opcode_kernel(tape, backend="cpu")

    rng = np.random.default_rng(7)
    x = rng.standard_normal((16,), dtype=np.float32)
    scale = rng.standard_normal((16,), dtype=np.float32)
    bias = rng.standard_normal((16,), dtype=np.float32)
    got = execute_compiled_kernel(compiled, {"x": x, "scale": scale, "bias": bias})
    expected = np.maximum((x * scale) + bias, 0.0)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_tinyjit_exposes_cached_opcode_tapes() -> None:
    def affine_relu_expr(x, scale, bias):
        return jit_relu((x * scale) + bias)

    jit = TinyRfxJit(affine_relu_expr, backend="cpu")
    x = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    scale = np.linspace(0.5, 1.5, 16, dtype=np.float32)
    bias = np.linspace(-0.2, 0.2, 16, dtype=np.float32)
    _ = jit(x, scale, bias)

    tapes = jit.cached_opcode_tapes()
    assert tapes
    only_tape = next(iter(tapes.values()))
    assert only_tape["instructions"]
