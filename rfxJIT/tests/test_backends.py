"""Tests for backend targets and source emission."""

from __future__ import annotations

import numpy as np
import pytest

from rfxJIT.kernels.codegen import (
    emit_cuda_kernel_source,
    emit_metal_kernel_source,
    emit_pseudo_asm,
)
from rfxJIT.kernels.ir import make_affine_relu_kernel
from rfxJIT.kernels.lowering import lower_kernel_ir
from rfxJIT.runtime.executor import (
    available_backends,
    compile_lowered_kernel,
    execute_compiled_kernel,
    resolve_backend_name,
)
from rfxJIT.runtime.tinyjit import TinyRfxJit, jit_relu


def _make_lowered():
    kernel = make_affine_relu_kernel(shape=(16,))
    return lower_kernel_ir(kernel)


def _affine_relu_expr(x, scale, bias):
    return jit_relu((x * scale) + bias)


def test_available_backends_includes_cpu() -> None:
    avail = available_backends()
    assert set(avail.keys()) == {"cpu", "cuda", "metal"}
    assert avail["cpu"] is True


def test_resolve_backend_auto_is_available() -> None:
    resolved = resolve_backend_name("auto")
    assert available_backends()[resolved] is True


def test_compile_cpu_and_execute_matches_numpy() -> None:
    lowered = _make_lowered()
    compiled = compile_lowered_kernel(lowered, backend="cpu")
    assert compiled.source is None
    assert "store out" in compiled.pseudo_asm

    rng = np.random.default_rng(0)
    x = rng.standard_normal((16,), dtype=np.float32)
    scale = rng.standard_normal((16,), dtype=np.float32)
    bias = rng.standard_normal((16,), dtype=np.float32)

    got = execute_compiled_kernel(compiled, {"x": x, "scale": scale, "bias": bias})
    expected = np.maximum((x * scale) + bias, 0.0)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_emit_sources_and_asm() -> None:
    lowered = _make_lowered()
    cuda_src = emit_cuda_kernel_source(lowered, fn_name="affine_relu")
    metal_src = emit_metal_kernel_source(lowered, fn_name="affine_relu")
    asm = emit_pseudo_asm(lowered)

    assert "__global__ void affine_relu" in cuda_src
    assert "kernel void affine_relu" in metal_src
    assert "store out" in asm


def test_compile_unavailable_gpu_backend_errors() -> None:
    lowered = _make_lowered()
    avail = available_backends()

    if not avail["cuda"]:
        with pytest.raises(RuntimeError, match="unavailable"):
            compile_lowered_kernel(lowered, backend="cuda")

    if not avail["metal"]:
        with pytest.raises(RuntimeError, match="unavailable"):
            compile_lowered_kernel(lowered, backend="metal")


def test_compile_available_gpu_backends_emit_source() -> None:
    lowered = _make_lowered()
    avail = available_backends()

    if avail["cuda"]:
        compiled_cuda = compile_lowered_kernel(lowered, backend="cuda")
        assert compiled_cuda.source is not None
        assert "__global__" in compiled_cuda.source
        assert "store out" in compiled_cuda.pseudo_asm

    if avail["metal"]:
        compiled_metal = compile_lowered_kernel(lowered, backend="metal")
        assert compiled_metal.source is not None
        assert "kernel void" in compiled_metal.source
        assert "store out" in compiled_metal.pseudo_asm


def test_tinyjit_auto_backend_executes() -> None:
    jit = TinyRfxJit(_affine_relu_expr, backend="auto")

    x = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    scale = np.linspace(0.5, 1.5, 16, dtype=np.float32)
    bias = np.linspace(-0.2, 0.2, 16, dtype=np.float32)

    got = jit(x, scale, bias)
    expected = np.maximum((x * scale) + bias, 0.0)
    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert jit.active_backends
