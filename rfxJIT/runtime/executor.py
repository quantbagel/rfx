"""Executor for lowered phase 1 kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from rfxJIT.kernels.codegen import (
    emit_cuda_kernel_source,
    emit_metal_kernel_source,
    emit_pseudo_asm,
)
from rfxJIT.kernels.ir import OpCode, TensorSpec
from rfxJIT.kernels.lowering import LoweredKernel, LoweredOp


def _coerce_input(value: np.ndarray, spec: TensorSpec) -> np.ndarray:
    arr = np.asarray(value, dtype=spec.dtype.value)
    if arr.shape != spec.shape:
        raise ValueError(f"Input {spec.name!r} has shape {arr.shape}, expected {spec.shape}")
    return arr


def _execute_op_numpy(
    op: LoweredOp,
    values: list[np.ndarray | None],
    *,
    shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    if op.op == OpCode.CONST:
        assert op.const_value is not None
        return np.full(shape, op.const_value, dtype=dtype)

    args = [values[idx] for idx in op.input_slots]
    if any(arg is None for arg in args):
        raise RuntimeError(f"Operation {op.op.value} references unset value slots")

    a0 = args[0]
    assert a0 is not None

    if op.op == OpCode.NEG:
        out = -a0
    elif op.op == OpCode.RELU:
        out = np.maximum(a0, 0.0)
    elif op.op == OpCode.STEP:
        out = (a0 > 0).astype(dtype)
    elif op.op == OpCode.EXP:
        out = np.exp(a0)
    elif op.op == OpCode.LOG:
        out = np.log(a0)
    else:
        a1 = args[1]
        assert a1 is not None
        if op.op == OpCode.ADD:
            out = a0 + a1
        elif op.op == OpCode.SUB:
            out = a0 - a1
        elif op.op == OpCode.MUL:
            out = a0 * a1
        elif op.op == OpCode.DIV:
            out = a0 / a1
        else:
            raise ValueError(f"Unsupported op: {op.op}")

    return out.astype(dtype, copy=False)


def _execute_lowered_cpu(
    kernel: LoweredKernel,
    named_inputs: Mapping[str, np.ndarray],
) -> np.ndarray:
    input_name_to_slot = kernel.input_name_to_slot()
    expected_names = set(input_name_to_slot.keys())
    provided_names = set(named_inputs.keys())

    missing = expected_names - provided_names
    extra = provided_names - expected_names
    if missing:
        raise ValueError(f"Missing required inputs: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected inputs provided: {sorted(extra)}")

    values: list[np.ndarray | None] = [None] * len(kernel.value_names)
    for spec in kernel.input_specs:
        slot = input_name_to_slot[spec.name]
        values[slot] = _coerce_input(named_inputs[spec.name], spec)

    for op in kernel.ops:
        values[op.out_slot] = _execute_op_numpy(
            op,
            values,
            shape=kernel.shape,
            dtype=kernel.dtype.value,
        )

    output = values[kernel.output_slot]
    if output is None:
        raise RuntimeError(f"Output slot {kernel.output_slot} was never materialized")
    return np.asarray(output, dtype=kernel.dtype.value)


def _available_tinygrad_devices() -> set[str]:
    try:
        from tinygrad import Device
    except Exception:
        return set()

    try:
        return {str(name).upper() for name in Device.get_available_devices()}
    except Exception:
        return set()


def _tinygrad_dtype(dtype: str):
    from tinygrad import dtypes

    if dtype == "float64":
        return dtypes.float64
    return dtypes.float32


def _execute_lowered_tinygrad(
    kernel: LoweredKernel,
    named_inputs: Mapping[str, np.ndarray],
    *,
    device: str,
) -> np.ndarray:
    try:
        from tinygrad import Tensor
    except Exception as exc:
        raise RuntimeError(f"Backend {device.lower()} requires tinygrad runtime support.") from exc

    tdtype = _tinygrad_dtype(kernel.dtype.value)
    input_name_to_slot = kernel.input_name_to_slot()
    expected_names = set(input_name_to_slot.keys())
    provided_names = set(named_inputs.keys())

    missing = expected_names - provided_names
    extra = provided_names - expected_names
    if missing:
        raise ValueError(f"Missing required inputs: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected inputs provided: {sorted(extra)}")

    values: list[object | None] = [None] * len(kernel.value_names)
    for spec in kernel.input_specs:
        slot = input_name_to_slot[spec.name]
        arr = _coerce_input(named_inputs[spec.name], spec)
        values[slot] = Tensor(arr, device=device, dtype=tdtype)

    for op in kernel.ops:
        if op.op == OpCode.CONST:
            assert op.const_value is not None
            values[op.out_slot] = Tensor.full(
                kernel.shape,
                float(op.const_value),
                device=device,
                dtype=tdtype,
            )
            continue

        args = [values[idx] for idx in op.input_slots]
        if any(arg is None for arg in args):
            raise RuntimeError(f"Operation {op.op.value} references unset value slots")

        a0 = args[0]
        assert a0 is not None

        if op.op == OpCode.NEG:
            out = -a0
        elif op.op == OpCode.RELU:
            out = a0.relu()
        elif op.op == OpCode.STEP:
            out = (a0 > 0).where(1.0, 0.0)
        elif op.op == OpCode.EXP:
            out = a0.exp()
        elif op.op == OpCode.LOG:
            out = a0.log()
        else:
            a1 = args[1]
            assert a1 is not None
            if op.op == OpCode.ADD:
                out = a0 + a1
            elif op.op == OpCode.SUB:
                out = a0 - a1
            elif op.op == OpCode.MUL:
                out = a0 * a1
            elif op.op == OpCode.DIV:
                out = a0 / a1
            else:
                raise ValueError(f"Unsupported op: {op.op}")

        values[op.out_slot] = out.cast(tdtype)

    output = values[kernel.output_slot]
    if output is None:
        raise RuntimeError(f"Output slot {kernel.output_slot} was never materialized")

    return np.asarray(output.realize().numpy(), dtype=kernel.dtype.value)


@dataclass(frozen=True)
class CompiledKernel:
    """Backend-bound lowered kernel artifact."""

    kernel: LoweredKernel
    backend: str
    source: str | None = None
    pseudo_asm: str = ""


def available_backends() -> dict[str, bool]:
    """Return runtime backend availability for lowered-kernel execution."""
    devices = _available_tinygrad_devices()
    return {
        "cpu": True,
        "cuda": "CUDA" in devices,
        "metal": "METAL" in devices,
    }


def resolve_backend_name(backend: str) -> str:
    """Resolve and validate a backend name."""
    requested = backend.strip().lower()
    if requested not in {"auto", "cpu", "cuda", "metal"}:
        raise ValueError(f"Unsupported backend {backend!r}. Expected auto/cpu/cuda/metal.")

    avail = available_backends()
    if requested == "auto":
        for candidate in ("cuda", "metal", "cpu"):
            if avail[candidate]:
                return candidate
        return "cpu"

    if not avail[requested]:
        raise RuntimeError(
            f"Backend {requested!r} is unavailable. Available backends: "
            f"{', '.join(name for name, ok in avail.items() if ok)}"
        )
    return requested


def compile_lowered_kernel(
    kernel: LoweredKernel,
    *,
    backend: str = "cpu",
) -> CompiledKernel:
    """Compile a lowered kernel for a runtime backend target."""
    resolved = resolve_backend_name(backend)
    source = None
    if resolved == "cuda":
        source = emit_cuda_kernel_source(kernel, fn_name=kernel.name)
    elif resolved == "metal":
        source = emit_metal_kernel_source(kernel, fn_name=kernel.name)
    return CompiledKernel(
        kernel=kernel,
        backend=resolved,
        source=source,
        pseudo_asm=emit_pseudo_asm(kernel),
    )


def execute_compiled_kernel(
    compiled: CompiledKernel,
    named_inputs: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Execute a compiled lowered kernel."""
    if compiled.backend == "cpu":
        return _execute_lowered_cpu(compiled.kernel, named_inputs)
    if compiled.backend == "cuda":
        return _execute_lowered_tinygrad(compiled.kernel, named_inputs, device="CUDA")
    if compiled.backend == "metal":
        return _execute_lowered_tinygrad(compiled.kernel, named_inputs, device="METAL")
    raise ValueError(f"Unsupported compiled backend: {compiled.backend}")


def execute_lowered_kernel(
    kernel: LoweredKernel,
    named_inputs: Mapping[str, np.ndarray],
    *,
    backend: str = "cpu",
) -> np.ndarray:
    """Execute a lowered kernel against a selected backend."""
    compiled = compile_lowered_kernel(kernel, backend=backend)
    return execute_compiled_kernel(compiled, named_inputs)
