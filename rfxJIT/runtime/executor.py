"""Executor for lowered phase 1 kernels."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rfxJIT.kernels.ir import OpCode, TensorSpec
from rfxJIT.kernels.lowering import LoweredKernel, LoweredOp


def _coerce_input(value: np.ndarray, spec: TensorSpec) -> np.ndarray:
    arr = np.asarray(value, dtype=spec.dtype.value)
    if arr.shape != spec.shape:
        raise ValueError(f"Input {spec.name!r} has shape {arr.shape}, expected {spec.shape}")
    return arr


def _execute_op(
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


def execute_lowered_kernel(
    kernel: LoweredKernel,
    named_inputs: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Execute a lowered kernel using NumPy arrays."""

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
        values[op.out_slot] = _execute_op(
            op,
            values,
            shape=kernel.shape,
            dtype=kernel.dtype.value,
        )

    output = values[kernel.output_slot]
    if output is None:
        raise RuntimeError(f"Output slot {kernel.output_slot} was never materialized")
    return np.asarray(output, dtype=kernel.dtype.value)
