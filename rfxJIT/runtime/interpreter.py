"""Reference interpreter for phase 0 rfxJIT IR."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rfxJIT.kernels.ir import KernelIR, OpCode, TensorSpec


def _coerce_input(value: np.ndarray, spec: TensorSpec) -> np.ndarray:
    arr = np.asarray(value, dtype=spec.dtype.value)
    if arr.shape != spec.shape:
        raise ValueError(f"Input {spec.name!r} has shape {arr.shape}, expected {spec.shape}")
    return arr


def execute_kernel(kernel: KernelIR, named_inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Execute a validated kernel with NumPy arrays."""

    kernel.validate()

    expected_names = {spec.name for spec in kernel.inputs}
    provided_names = set(named_inputs.keys())

    missing = expected_names - provided_names
    extra = provided_names - expected_names
    if missing:
        raise ValueError(f"Missing required inputs: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected inputs provided: {sorted(extra)}")

    values: dict[str, np.ndarray] = {}
    for spec in kernel.inputs:
        values[spec.name] = _coerce_input(named_inputs[spec.name], spec)

    for op in kernel.ops:
        if op.op == OpCode.CONST:
            values[op.out] = np.full(
                kernel.shape,
                float(op.const_value),
                dtype=kernel.output.dtype.value,
            )
            continue

        args = [values[name] for name in op.inputs]
        if op.op == OpCode.ADD:
            out = args[0] + args[1]
        elif op.op == OpCode.SUB:
            out = args[0] - args[1]
        elif op.op == OpCode.MUL:
            out = args[0] * args[1]
        elif op.op == OpCode.DIV:
            out = args[0] / args[1]
        elif op.op == OpCode.NEG:
            out = -args[0]
        elif op.op == OpCode.RELU:
            out = np.maximum(args[0], 0.0)
        elif op.op == OpCode.EXP:
            out = np.exp(args[0])
        elif op.op == OpCode.LOG:
            out = np.log(args[0])
        else:
            raise ValueError(f"Unsupported op: {op.op}")

        values[op.out] = out.astype(kernel.output.dtype.value, copy=False)

    result = values[kernel.output.name]
    return np.asarray(result, dtype=kernel.output.dtype.value)
