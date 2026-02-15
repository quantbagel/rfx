"""Reference interpreter for phase 0 rfxJIT IR."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rfxJIT.kernels.ir import KernelIR, OpCode
from rfxJIT.runtime.core_exec import coerce_named_inputs, execute_numpy_op


def execute_kernel(kernel: KernelIR, named_inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Execute a validated kernel with NumPy arrays."""

    kernel.validate()

    values = coerce_named_inputs(kernel.inputs, named_inputs)
    out_dtype = kernel.output.dtype.value

    for op in kernel.ops:
        if op.op == OpCode.CONST:
            args: tuple[np.ndarray, ...] = ()
        else:
            args = tuple(values[name] for name in op.inputs)
        values[op.out] = execute_numpy_op(
            op=op.op,
            args=args,
            shape=kernel.shape,
            dtype=out_dtype,
            const_value=op.const_value,
        )

    result = values[kernel.output.name]
    return np.asarray(result, dtype=out_dtype)
