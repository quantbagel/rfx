"""Kernel source emitters for CPU/CUDA/Metal style targets."""

from __future__ import annotations

from rfxJIT.kernels.ir import OpCode
from rfxJIT.kernels.lowering import LoweredKernel, LoweredOp


def _ctype_from_dtype(dtype: str) -> str:
    if dtype == "float64":
        return "double"
    return "float"


def _literal_from_const(value: float, *, ctype: str) -> str:
    if ctype == "double":
        return f"{value:.17g}"
    return f"{value:.9g}f"


def _expr_for_op(op: LoweredOp, slot_names: dict[int, str], *, ctype: str) -> str:
    if op.op == OpCode.CONST:
        assert op.const_value is not None
        return _literal_from_const(float(op.const_value), ctype=ctype)

    a0 = slot_names[op.input_slots[0]]
    if op.op == OpCode.NEG:
        return f"(-{a0})"
    if op.op == OpCode.RELU:
        return f"(({a0}) > 0 ? ({a0}) : 0)"
    if op.op == OpCode.STEP:
        return f"(({a0}) > 0 ? 1 : 0)"
    if op.op == OpCode.EXP:
        return f"exp({a0})"
    if op.op == OpCode.LOG:
        return f"log({a0})"

    a1 = slot_names[op.input_slots[1]]
    if op.op == OpCode.ADD:
        return f"({a0} + {a1})"
    if op.op == OpCode.SUB:
        return f"({a0} - {a1})"
    if op.op == OpCode.MUL:
        return f"({a0} * {a1})"
    if op.op == OpCode.DIV:
        return f"({a0} / {a1})"

    raise ValueError(f"Unsupported op for source emission: {op.op}")


def _slot_names(kernel: LoweredKernel) -> dict[int, str]:
    return {idx: f"v{idx}" for idx, _ in enumerate(kernel.value_names)}


def emit_cuda_kernel_source(kernel: LoweredKernel, *, fn_name: str = "rfxjit_kernel") -> str:
    """Emit a CUDA C kernel for a lowered elementwise kernel."""
    ctype = _ctype_from_dtype(kernel.dtype.value)
    slot_names = _slot_names(kernel)
    n_inputs = len(kernel.input_specs)

    params = [f"const {ctype}* in{i}" for i in range(n_inputs)]
    params.append(f"{ctype}* out")
    params.append("int n")
    lines = [f'extern "C" __global__ void {fn_name}({", ".join(params)}) {{']
    lines.append("  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);")
    lines.append("  if (idx >= n) return;")

    for i in range(n_inputs):
        lines.append(f"  {ctype} {slot_names[i]} = in{i}[idx];")

    for op in kernel.ops:
        out_name = slot_names[op.out_slot]
        expr = _expr_for_op(op, slot_names, ctype=ctype)
        lines.append(f"  {ctype} {out_name} = {expr};")

    lines.append(f"  out[idx] = {slot_names[kernel.output_slot]};")
    lines.append("}")
    return "\n".join(lines)


def emit_metal_kernel_source(kernel: LoweredKernel, *, fn_name: str = "rfxjit_kernel") -> str:
    """Emit a Metal Shading Language kernel for a lowered elementwise kernel."""
    ctype = _ctype_from_dtype(kernel.dtype.value)
    slot_names = _slot_names(kernel)
    n_inputs = len(kernel.input_specs)

    params = [f"const device {ctype}* in{i} [[buffer({i})]]" for i in range(n_inputs)]
    params.append(f"device {ctype}* out [[buffer({n_inputs})]]")
    params.append(f"constant uint& n [[buffer({n_inputs + 1})]]")
    params.append("uint gid [[thread_position_in_grid]]")

    lines = ["#include <metal_stdlib>", "using namespace metal;", ""]
    lines.append(f"kernel void {fn_name}({', '.join(params)}) {{")
    lines.append("  if (gid >= n) return;")

    for i in range(n_inputs):
        lines.append(f"  {ctype} {slot_names[i]} = in{i}[gid];")

    for op in kernel.ops:
        out_name = slot_names[op.out_slot]
        expr = _expr_for_op(op, slot_names, ctype=ctype)
        lines.append(f"  {ctype} {out_name} = {expr};")

    lines.append(f"  out[gid] = {slot_names[kernel.output_slot]};")
    lines.append("}")
    return "\n".join(lines)


def emit_pseudo_asm(kernel: LoweredKernel) -> str:
    """Emit a readable pseudo-assembly listing of lowered ops."""
    slot_names = _slot_names(kernel)
    lines = [f"; kernel {kernel.name}", f"; dtype {kernel.dtype.value} shape {kernel.shape}", ";"]

    for i, spec in enumerate(kernel.input_specs):
        lines.append(f"load {slot_names[i]}, {spec.name}")

    for op in kernel.ops:
        out_name = slot_names[op.out_slot]
        if op.op == OpCode.CONST:
            assert op.const_value is not None
            lines.append(f"mov {out_name}, {float(op.const_value):.17g}")
            continue

        in0 = slot_names[op.input_slots[0]]
        if op.op in {OpCode.NEG, OpCode.RELU, OpCode.STEP, OpCode.EXP, OpCode.LOG}:
            lines.append(f"{op.op.value} {out_name}, {in0}")
        else:
            in1 = slot_names[op.input_slots[1]]
            lines.append(f"{op.op.value} {out_name}, {in0}, {in1}")

    lines.append(f"store out, {slot_names[kernel.output_slot]}")
    return "\n".join(lines)


__all__ = [
    "emit_cuda_kernel_source",
    "emit_metal_kernel_source",
    "emit_pseudo_asm",
]
