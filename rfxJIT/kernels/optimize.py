"""Phase 2 optimization passes for rfxJIT kernel IR."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

from rfxJIT.kernels.ir import DType, KernelIR, KernelOp, OpCode, TensorSpec


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    resolved = name
    seen: set[str] = set()
    while resolved in aliases and resolved not in seen:
        seen.add(resolved)
        resolved = aliases[resolved]
    return resolved


def _const_key(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.17g}"


def _is_close(value: float, target: float) -> bool:
    return math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12)


def _unique_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        taken.add(base)
        return base

    idx = 1
    while True:
        candidate = f"{base}_{idx}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
        idx += 1


def _make_kernel_from_ops(kernel: KernelIR, ops: list[KernelOp]) -> KernelIR:
    optimized = KernelIR(
        name=kernel.name,
        shape=kernel.shape,
        inputs=kernel.inputs,
        output=kernel.output,
        ops=ops,
    )
    optimized.validate()
    return optimized


def _materialize_output(
    *,
    kernel: KernelIR,
    ops: list[KernelOp],
    aliases: dict[str, str],
    const_values: dict[str, float],
    const_name_by_key: dict[str, str],
    taken_names: set[str],
) -> None:
    output_name = kernel.output.name
    resolved_output = _resolve_alias(output_name, aliases)
    produced = {op.out for op in ops}

    if resolved_output == output_name and output_name in produced:
        return

    const_key = _const_key(0.0)
    zero_name = const_name_by_key.get(const_key)
    if zero_name is None or zero_name not in produced:
        zero_name = _unique_name("__opt_zero", taken_names)
        ops.append(KernelOp(op=OpCode.CONST, out=zero_name, const_value=0.0))
        const_values[zero_name] = 0.0
        const_name_by_key[const_key] = zero_name

    if resolved_output in const_values:
        ops.append(
            KernelOp(
                op=OpCode.CONST,
                out=output_name,
                const_value=const_values[resolved_output],
            )
        )
        return

    ops.append(
        KernelOp(
            op=OpCode.ADD,
            out=output_name,
            inputs=(resolved_output, zero_name),
        )
    )


def constant_fold_ir(kernel: KernelIR) -> KernelIR:
    """Fold constant expressions and simplify algebraic identities."""

    kernel.validate()

    aliases: dict[str, str] = {}
    const_values: dict[str, float] = {}
    const_name_by_key: dict[str, str] = {}
    taken_names = {spec.name for spec in kernel.inputs}
    taken_names.update(op.out for op in kernel.ops)
    taken_names.add(kernel.output.name)

    new_ops: list[KernelOp] = []

    def emit_const(out_name: str, value: float) -> None:
        const_val = float(value)
        new_ops.append(KernelOp(op=OpCode.CONST, out=out_name, const_value=const_val))
        const_values[out_name] = const_val
        const_name_by_key[_const_key(const_val)] = out_name

    def alias_output(out_name: str, target: str) -> None:
        resolved_target = _resolve_alias(target, aliases)
        aliases[out_name] = resolved_target
        if resolved_target in const_values:
            const_values[out_name] = const_values[resolved_target]
        else:
            const_values.pop(out_name, None)

    for op in kernel.ops:
        inputs = tuple(_resolve_alias(name, aliases) for name in op.inputs)
        folded = replace(op, inputs=inputs)
        const_values.pop(folded.out, None)

        if folded.op == OpCode.CONST:
            emit_const(folded.out, float(folded.const_value))
            continue

        input_consts = [const_values.get(name) for name in folded.inputs]

        if folded.op in {OpCode.NEG, OpCode.RELU, OpCode.STEP, OpCode.EXP, OpCode.LOG}:
            const_input = input_consts[0]
            if const_input is not None:
                if folded.op == OpCode.NEG:
                    emit_const(folded.out, -const_input)
                elif folded.op == OpCode.RELU:
                    emit_const(folded.out, max(const_input, 0.0))
                elif folded.op == OpCode.STEP:
                    emit_const(folded.out, 1.0 if const_input > 0.0 else 0.0)
                elif folded.op == OpCode.EXP:
                    emit_const(folded.out, float(math.exp(const_input)))
                elif folded.op == OpCode.LOG:
                    emit_const(folded.out, float(math.log(const_input)))
                continue

            new_ops.append(folded)
            continue

        left_name, right_name = folded.inputs
        left_const, right_const = input_consts

        if left_const is not None and right_const is not None:
            if folded.op == OpCode.ADD:
                emit_const(folded.out, left_const + right_const)
            elif folded.op == OpCode.SUB:
                emit_const(folded.out, left_const - right_const)
            elif folded.op == OpCode.MUL:
                emit_const(folded.out, left_const * right_const)
            elif folded.op == OpCode.DIV:
                emit_const(folded.out, left_const / right_const)
            continue

        if folded.op == OpCode.ADD:
            if left_const is not None and _is_close(left_const, 0.0):
                alias_output(folded.out, right_name)
                continue
            if right_const is not None and _is_close(right_const, 0.0):
                alias_output(folded.out, left_name)
                continue

        if folded.op == OpCode.SUB:
            if right_const is not None and _is_close(right_const, 0.0):
                alias_output(folded.out, left_name)
                continue
            if left_const is not None and _is_close(left_const, 0.0):
                new_ops.append(KernelOp(op=OpCode.NEG, out=folded.out, inputs=(right_name,)))
                continue

        if folded.op == OpCode.MUL:
            if (left_const is not None and _is_close(left_const, 0.0)) or (
                right_const is not None and _is_close(right_const, 0.0)
            ):
                emit_const(folded.out, 0.0)
                continue
            if left_const is not None and _is_close(left_const, 1.0):
                alias_output(folded.out, right_name)
                continue
            if right_const is not None and _is_close(right_const, 1.0):
                alias_output(folded.out, left_name)
                continue

        if folded.op == OpCode.DIV and right_const is not None and _is_close(right_const, 1.0):
            alias_output(folded.out, left_name)
            continue

        new_ops.append(folded)

    _materialize_output(
        kernel=kernel,
        ops=new_ops,
        aliases=aliases,
        const_values=const_values,
        const_name_by_key=const_name_by_key,
        taken_names=taken_names,
    )
    return _make_kernel_from_ops(kernel, new_ops)


def _single_const_operand(
    op: KernelOp,
    const_values: dict[str, float],
) -> tuple[str, float] | None:
    left_name, right_name = op.inputs
    left_const = const_values.get(left_name)
    right_const = const_values.get(right_name)

    if left_const is not None and right_const is None:
        return right_name, left_const
    if right_const is not None and left_const is None:
        return left_name, right_const
    return None


def fuse_elementwise_chains(kernel: KernelIR) -> KernelIR:
    """Fuse simple elementwise chains and remove redundant unary chains."""

    kernel.validate()

    use_count = Counter(input_name for op in kernel.ops for input_name in op.inputs)
    use_count[kernel.output.name] += 1

    aliases: dict[str, str] = {}
    const_values: dict[str, float] = {}
    const_name_by_key: dict[str, str] = {}
    taken_names = {spec.name for spec in kernel.inputs}
    taken_names.update(op.out for op in kernel.ops)
    taken_names.add(kernel.output.name)

    new_ops: list[KernelOp] = []
    op_by_out: dict[str, KernelOp] = {}

    def emit_const(out_name: str, value: float) -> None:
        const_val = float(value)
        op = KernelOp(op=OpCode.CONST, out=out_name, const_value=const_val)
        new_ops.append(op)
        op_by_out[out_name] = op
        const_values[out_name] = const_val
        const_name_by_key[_const_key(const_val)] = out_name

    def get_or_create_const(value: float) -> str:
        key = _const_key(value)
        existing = const_name_by_key.get(key)
        if existing is not None:
            return existing

        name = _unique_name("__opt_const", taken_names)
        emit_const(name, value)
        return name

    def alias_output(out_name: str, target: str) -> None:
        resolved_target = _resolve_alias(target, aliases)
        aliases[out_name] = resolved_target
        if resolved_target in const_values:
            const_values[out_name] = const_values[resolved_target]
        else:
            const_values.pop(out_name, None)

    for raw_op in kernel.ops:
        inputs = tuple(_resolve_alias(name, aliases) for name in raw_op.inputs)
        op = replace(raw_op, inputs=inputs)
        const_values.pop(op.out, None)

        if op.op == OpCode.CONST:
            emit_const(op.out, float(op.const_value))
            continue

        if op.op == OpCode.RELU and use_count[op.inputs[0]] == 1:
            parent = op_by_out.get(op.inputs[0])
            if parent is not None and parent.op == OpCode.RELU:
                op = replace(op, inputs=(parent.inputs[0],))

        if op.op == OpCode.NEG and use_count[op.inputs[0]] == 1:
            parent = op_by_out.get(op.inputs[0])
            if parent is not None and parent.op == OpCode.NEG:
                alias_output(op.out, parent.inputs[0])
                continue

        if op.op in {OpCode.ADD, OpCode.MUL}:
            split = _single_const_operand(op, const_values)
            if split is not None:
                inner_name, outer_const = split
                parent = op_by_out.get(inner_name)
                if parent is not None and parent.op == op.op and use_count[inner_name] == 1:
                    parent_split = _single_const_operand(parent, const_values)
                    if parent_split is not None:
                        base_name, inner_const = parent_split
                        if op.op == OpCode.ADD:
                            combined = inner_const + outer_const
                        else:
                            combined = inner_const * outer_const
                        const_name = get_or_create_const(combined)
                        op = KernelOp(
                            op=op.op,
                            out=op.out,
                            inputs=(base_name, const_name),
                        )

        new_ops.append(op)
        op_by_out[op.out] = op

    _materialize_output(
        kernel=kernel,
        ops=new_ops,
        aliases=aliases,
        const_values=const_values,
        const_name_by_key=const_name_by_key,
        taken_names=taken_names,
    )
    return _make_kernel_from_ops(kernel, new_ops)


def eliminate_dead_ops(kernel: KernelIR) -> KernelIR:
    """Remove ops that do not contribute to the kernel output."""

    kernel.validate()

    live_values = {kernel.output.name}
    kept_reversed: list[KernelOp] = []

    for op in reversed(kernel.ops):
        if op.out not in live_values:
            continue

        kept_reversed.append(op)
        live_values.discard(op.out)
        live_values.update(op.inputs)

    kept_ops = list(reversed(kept_reversed))
    return _make_kernel_from_ops(kernel, kept_ops)


def optimize_kernel_ir(kernel: KernelIR, *, max_rounds: int = 4) -> KernelIR:
    """Run phase 2 optimization pipeline to a fixed point."""

    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    optimized = kernel
    for _ in range(max_rounds):
        signature_before = tuple((op.op, op.out, op.inputs, op.const_value) for op in optimized.ops)

        candidate = constant_fold_ir(optimized)
        candidate = fuse_elementwise_chains(candidate)
        candidate = eliminate_dead_ops(candidate)
        candidate = constant_fold_ir(candidate)

        signature_after = tuple((op.op, op.out, op.inputs, op.const_value) for op in candidate.ops)
        optimized = candidate

        if signature_after == signature_before:
            break

    optimized.validate()
    return optimized


def make_redundant_affine_relu_kernel(
    shape: tuple[int, ...],
    *,
    name: str = "affine_relu_redundant",
    dtype: DType = DType.F32,
) -> KernelIR:
    """Build a redundant kernel equivalent to relu((x * scale) + bias)."""

    kernel = KernelIR(
        name=name,
        shape=shape,
        inputs=(
            TensorSpec(name="x", shape=shape, dtype=dtype),
            TensorSpec(name="scale", shape=shape, dtype=dtype),
            TensorSpec(name="bias", shape=shape, dtype=dtype),
        ),
        output=TensorSpec(name="y", shape=shape, dtype=dtype),
        ops=[
            KernelOp(op=OpCode.CONST, out="c0", const_value=0.0),
            KernelOp(op=OpCode.CONST, out="c1", const_value=1.0),
            KernelOp(op=OpCode.MUL, out="mul0", inputs=("x", "scale")),
            KernelOp(op=OpCode.ADD, out="add0", inputs=("mul0", "bias")),
            KernelOp(op=OpCode.RELU, out="relu0", inputs=("add0",)),
            KernelOp(op=OpCode.RELU, out="relu1", inputs=("relu0",)),
            KernelOp(op=OpCode.ADD, out="add1", inputs=("relu1", "c0")),
            KernelOp(op=OpCode.MUL, out="mul1", inputs=("add1", "c1")),
            KernelOp(op=OpCode.ADD, out="y", inputs=("mul1", "c0")),
        ],
    )
    kernel.validate()
    return kernel
