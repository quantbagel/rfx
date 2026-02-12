"""IR-based autodiff transforms for elementwise kernels."""

from __future__ import annotations

from dataclasses import dataclass

from rfxJIT.kernels.ir import KernelIR, KernelOp, OpCode, TensorSpec


@dataclass(frozen=True)
class GradKernelBundle:
    """Gradient kernels keyed by input tensor name."""

    forward: KernelIR
    grads: dict[str, KernelIR]


class _GradBuilder:
    def __init__(self, kernel: KernelIR):
        self.kernel = kernel
        self.ops: list[KernelOp] = list(kernel.ops)
        self.taken_names = {spec.name for spec in kernel.inputs}
        self.taken_names.update(op.out for op in kernel.ops)
        self.taken_names.add(kernel.output.name)
        self._counter = 0

        self.grad_of: dict[str, str] = {}
        self.const_names: dict[float, str] = {}

        self.zero = self._const(0.0)
        self.one = self._const(1.0)

    def _fresh(self, prefix: str = "g") -> str:
        while True:
            name = f"{prefix}{self._counter}"
            self._counter += 1
            if name not in self.taken_names:
                self.taken_names.add(name)
                return name

    def _const(self, value: float) -> str:
        key = float(value)
        existing = self.const_names.get(key)
        if existing is not None:
            return existing
        name = self._fresh("c")
        self.ops.append(KernelOp(op=OpCode.CONST, out=name, const_value=key))
        self.const_names[key] = name
        return name

    def _unary(self, op: OpCode, src: str) -> str:
        out = self._fresh()
        self.ops.append(KernelOp(op=op, out=out, inputs=(src,)))
        return out

    def _binary(self, op: OpCode, lhs: str, rhs: str) -> str:
        out = self._fresh()
        self.ops.append(KernelOp(op=op, out=out, inputs=(lhs, rhs)))
        return out

    def _add_grad(self, value_name: str, contrib_name: str) -> None:
        existing = self.grad_of.get(value_name)
        if existing is None:
            self.grad_of[value_name] = contrib_name
            return
        self.grad_of[value_name] = self._binary(OpCode.ADD, existing, contrib_name)

    def seed_output(self) -> None:
        self.grad_of[self.kernel.output.name] = self.one

    def backward(self) -> None:
        self.seed_output()

        for op in reversed(self.kernel.ops):
            out_grad = self.grad_of.get(op.out)
            if out_grad is None:
                continue

            if op.op == OpCode.CONST:
                continue

            if op.op == OpCode.ADD:
                self._add_grad(op.inputs[0], out_grad)
                self._add_grad(op.inputs[1], out_grad)
                continue

            if op.op == OpCode.SUB:
                self._add_grad(op.inputs[0], out_grad)
                self._add_grad(op.inputs[1], self._unary(OpCode.NEG, out_grad))
                continue

            if op.op == OpCode.MUL:
                lhs = self._binary(OpCode.MUL, out_grad, op.inputs[1])
                rhs = self._binary(OpCode.MUL, out_grad, op.inputs[0])
                self._add_grad(op.inputs[0], lhs)
                self._add_grad(op.inputs[1], rhs)
                continue

            if op.op == OpCode.DIV:
                lhs = self._binary(OpCode.DIV, out_grad, op.inputs[1])
                y_sq = self._binary(OpCode.MUL, op.inputs[1], op.inputs[1])
                x_over_y_sq = self._binary(OpCode.DIV, op.inputs[0], y_sq)
                rhs = self._binary(OpCode.MUL, out_grad, self._unary(OpCode.NEG, x_over_y_sq))
                self._add_grad(op.inputs[0], lhs)
                self._add_grad(op.inputs[1], rhs)
                continue

            if op.op == OpCode.NEG:
                self._add_grad(op.inputs[0], self._unary(OpCode.NEG, out_grad))
                continue

            if op.op == OpCode.RELU:
                mask = self._unary(OpCode.STEP, op.inputs[0])
                self._add_grad(op.inputs[0], self._binary(OpCode.MUL, out_grad, mask))
                continue

            if op.op == OpCode.STEP:
                # Step is treated as non-differentiable (zero gradient almost everywhere).
                continue

            if op.op == OpCode.EXP:
                self._add_grad(op.inputs[0], self._binary(OpCode.MUL, out_grad, op.out))
                continue

            if op.op == OpCode.LOG:
                self._add_grad(op.inputs[0], self._binary(OpCode.DIV, out_grad, op.inputs[0]))
                continue

            raise ValueError(f"Unsupported op in autodiff: {op.op}")

    def grad_kernel_for_input(self, input_spec: TensorSpec) -> KernelIR:
        grad_name = self.grad_of.get(input_spec.name, self.zero)
        out_name = f"d_{input_spec.name}"
        ops = list(self.ops)
        if grad_name != out_name:
            ops.append(KernelOp(op=OpCode.ADD, out=out_name, inputs=(grad_name, self.zero)))

        kernel = KernelIR(
            name=f"{self.kernel.name}_grad_{input_spec.name}",
            shape=self.kernel.shape,
            inputs=self.kernel.inputs,
            output=TensorSpec(name=out_name, shape=input_spec.shape, dtype=input_spec.dtype),
            ops=ops,
        )
        kernel.validate()
        return kernel


def grad_kernels(
    kernel: KernelIR,
    *,
    wrt: tuple[str, ...] | None = None,
) -> GradKernelBundle:
    """
    Build gradient kernels for selected inputs.

    Gradients are for `sum(kernel_output)` with respect to each selected input.
    """

    kernel.validate()
    input_specs = {spec.name: spec for spec in kernel.inputs}

    if wrt is None:
        wrt_names = tuple(spec.name for spec in kernel.inputs)
    else:
        wrt_names = wrt
        missing = [name for name in wrt_names if name not in input_specs]
        if missing:
            raise ValueError(f"Unknown wrt inputs: {missing}")

    builder = _GradBuilder(kernel)
    builder.backward()

    grads = {name: builder.grad_kernel_for_input(input_specs[name]) for name in wrt_names}
    return GradKernelBundle(forward=kernel, grads=grads)
