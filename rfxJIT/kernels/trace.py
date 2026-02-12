"""Tracing utilities to build KernelIR from Python expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat

from rfxJIT.kernels.ir import DType, KernelIR, KernelOp, OpCode, TensorSpec

Number = int | float | SupportsFloat


def _to_dtype(value: str) -> DType:
    if value == "float64":
        return DType.F64
    return DType.F32


def _const_key(value: float) -> str:
    return f"{float(value):.17g}"


@dataclass(frozen=True)
class TracedTensor:
    """Symbolic tensor reference owned by a KernelTracer."""

    tracer: KernelTracer
    name: str

    def _coerce(self, other: TracedTensor | Number) -> TracedTensor:
        return self.tracer.coerce(other)

    def __add__(self, other: TracedTensor | Number) -> TracedTensor:
        return self.tracer.binary(OpCode.ADD, self, self._coerce(other))

    def __radd__(self, other: TracedTensor | Number) -> TracedTensor:
        return self.tracer.binary(OpCode.ADD, self._coerce(other), self)

    def __sub__(self, other: TracedTensor | Number) -> TracedTensor:
        return self.tracer.binary(OpCode.SUB, self, self._coerce(other))

    def __rsub__(self, other: TracedTensor | Number) -> TracedTensor:
        return self.tracer.binary(OpCode.SUB, self._coerce(other), self)

    def __mul__(self, other: TracedTensor | Number) -> TracedTensor:
        return self.tracer.binary(OpCode.MUL, self, self._coerce(other))

    def __rmul__(self, other: TracedTensor | Number) -> TracedTensor:
        return self.tracer.binary(OpCode.MUL, self._coerce(other), self)

    def __truediv__(self, other: TracedTensor | Number) -> TracedTensor:
        return self.tracer.binary(OpCode.DIV, self, self._coerce(other))

    def __rtruediv__(self, other: TracedTensor | Number) -> TracedTensor:
        return self.tracer.binary(OpCode.DIV, self._coerce(other), self)

    def __neg__(self) -> TracedTensor:
        return self.tracer.unary(OpCode.NEG, self)

    def relu(self) -> TracedTensor:
        return self.tracer.unary(OpCode.RELU, self)

    def exp(self) -> TracedTensor:
        return self.tracer.unary(OpCode.EXP, self)

    def log(self) -> TracedTensor:
        return self.tracer.unary(OpCode.LOG, self)

    def step(self) -> TracedTensor:
        return self.tracer.unary(OpCode.STEP, self)


class KernelTracer:
    """Build a fixed-shape elementwise KernelIR by tracing tensor expressions."""

    def __init__(self, *, shape: tuple[int, ...], dtype: DType = DType.F32):
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"Invalid trace shape: {shape}")
        self.shape = shape
        self.dtype = dtype

        self._inputs: list[TensorSpec] = []
        self._input_names: set[str] = set()
        self._ops: list[KernelOp] = []
        self._const_name_by_key: dict[str, str] = {}
        self._counter = 0
        self._taken_names: set[str] = set()

    @classmethod
    def from_numpy_dtype(cls, *, shape: tuple[int, ...], numpy_dtype: str) -> KernelTracer:
        return cls(shape=shape, dtype=_to_dtype(numpy_dtype))

    def input(self, name: str) -> TracedTensor:
        if not name:
            raise ValueError("Input name must not be empty")
        if name in self._taken_names:
            raise ValueError(f"Input name {name!r} already used")
        spec = TensorSpec(name=name, shape=self.shape, dtype=self.dtype)
        self._inputs.append(spec)
        self._input_names.add(name)
        self._taken_names.add(name)
        return TracedTensor(self, name)

    def constant(self, value: Number) -> TracedTensor:
        const_value = float(value)
        key = _const_key(const_value)
        existing = self._const_name_by_key.get(key)
        if existing is not None:
            return TracedTensor(self, existing)

        name = self._fresh_name("c")
        self._ops.append(KernelOp(op=OpCode.CONST, out=name, const_value=const_value))
        self._const_name_by_key[key] = name
        return TracedTensor(self, name)

    def coerce(self, value: TracedTensor | Number) -> TracedTensor:
        if isinstance(value, TracedTensor):
            if value.tracer is not self:
                raise ValueError("Cannot mix tensors from different tracers")
            return value
        return self.constant(value)

    def unary(self, op: OpCode, value: TracedTensor) -> TracedTensor:
        if op not in {OpCode.NEG, OpCode.RELU, OpCode.STEP, OpCode.EXP, OpCode.LOG}:
            raise ValueError(f"Unsupported unary op for tracer: {op}")
        out = self._fresh_name("t")
        self._ops.append(KernelOp(op=op, out=out, inputs=(value.name,)))
        return TracedTensor(self, out)

    def binary(self, op: OpCode, lhs: TracedTensor, rhs: TracedTensor) -> TracedTensor:
        if op not in {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV}:
            raise ValueError(f"Unsupported binary op for tracer: {op}")
        out = self._fresh_name("t")
        self._ops.append(KernelOp(op=op, out=out, inputs=(lhs.name, rhs.name)))
        return TracedTensor(self, out)

    def compile(
        self,
        output: TracedTensor,
        *,
        name: str = "traced_kernel",
        output_name: str = "out",
    ) -> KernelIR:
        if output.tracer is not self:
            raise ValueError("Output tensor does not belong to this tracer")
        if output_name in self._input_names:
            output_name = self._fresh_name("out")

        ops = list(self._ops)
        if output.name != output_name:
            zero = self.constant(0.0)
            ops = list(self._ops)
            ops.append(KernelOp(op=OpCode.ADD, out=output_name, inputs=(output.name, zero.name)))

        kernel = KernelIR(
            name=name,
            shape=self.shape,
            inputs=tuple(self._inputs),
            output=TensorSpec(name=output_name, shape=self.shape, dtype=self.dtype),
            ops=ops,
        )
        kernel.validate()
        return kernel

    def _fresh_name(self, prefix: str) -> str:
        while True:
            candidate = f"{prefix}{self._counter}"
            self._counter += 1
            if candidate not in self._taken_names:
                self._taken_names.add(candidate)
                return candidate
