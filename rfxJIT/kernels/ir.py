"""Phase 0 kernel IR for elementwise rfxJIT kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DType(str, Enum):
    """Supported scalar dtypes for phase 0."""

    F32 = "float32"
    F64 = "float64"


class OpCode(str, Enum):
    """IR ops supported by the phase 0 interpreter."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    NEG = "neg"
    RELU = "relu"
    STEP = "step"
    EXP = "exp"
    LOG = "log"
    CONST = "const"


_UNARY_OPS = {OpCode.NEG, OpCode.RELU, OpCode.STEP, OpCode.EXP, OpCode.LOG}
_BINARY_OPS = {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV}


@dataclass(frozen=True)
class TensorSpec:
    """Tensor descriptor used by kernel inputs and outputs."""

    name: str
    shape: tuple[int, ...]
    dtype: DType = DType.F32

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TensorSpec.name must not be empty")
        if not self.shape:
            raise ValueError(f"TensorSpec {self.name!r} must have at least one dimension")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"TensorSpec {self.name!r} has non-positive dimension: {self.shape}")


@dataclass(frozen=True)
class KernelOp:
    """Single SSA-style op in the kernel graph."""

    op: OpCode
    out: str
    inputs: tuple[str, ...] = ()
    const_value: float | None = None

    def __post_init__(self) -> None:
        if not self.out:
            raise ValueError("KernelOp.out must not be empty")

        if self.op == OpCode.CONST:
            if len(self.inputs) != 0:
                raise ValueError("CONST op must not take inputs")
            if self.const_value is None:
                raise ValueError("CONST op must define const_value")
            return

        if self.const_value is not None:
            raise ValueError(f"{self.op.value} does not support const_value")

        expected_arity = 1 if self.op in _UNARY_OPS else 2
        if len(self.inputs) != expected_arity:
            raise ValueError(
                f"{self.op.value} expects {expected_arity} input(s), got {len(self.inputs)}"
            )


@dataclass
class KernelIR:
    """Phase 0 kernel: fixed-shape elementwise graph with typed tensors."""

    name: str
    shape: tuple[int, ...]
    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    ops: list[KernelOp] = field(default_factory=list)

    def validate(self) -> None:
        """Validate graph structure and tensor constraints."""
        if not self.name:
            raise ValueError("KernelIR.name must not be empty")
        if not self.shape:
            raise ValueError("KernelIR.shape must not be empty")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"KernelIR.shape has non-positive dimension: {self.shape}")

        input_names = [spec.name for spec in self.inputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError("KernelIR inputs contain duplicate tensor names")

        if self.output.shape != self.shape:
            raise ValueError(
                f"Output tensor shape {self.output.shape} does not match kernel shape {self.shape}"
            )
        if self.output.name in set(input_names):
            raise ValueError("Output tensor name must not collide with input tensor name")

        for spec in self.inputs:
            if spec.shape != self.shape:
                raise ValueError(
                    f"Input tensor {spec.name!r} shape {spec.shape} does not match {self.shape}"
                )
            if spec.dtype != self.output.dtype:
                raise ValueError("All tensors must share dtype in phase 0")

        if not self.ops:
            raise ValueError("KernelIR.ops must contain at least one operation")

        available_values = set(input_names)
        produced_values: set[str] = set()
        for idx, op in enumerate(self.ops):
            if op.out in available_values:
                raise ValueError(f"Op #{idx} writes to existing value {op.out!r}")
            for src in op.inputs:
                if src not in available_values:
                    raise ValueError(f"Op #{idx} references unknown input {src!r}")

            available_values.add(op.out)
            produced_values.add(op.out)

        if self.output.name not in produced_values:
            raise ValueError(
                f"Kernel output {self.output.name!r} is never produced by IR operations"
            )


def make_affine_relu_kernel(
    shape: tuple[int, ...],
    *,
    name: str = "affine_relu",
    x_name: str = "x",
    scale_name: str = "scale",
    bias_name: str = "bias",
    output_name: str = "y",
    dtype: DType = DType.F32,
) -> KernelIR:
    """Construct a simple reference kernel: relu((x * scale) + bias)."""

    kernel = KernelIR(
        name=name,
        shape=shape,
        inputs=(
            TensorSpec(name=x_name, shape=shape, dtype=dtype),
            TensorSpec(name=scale_name, shape=shape, dtype=dtype),
            TensorSpec(name=bias_name, shape=shape, dtype=dtype),
        ),
        output=TensorSpec(name=output_name, shape=shape, dtype=dtype),
        ops=[
            KernelOp(op=OpCode.MUL, out="mul0", inputs=(x_name, scale_name)),
            KernelOp(op=OpCode.ADD, out="add0", inputs=("mul0", bias_name)),
            KernelOp(op=OpCode.RELU, out=output_name, inputs=("add0",)),
        ],
    )
    kernel.validate()
    return kernel
