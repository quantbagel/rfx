"""Opcode tape contract for lowered rfxJIT kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rfxJIT.kernels.ir import DType, OpCode, TensorSpec
from rfxJIT.kernels.lowering import LoweredKernel, LoweredOp

OPCODE_TAPE_VERSION = 1


@dataclass(frozen=True)
class OpcodeInstruction:
    """Stable serialized instruction form for lowered kernel operations."""

    opcode: str
    out_slot: int
    input_slots: tuple[int, ...] = ()
    const_value: float | None = None

    @classmethod
    def from_lowered(cls, op: LoweredOp) -> OpcodeInstruction:
        return cls(
            opcode=op.op.value,
            out_slot=op.out_slot,
            input_slots=op.input_slots,
            const_value=op.const_value,
        )

    def to_lowered(self) -> LoweredOp:
        return LoweredOp(
            op=OpCode(self.opcode),
            out_slot=int(self.out_slot),
            input_slots=tuple(int(v) for v in self.input_slots),
            const_value=None if self.const_value is None else float(self.const_value),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode,
            "out_slot": int(self.out_slot),
            "input_slots": [int(v) for v in self.input_slots],
            "const_value": None if self.const_value is None else float(self.const_value),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OpcodeInstruction:
        return cls(
            opcode=str(payload["opcode"]),
            out_slot=int(payload["out_slot"]),
            input_slots=tuple(int(v) for v in payload.get("input_slots", ())),
            const_value=(
                None if payload.get("const_value") is None else float(payload["const_value"])
            ),
        )


@dataclass(frozen=True)
class OpcodeKernel:
    """Stable opcode tape artifact used by runtime backends and debug tooling."""

    version: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    input_specs: tuple[TensorSpec, ...]
    output_slot: int
    value_names: tuple[str, ...]
    instructions: tuple[OpcodeInstruction, ...]

    @classmethod
    def from_lowered(cls, kernel: LoweredKernel) -> OpcodeKernel:
        return cls(
            version=OPCODE_TAPE_VERSION,
            name=kernel.name,
            shape=kernel.shape,
            dtype=kernel.dtype.value,
            input_specs=kernel.input_specs,
            output_slot=kernel.output_slot,
            value_names=kernel.value_names,
            instructions=tuple(OpcodeInstruction.from_lowered(op) for op in kernel.ops),
        )

    def to_lowered(self) -> LoweredKernel:
        return LoweredKernel(
            name=self.name,
            shape=tuple(int(v) for v in self.shape),
            dtype=DType(self.dtype),
            input_specs=self.input_specs,
            output_slot=int(self.output_slot),
            value_names=tuple(self.value_names),
            ops=tuple(inst.to_lowered() for inst in self.instructions),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "name": self.name,
            "shape": [int(v) for v in self.shape],
            "dtype": self.dtype,
            "input_specs": [
                {
                    "name": spec.name,
                    "shape": [int(v) for v in spec.shape],
                    "dtype": spec.dtype.value,
                }
                for spec in self.input_specs
            ],
            "output_slot": int(self.output_slot),
            "value_names": list(self.value_names),
            "instructions": [inst.to_dict() for inst in self.instructions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OpcodeKernel:
        input_specs = tuple(
            TensorSpec(
                name=str(spec["name"]),
                shape=tuple(int(v) for v in spec["shape"]),
                dtype=DType(str(spec["dtype"])),
            )
            for spec in payload["input_specs"]
        )
        instructions = tuple(OpcodeInstruction.from_dict(inst) for inst in payload["instructions"])
        return cls(
            version=int(payload["version"]),
            name=str(payload["name"]),
            shape=tuple(int(v) for v in payload["shape"]),
            dtype=str(payload["dtype"]),
            input_specs=input_specs,
            output_slot=int(payload["output_slot"]),
            value_names=tuple(str(v) for v in payload["value_names"]),
            instructions=instructions,
        )


def dump_opcode_tape(kernel: LoweredKernel) -> dict[str, Any]:
    """Convenience helper to dump a lowered kernel as a serializable opcode tape."""
    return OpcodeKernel.from_lowered(kernel).to_dict()
