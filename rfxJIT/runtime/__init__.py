"""Runtime components for rfxJIT."""

from rfxJIT.runtime.executor import (
    CompiledKernel,
    available_backends,
    compile_lowered_kernel,
    compile_opcode_kernel,
    execute_compiled_kernel,
    execute_lowered_kernel,
    resolve_backend_name,
)
from rfxJIT.runtime.interpreter import execute_kernel
from rfxJIT.runtime.opcode import (
    OPCODE_TAPE_VERSION,
    OpcodeInstruction,
    OpcodeKernel,
    dump_opcode_tape,
)
from rfxJIT.runtime.queue import KernelDispatchQueue
from rfxJIT.runtime.tinyjit import (
    TinyRfxGrad,
    TinyRfxJit,
    TinyRfxValueAndGrad,
    grad,
    jit_relu,
    value_and_grad,
)

__all__ = [
    "KernelDispatchQueue",
    "CompiledKernel",
    "TinyRfxGrad",
    "TinyRfxJit",
    "TinyRfxValueAndGrad",
    "available_backends",
    "compile_lowered_kernel",
    "compile_opcode_kernel",
    "OpcodeInstruction",
    "OpcodeKernel",
    "OPCODE_TAPE_VERSION",
    "dump_opcode_tape",
    "execute_compiled_kernel",
    "execute_kernel",
    "execute_lowered_kernel",
    "grad",
    "jit_relu",
    "resolve_backend_name",
    "value_and_grad",
]
