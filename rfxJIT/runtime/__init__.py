"""Runtime prototypes for rfxJIT."""

from rfxJIT.runtime.executor import execute_lowered_kernel
from rfxJIT.runtime.interpreter import execute_kernel
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
    "TinyRfxGrad",
    "TinyRfxJit",
    "TinyRfxValueAndGrad",
    "execute_kernel",
    "execute_lowered_kernel",
    "grad",
    "jit_relu",
    "value_and_grad",
]
