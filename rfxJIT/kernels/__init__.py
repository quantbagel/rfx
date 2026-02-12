"""Kernel IR definitions for rfxJIT."""

from rfxJIT.kernels.ad import GradKernelBundle, grad_kernels
from rfxJIT.kernels.codegen import (
    emit_cuda_kernel_source,
    emit_metal_kernel_source,
    emit_pseudo_asm,
)
from rfxJIT.kernels.ir import DType, KernelIR, KernelOp, OpCode, TensorSpec, make_affine_relu_kernel
from rfxJIT.kernels.lowering import LoweredKernel, LoweredOp, lower_kernel_ir
from rfxJIT.kernels.optimize import (
    constant_fold_ir,
    eliminate_dead_ops,
    fuse_elementwise_chains,
    make_redundant_affine_relu_kernel,
    optimize_kernel_ir,
)
from rfxJIT.kernels.trace import KernelTracer, TracedTensor

__all__ = [
    "DType",
    "GradKernelBundle",
    "KernelIR",
    "KernelOp",
    "KernelTracer",
    "LoweredKernel",
    "LoweredOp",
    "OpCode",
    "TracedTensor",
    "TensorSpec",
    "grad_kernels",
    "emit_cuda_kernel_source",
    "emit_metal_kernel_source",
    "emit_pseudo_asm",
    "constant_fold_ir",
    "eliminate_dead_ops",
    "fuse_elementwise_chains",
    "lower_kernel_ir",
    "make_redundant_affine_relu_kernel",
    "make_affine_relu_kernel",
    "optimize_kernel_ir",
]
