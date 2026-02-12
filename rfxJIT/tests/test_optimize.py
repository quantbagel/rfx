"""Tests for phase 2 optimization passes."""

from __future__ import annotations

import numpy as np

from rfxJIT.kernels.ir import KernelIR, KernelOp, OpCode, TensorSpec
from rfxJIT.kernels.optimize import (
    constant_fold_ir,
    eliminate_dead_ops,
    fuse_elementwise_chains,
    make_redundant_affine_relu_kernel,
    optimize_kernel_ir,
)
from rfxJIT.runtime.interpreter import execute_kernel


def _sample_inputs(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(99)
    return {
        "x": rng.standard_normal(size=shape, dtype=np.float32),
        "scale": rng.standard_normal(size=shape, dtype=np.float32),
        "bias": rng.standard_normal(size=shape, dtype=np.float32),
    }


def test_constant_fold_reduces_identity_ops() -> None:
    kernel = make_redundant_affine_relu_kernel(shape=(32,))
    folded = constant_fold_ir(kernel)

    assert len(folded.ops) < len(kernel.ops)

    inputs = _sample_inputs((32,))
    expected = execute_kernel(kernel, inputs)
    result = execute_kernel(folded, inputs)
    assert np.allclose(result, expected, atol=1e-6)


def test_dead_op_elimination_removes_unused_branch() -> None:
    kernel = KernelIR(
        name="dead_branch",
        shape=(8,),
        inputs=(TensorSpec("x", (8,)),),
        output=TensorSpec("y", (8,)),
        ops=[
            KernelOp(op=OpCode.CONST, out="c1", const_value=1.0),
            KernelOp(op=OpCode.ADD, out="y", inputs=("x", "c1")),
            KernelOp(op=OpCode.NEG, out="dead0", inputs=("x",)),
        ],
    )
    kernel.validate()

    optimized = eliminate_dead_ops(kernel)

    assert all(op.out != "dead0" for op in optimized.ops)


def test_fusion_collapses_double_relu_chain() -> None:
    kernel = KernelIR(
        name="relu_chain",
        shape=(8,),
        inputs=(TensorSpec("x", (8,)),),
        output=TensorSpec("y", (8,)),
        ops=[
            KernelOp(op=OpCode.RELU, out="relu0", inputs=("x",)),
            KernelOp(op=OpCode.RELU, out="y", inputs=("relu0",)),
        ],
    )
    kernel.validate()

    fused = fuse_elementwise_chains(kernel)
    optimized = eliminate_dead_ops(fused)

    assert len(optimized.ops) == 1
    assert optimized.ops[0].op == OpCode.RELU
    assert optimized.ops[0].out == "y"
    assert optimized.ops[0].inputs == ("x",)


def test_optimize_pipeline_preserves_output() -> None:
    kernel = make_redundant_affine_relu_kernel(shape=(64,))
    optimized = optimize_kernel_ir(kernel)
    inputs = _sample_inputs((64,))

    expected = execute_kernel(kernel, inputs)
    result = execute_kernel(optimized, inputs)

    assert len(optimized.ops) < len(kernel.ops)
    assert np.allclose(result, expected, atol=1e-6)
