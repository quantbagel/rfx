"""tinyJIT-style runtime for traced rfxJIT kernels."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rfxJIT.kernels.ad import grad_kernels
from rfxJIT.kernels.lowering import LoweredKernel, lower_kernel_ir
from rfxJIT.kernels.optimize import optimize_kernel_ir
from rfxJIT.kernels.trace import KernelTracer, TracedTensor
from rfxJIT.runtime.executor import execute_lowered_kernel
from rfxJIT.runtime.queue import KernelDispatchQueue

Array = np.ndarray[Any, Any]
SignatureKey = tuple[tuple[tuple[int, ...], str], ...]
Argnums = int | tuple[int, ...]


def jit_relu(x: TracedTensor | Array) -> TracedTensor | Array:
    """Dispatchable relu helper for traced tensors and NumPy arrays."""
    if isinstance(x, TracedTensor):
        return x.relu()
    return np.maximum(x, 0.0)


def _normalize_argnums(argnums: Argnums, arity: int) -> tuple[int, ...]:
    if isinstance(argnums, int):
        argnums_tuple = (argnums,)
    else:
        argnums_tuple = tuple(argnums)
        if not argnums_tuple:
            raise ValueError("argnums must not be empty")

    normalized = []
    for idx in argnums_tuple:
        if idx < 0 or idx >= arity:
            raise ValueError(f"argnums index {idx} out of range for arity {arity}")
        normalized.append(idx)

    if len(set(normalized)) != len(normalized):
        raise ValueError("argnums contains duplicates")
    return tuple(normalized)


@dataclass
class _CompiledPlan:
    arg_names: tuple[str, ...]
    lowered: LoweredKernel
    queue: KernelDispatchQueue | None


@dataclass
class _CompiledGradPlan:
    arg_names: tuple[str, ...]
    argnums: tuple[int, ...]
    lowered_forward: LoweredKernel
    lowered_grads: dict[int, LoweredKernel]
    queue: KernelDispatchQueue | None


class TinyRfxJit:
    """
    tinyJIT-style wrapper for elementwise kernels.

    - first call for a new signature: trace -> optimize -> lower
    - subsequent calls: execute cached lowered kernel
    """

    def __init__(
        self,
        fn: Callable[..., TracedTensor | Array],
        *,
        name: str | None = None,
        optimize: bool = True,
        use_queue: bool = False,
    ):
        self._fn = fn
        self._name = name or fn.__name__
        self._optimize = optimize
        self._use_queue = use_queue
        self._plans: dict[SignatureKey, _CompiledPlan] = {}
        self._compile_count = 0

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params):
            raise TypeError("TinyRfxJit only supports positional parameters")
        self._param_names = tuple(p.name for p in params)

    @property
    def compile_count(self) -> int:
        return self._compile_count

    def clear_cache(self) -> None:
        self.close()
        self._plans.clear()
        self._compile_count = 0

    def close(self) -> None:
        for plan in self._plans.values():
            if plan.queue is not None:
                plan.queue.stop()

    def __call__(self, *args: Any) -> Array:
        if len(args) != len(self._param_names):
            raise TypeError(f"{self._name} expects {len(self._param_names)} args, got {len(args)}")

        arrays = tuple(np.asarray(arg) for arg in args)
        key = self._signature_key(arrays)

        plan = self._plans.get(key)
        if plan is None:
            plan = self._compile_plan(arrays)
            self._plans[key] = plan
            self._compile_count += 1

        named_inputs = dict(zip(plan.arg_names, arrays, strict=True))
        if plan.queue is not None:
            return plan.queue.submit(plan.lowered, named_inputs).result()
        return execute_lowered_kernel(plan.lowered, named_inputs)

    def _signature_key(self, arrays: tuple[Array, ...]) -> SignatureKey:
        return tuple((arr.shape, str(arr.dtype)) for arr in arrays)

    def _compile_plan(self, arrays: tuple[Array, ...]) -> _CompiledPlan:
        if not arrays:
            raise ValueError("TinyRfxJit requires at least one tensor input")

        shape = arrays[0].shape
        if not shape:
            raise ValueError("TinyRfxJit requires at least 1D tensors")
        if any(arr.shape != shape for arr in arrays):
            raise ValueError("All input tensors must share the same shape")

        tracer = KernelTracer.from_numpy_dtype(shape=shape, numpy_dtype=str(arrays[0].dtype))
        traced_args = tuple(tracer.input(name) for name in self._param_names)

        traced_output = self._fn(*traced_args)
        if not isinstance(traced_output, TracedTensor):
            raise TypeError(f"{self._name} must return a TracedTensor when traced")

        kernel = tracer.compile(traced_output, name=self._name)
        if self._optimize:
            kernel = optimize_kernel_ir(kernel)

        lowered = lower_kernel_ir(kernel)
        queue = KernelDispatchQueue() if self._use_queue else None
        return _CompiledPlan(
            arg_names=self._param_names,
            lowered=lowered,
            queue=queue,
        )


class TinyRfxValueAndGrad:
    """
    JAX-like value-and-grad transform over traced kernels.

    Gradients are computed for `sum(output)` over selected inputs.
    """

    def __init__(
        self,
        fn: Callable[..., TracedTensor | Array],
        *,
        argnums: Argnums = 0,
        name: str | None = None,
        optimize: bool = True,
        use_queue: bool = False,
    ):
        self._fn = fn
        self._name = name or fn.__name__
        self._argnums = argnums
        self._optimize = optimize
        self._use_queue = use_queue
        self._plans: dict[SignatureKey, _CompiledGradPlan] = {}
        self._compile_count = 0

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params):
            raise TypeError("TinyRfxValueAndGrad only supports positional parameters")
        self._param_names = tuple(p.name for p in params)
        self._argnums_tuple = _normalize_argnums(argnums, len(self._param_names))

    @property
    def compile_count(self) -> int:
        return self._compile_count

    def clear_cache(self) -> None:
        self.close()
        self._plans.clear()
        self._compile_count = 0

    def close(self) -> None:
        for plan in self._plans.values():
            if plan.queue is not None:
                plan.queue.stop()

    def __call__(self, *args: Any) -> tuple[Array, Array | tuple[Array, ...]]:
        if len(args) != len(self._param_names):
            raise TypeError(f"{self._name} expects {len(self._param_names)} args, got {len(args)}")

        arrays = tuple(np.asarray(arg) for arg in args)
        key = tuple((arr.shape, str(arr.dtype)) for arr in arrays)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._compile_plan(arrays)
            self._plans[key] = plan
            self._compile_count += 1

        named_inputs = dict(zip(plan.arg_names, arrays, strict=True))

        if plan.queue is not None:
            value = plan.queue.submit(plan.lowered_forward, named_inputs).result()
            grads = tuple(
                plan.queue.submit(plan.lowered_grads[idx], named_inputs).result()
                for idx in plan.argnums
            )
        else:
            value = execute_lowered_kernel(plan.lowered_forward, named_inputs)
            grads = tuple(
                execute_lowered_kernel(plan.lowered_grads[idx], named_inputs)
                for idx in plan.argnums
            )

        if len(grads) == 1:
            return value, grads[0]
        return value, grads

    def _compile_plan(self, arrays: tuple[Array, ...]) -> _CompiledGradPlan:
        if not arrays:
            raise ValueError("TinyRfxValueAndGrad requires at least one tensor input")

        shape = arrays[0].shape
        if not shape:
            raise ValueError("TinyRfxValueAndGrad requires at least 1D tensors")
        if any(arr.shape != shape for arr in arrays):
            raise ValueError("All input tensors must share the same shape")

        tracer = KernelTracer.from_numpy_dtype(shape=shape, numpy_dtype=str(arrays[0].dtype))
        traced_args = tuple(tracer.input(name) for name in self._param_names)

        traced_output = self._fn(*traced_args)
        if not isinstance(traced_output, TracedTensor):
            raise TypeError(f"{self._name} must return a TracedTensor when traced")

        forward = tracer.compile(traced_output, name=self._name)
        if self._optimize:
            forward = optimize_kernel_ir(forward)

        wrt_names = tuple(self._param_names[idx] for idx in self._argnums_tuple)
        grad_bundle = grad_kernels(forward, wrt=wrt_names)

        lowered_forward = lower_kernel_ir(grad_bundle.forward)
        lowered_grads: dict[int, LoweredKernel] = {}
        for idx in self._argnums_tuple:
            grad_kernel = grad_bundle.grads[self._param_names[idx]]
            if self._optimize:
                grad_kernel = optimize_kernel_ir(grad_kernel)
            lowered_grads[idx] = lower_kernel_ir(grad_kernel)

        queue = KernelDispatchQueue() if self._use_queue else None
        return _CompiledGradPlan(
            arg_names=self._param_names,
            argnums=self._argnums_tuple,
            lowered_forward=lowered_forward,
            lowered_grads=lowered_grads,
            queue=queue,
        )


class TinyRfxGrad:
    """JAX-like grad transform returning only gradients."""

    def __init__(
        self,
        fn: Callable[..., TracedTensor | Array],
        *,
        argnums: Argnums = 0,
        name: str | None = None,
        optimize: bool = True,
        use_queue: bool = False,
    ):
        self._inner = TinyRfxValueAndGrad(
            fn,
            argnums=argnums,
            name=name,
            optimize=optimize,
            use_queue=use_queue,
        )

    @property
    def compile_count(self) -> int:
        return self._inner.compile_count

    def clear_cache(self) -> None:
        self._inner.clear_cache()

    def close(self) -> None:
        self._inner.close()

    def __call__(self, *args: Any) -> Array | tuple[Array, ...]:
        _, grads = self._inner(*args)
        return grads


def value_and_grad(
    fn: Callable[..., TracedTensor | Array],
    *,
    argnums: Argnums = 0,
    name: str | None = None,
    optimize: bool = True,
    use_queue: bool = False,
) -> TinyRfxValueAndGrad:
    """Return a callable that computes `(value, grad)`."""

    return TinyRfxValueAndGrad(
        fn,
        argnums=argnums,
        name=name,
        optimize=optimize,
        use_queue=use_queue,
    )


def grad(
    fn: Callable[..., TracedTensor | Array],
    *,
    argnums: Argnums = 0,
    name: str | None = None,
    optimize: bool = True,
    use_queue: bool = False,
) -> TinyRfxGrad:
    """Return a callable that computes gradients only."""

    return TinyRfxGrad(
        fn,
        argnums=argnums,
        name=name,
        optimize=optimize,
        use_queue=use_queue,
    )
