"""
rfxJIT integration helpers for rfx runtime paths.

Enable with:
    RFX_JIT=1
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

RFX_JIT_ENV_VAR = "RFX_JIT"
RFX_JIT_BACKEND_ENV_VAR = "RFX_JIT_BACKEND"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def rfx_jit_enabled() -> bool:
    """Return True when rfxJIT-backed execution is enabled."""
    return os.getenv(RFX_JIT_ENV_VAR, "").strip().lower() in _TRUE_VALUES


def rfx_jit_backend() -> str:
    """Requested rfxJIT backend: auto/cpu/cuda/metal."""
    value = os.getenv(RFX_JIT_BACKEND_ENV_VAR, "auto").strip().lower()
    if value not in {"auto", "cpu", "cuda", "metal"}:
        return "auto"
    return value


def _numpy_tensor_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if kwargs:
        return False
    if not args:
        return False

    for arg in args:
        if not isinstance(arg, np.ndarray):
            return False
        if arg.ndim == 0:
            return False
    return True


class PolicyJitRuntime:
    """
    Dispatch policy execution between fallback JIT and rfxJIT.

    - fallback path: tinygrad TinyJit (or plain function)
    - rfxJIT path: selected only when RFX_JIT=1 and args are NumPy tensors
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        fallback: Callable[..., Any] | None = None,
        name: str | None = None,
    ):
        self._fn = fn
        self._fallback = fallback or fn
        self._name = name or fn.__name__
        self._rfx_jit: Any = None
        self._rfx_jit_error: Exception | None = None

        if rfx_jit_enabled():
            try:
                from rfxJIT.runtime.tinyjit import TinyRfxJit

                self._rfx_jit = TinyRfxJit(fn, name=self._name, backend=rfx_jit_backend())
            except Exception as exc:  # pragma: no cover - defensive import/trace fallback
                self._rfx_jit_error = exc

    @property
    def backend(self) -> str:
        if self._rfx_jit is not None:
            active = getattr(self._rfx_jit, "active_backends", ())
            if active:
                return "+".join(active)
            return "auto"
        return "fallback"

    @property
    def has_rfx_jit(self) -> bool:
        return self._rfx_jit is not None

    @property
    def compile_count(self) -> int:
        if self._rfx_jit is None:
            return 0
        return int(self._rfx_jit.compile_count)

    @property
    def rfx_jit_error(self) -> Exception | None:
        return self._rfx_jit_error

    def close(self) -> None:
        if self._rfx_jit is not None:
            self._rfx_jit.close()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._rfx_jit is not None and _numpy_tensor_args(args, kwargs):
            return self._rfx_jit(*args)
        return self._fallback(*args, **kwargs)


def value_and_grad(
    fn: Callable[..., Any],
    *,
    argnums: int | tuple[int, ...] = 0,
    name: str | None = None,
):
    """rfxJIT value_and_grad transform (requires RFX_JIT=1)."""
    if not rfx_jit_enabled():
        raise RuntimeError("rfxJIT transforms are disabled. Set RFX_JIT=1 to enable.")

    try:
        from rfxJIT import value_and_grad as _value_and_grad
    except Exception as exc:  # pragma: no cover - defensive import fallback
        raise RuntimeError("rfxJIT runtime is unavailable in this environment.") from exc

    return _value_and_grad(fn, argnums=argnums, name=name)


def grad(
    fn: Callable[..., Any],
    *,
    argnums: int | tuple[int, ...] = 0,
    name: str | None = None,
):
    """rfxJIT grad transform (requires RFX_JIT=1)."""
    if not rfx_jit_enabled():
        raise RuntimeError("rfxJIT transforms are disabled. Set RFX_JIT=1 to enable.")

    try:
        from rfxJIT import grad as _grad
    except Exception as exc:  # pragma: no cover - defensive import fallback
        raise RuntimeError("rfxJIT runtime is unavailable in this environment.") from exc

    return _grad(fn, argnums=argnums, name=name)


__all__ = [
    "RFX_JIT_BACKEND_ENV_VAR",
    "RFX_JIT_ENV_VAR",
    "PolicyJitRuntime",
    "grad",
    "rfx_jit_backend",
    "rfx_jit_enabled",
    "value_and_grad",
]
