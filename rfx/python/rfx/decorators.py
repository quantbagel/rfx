"""
Decorators for control loops and neural policies

Provides decorators for defining control loops and tinygrad-based neural policies.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from tinygrad.engine.jit import TinyJit

    TINYGRAD_AVAILABLE = True
except ImportError:
    TinyJit = lambda x: x  # no-op if tinygrad not available
    TINYGRAD_AVAILABLE = False


def control_loop(
    rate_hz: float = 500.0,
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator to mark a function as a control loop callback.

    The decorated function should accept (state) and return a command.
    The loop will run at the specified rate.

    Args:
        rate_hz: Target loop rate in Hz (default: 500)
        name: Optional name for the loop

    Example:
        >>> @rfx.control_loop(rate_hz=500)
        >>> def balance_policy(state: rfx.Go2State) -> rfx.MotorCommands:
        ...     roll_error = state.imu.roll
        ...     return rfx.MotorCommands.from_positions({
        ...         "FL_hip": -0.5 * roll_error,
        ...         "FR_hip": -0.5 * roll_error,
        ...     })
        >>>
        >>> go2.run(balance_policy, timeout=30.0)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Attach metadata to the function
        wrapper._rfx_control_loop = True  # type: ignore
        wrapper._rfx_rate_hz = rate_hz  # type: ignore
        wrapper._rfx_name = name or func.__name__  # type: ignore
        # Backward-compatible metadata aliases.
        wrapper._pi_control_loop = True  # type: ignore
        wrapper._pi_rate_hz = rate_hz  # type: ignore
        wrapper._pi_name = name or func.__name__  # type: ignore

        return wrapper  # type: ignore

    return decorator


def policy(
    model: Optional[str] = None,
    jit: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to mark a function as a neural policy.

    When jit=True, the function will be JIT compiled using tinygrad's
    TinyJit for optimized inference after the first call.

    Args:
        model: Optional model path. If provided, raises NotImplementedError at runtime.
        jit: Whether to JIT compile the policy (default: False)

    Example:
        >>> from tinygrad import Tensor
        >>> import rfx
        >>>
        >>> # Simple inline policy with JIT
        >>> @rfx.policy(jit=True)
        >>> def walking_policy(obs: Tensor) -> Tensor:
        ...     # Your tinygrad forward pass here
        ...     return obs @ weights
        >>>
        >>> # Without JIT (for debugging)
        >>> @rfx.policy(jit=False)
        >>> def debug_policy(obs: Tensor) -> Tensor:
        ...     print(f"obs shape: {obs.shape}")
        ...     return obs @ weights

    For more complex policies, consider subclassing rfx.nn.Policy instead:

        >>> class WalkingPolicy(rfx.nn.Policy):
        ...     def __init__(self):
        ...         self.mlp = rfx.nn.MLP(48, 12)
        ...
        ...     def forward(self, obs: Tensor) -> Tensor:
        ...         return self.mlp(obs)
    """

    def decorator(func: F) -> F:
        if model is not None:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError(
                    "Neural network policies from model files are not implemented yet."
                )
        # Apply TinyJit if requested and available
        elif jit and TINYGRAD_AVAILABLE:
            jit_func = TinyJit(func)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return jit_func(*args, **kwargs)
        else:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

        # Apply TinyJit if requested and available
        # Attach metadata
        wrapper._rfx_policy = True  # type: ignore
        wrapper._rfx_jit = jit  # type: ignore
        wrapper._rfx_model = model  # type: ignore
        # Backward-compatible metadata aliases.
        wrapper._pi_policy = True  # type: ignore
        wrapper._pi_jit = jit  # type: ignore
        wrapper._pi_model = model  # type: ignore

        return wrapper  # type: ignore

    return decorator


class MotorCommands:
    """
    Motor commands to send to the robot.

    A convenience class for constructing motor command arrays from
    named positions, velocities, or torques.
    """

    def __init__(
        self,
        positions: Optional[dict[str, float]] = None,
        velocities: Optional[dict[str, float]] = None,
        torques: Optional[dict[str, float]] = None,
        kp: float = 20.0,
        kd: float = 0.5,
    ) -> None:
        self.positions = positions or {}
        self.velocities = velocities or {}
        self.torques = torques or {}
        self.kp = kp
        self.kd = kd

    @classmethod
    def from_positions(
        cls,
        positions: dict[str, float],
        kp: float = 20.0,
        kd: float = 0.5,
    ) -> "MotorCommands":
        """Create commands from named positions."""
        return cls(positions=positions, kp=kp, kd=kd)

    @classmethod
    def from_velocities(
        cls,
        velocities: dict[str, float],
        kd: float = 0.5,
    ) -> "MotorCommands":
        """Create commands from named velocities."""
        return cls(velocities=velocities, kd=kd)

    @classmethod
    def from_torques(
        cls,
        torques: dict[str, float],
    ) -> "MotorCommands":
        """Create commands from named torques."""
        return cls(torques=torques)

    def to_array(self, num_motors: int = 12) -> list[float]:
        """Convert to position array."""
        from . import motor_index_by_name

        result = [0.0] * num_motors
        for name, value in self.positions.items():
            idx = motor_index_by_name(name)
            if idx is not None:
                result[idx] = value
        return result

    def __repr__(self) -> str:
        parts = []
        if self.positions:
            parts.append(f"positions={self.positions}")
        if self.velocities:
            parts.append(f"velocities={self.velocities}")
        if self.torques:
            parts.append(f"torques={self.torques}")
        return f"MotorCommands({', '.join(parts)})"
