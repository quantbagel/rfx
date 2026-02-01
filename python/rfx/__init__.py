"""
rfx: A tinygrad-inspired robotics framework for Unitree Go2

A minimal, Python-first robotics framework with Rust performance.
No ROS2 complexity, no DDS boilerplate - just clean APIs for physical intelligence experiments.

Example:
    >>> import rfx
    >>> # Quick training with tinygrad
    >>> from rfx.nn import go2_mlp
    >>> from rfx.rl import PPOTrainer
    >>> from rfx.envs import Go2Env
    >>>
    >>> env = Go2Env(sim=True)
    >>> policy = go2_mlp()
    >>> trainer = PPOTrainer(policy)
    >>>
    >>> # Or connect to real hardware
    >>> go2 = rfx.Go2.connect("192.168.123.161")
    >>> go2.walk(0.5, 0, 0)
"""

from __future__ import annotations

# Import from Rust extension (with fallback for when not built)
try:
    from _rfx import (
        # Version
        __version__,
        VERSION,
        # Math
        Quaternion,
        Transform,
        LowPassFilter,
        # Communication
        Topic,
        # Control
        Pid,
        PidConfig,
        ControlLoopHandle,
        ControlLoopStats,
        run_control_loop,
        # Hardware
        Go2,
        Go2Config,
        Go2State,
        ImuState,
        MotorState,
        MotorCmd,
        # Motor indices
        motor_idx,
        MOTOR_NAMES,
        motor_index_by_name,
        # Simulation
        PhysicsConfig,
        SimConfig,
        SimState,
        MockSimBackend,
    )
    _RUST_AVAILABLE = True
except ImportError:
    # Rust extension not built - provide stubs for pure Python usage
    __version__ = "0.1.0"
    VERSION = __version__
    _RUST_AVAILABLE = False

    # Placeholder classes for documentation/type hints
    Quaternion = None
    Transform = None
    LowPassFilter = None
    Topic = None
    Pid = None
    PidConfig = None
    ControlLoopHandle = None
    ControlLoopStats = None
    run_control_loop = None
    Go2 = None
    Go2Config = None
    Go2State = None
    ImuState = None
    MotorState = None
    MotorCmd = None
    motor_idx = None
    MOTOR_NAMES = None
    motor_index_by_name = None
    PhysicsConfig = None
    SimConfig = None
    SimState = None
    MockSimBackend = None

# Import Python modules
from .skills import skill, Skill, SkillRegistry
from .agent import Agent

# Decorators for control loops and policies
from .decorators import control_loop, policy, MotorCommands

# Neural network module (tinygrad-based)
from . import nn
from . import rl
from . import envs

__all__ = [
    # Version
    "__version__",
    "VERSION",
    # Python modules
    "nn",
    "rl",
    "envs",
    # Math
    "Quaternion",
    "Transform",
    "LowPassFilter",
    # Communication
    "Topic",
    # Control
    "Pid",
    "PidConfig",
    "ControlLoopHandle",
    "ControlLoopStats",
    "run_control_loop",
    "control_loop",
    "policy",
    "MotorCommands",
    # Hardware
    "Go2",
    "Go2Config",
    "Go2State",
    "ImuState",
    "MotorState",
    "MotorCmd",
    # Motor indices
    "motor_idx",
    "MOTOR_NAMES",
    "motor_index_by_name",
    # Simulation
    "PhysicsConfig",
    "SimConfig",
    "SimState",
    "MockSimBackend",
    # Skills
    "skill",
    "Skill",
    "SkillRegistry",
    # Agent
    "Agent",
]
