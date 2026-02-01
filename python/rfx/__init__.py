"""
rfx: The PyTorch for Robots

Three methods. That's it.

    >>> robot = rfx.SimRobot.from_config("so101.yaml", num_envs=4096)
    >>> obs = robot.observe()           # Dict[str, Tensor]
    >>> robot.act(action)               # Execute action
    >>> obs = robot.reset()             # Reset

Real robot - same interface:

    >>> robot = rfx.RealRobot.from_config("so101.yaml", port="/dev/ttyACM0")
    >>> obs = robot.observe()           # (1, 64) tensor
    >>> robot.act(action)               # Send to hardware

v1 API still supported:

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

# ============================================================================
# rfx v2 API
# ============================================================================

from .robot import Robot, RobotBase
from .config import RobotConfig, CameraConfig, JointConfig, load_config
from .observation import ObservationSpec, make_observation, unpad_action
from .sim import SimRobot, MockRobot
from .real import RealRobot
from . import sim
from . import real
from . import utils

# ============================================================================
# rfx v1 API (backward compatible)
# ============================================================================

from .skills import skill, Skill, SkillRegistry
from .agent import Agent
from .decorators import control_loop, policy, MotorCommands
from . import nn
from . import rl
from . import envs

__all__ = [
    # v2 API
    "Robot",
    "RobotBase",
    "SimRobot",
    "MockRobot",
    "RealRobot",
    "RobotConfig",
    "CameraConfig",
    "JointConfig",
    "load_config",
    "ObservationSpec",
    "make_observation",
    "unpad_action",
    "sim",
    "real",
    "utils",
    # v1 API
    "__version__",
    "VERSION",
    "nn",
    "rl",
    "envs",
    "Quaternion",
    "Transform",
    "LowPassFilter",
    "Topic",
    "Pid",
    "PidConfig",
    "ControlLoopHandle",
    "ControlLoopStats",
    "run_control_loop",
    "control_loop",
    "policy",
    "MotorCommands",
    "Go2",
    "Go2Config",
    "Go2State",
    "ImuState",
    "MotorState",
    "MotorCmd",
    "motor_idx",
    "MOTOR_NAMES",
    "motor_index_by_name",
    "PhysicsConfig",
    "SimConfig",
    "SimState",
    "MockSimBackend",
    "skill",
    "Skill",
    "SkillRegistry",
    "Agent",
]
