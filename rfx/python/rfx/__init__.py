"""rfx: The PyTorch for Robots."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Rust extension (with fallback for when not built)
# ---------------------------------------------------------------------------
try:
    from . import _rfx

    __version__ = _rfx.__version__
    VERSION = _rfx.VERSION

    Quaternion = getattr(_rfx, "Quaternion", None)
    Transform = getattr(_rfx, "Transform", None)
    LowPassFilter = getattr(_rfx, "LowPassFilter", None)
    Topic = getattr(_rfx, "Topic", None)
    Pid = getattr(_rfx, "Pid", None)
    PidConfig = getattr(_rfx, "PidConfig", None)
    ControlLoopHandle = getattr(_rfx, "ControlLoopHandle", None)
    ControlLoopStats = getattr(_rfx, "ControlLoopStats", None)
    run_control_loop = getattr(_rfx, "run_control_loop", None)
    Go2 = getattr(_rfx, "Go2", None)
    Go2Config = getattr(_rfx, "Go2Config", None)
    Go2State = getattr(_rfx, "Go2State", None)
    ImuState = getattr(_rfx, "ImuState", None)
    MotorState = getattr(_rfx, "MotorState", None)
    MotorCmd = getattr(_rfx, "MotorCmd", None)
    motor_idx = getattr(_rfx, "motor_idx", None)
    MOTOR_NAMES = getattr(_rfx, "MOTOR_NAMES", None)
    motor_index_by_name = getattr(_rfx, "motor_index_by_name", None)
    PhysicsConfig = getattr(_rfx, "PhysicsConfig", None)
    SimConfig = getattr(_rfx, "SimConfig", None)
    SimState = getattr(_rfx, "SimState", None)
    MockSimBackend = getattr(_rfx, "MockSimBackend", None)

    _RUST_AVAILABLE = True
except ImportError:
    __version__ = "0.2.0"
    VERSION = __version__
    _RUST_AVAILABLE = False

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

# ---------------------------------------------------------------------------
# Core protocol & modules
# ---------------------------------------------------------------------------
from . import jit, node, robot, runtime, utils, workflow  # noqa: E402
from .agent import Agent  # noqa: E402
from .decorators import MotorCommands, policy  # noqa: E402
from .deploy import deploy  # noqa: E402
from .hub import LoadedPolicy, inspect_policy, load_policy, push_policy  # noqa: E402
from .observation import ObservationSpec, make_observation, unpad_action  # noqa: E402
from .robot import URDF, Robot, RobotBase, RobotConfig, load_config  # noqa: E402
from .robot.config import (  # noqa: E402
    G1_CONFIG,
    GO2_CONFIG,
    INNATE_CONFIG,
    SO101_CONFIG,
)
from .session import Session, SessionStats, run  # noqa: E402
from .skills import Skill, SkillRegistry, skill  # noqa: E402

# ---------------------------------------------------------------------------
# Optional: transforms, drivers, hardware, ML, teleop
# ---------------------------------------------------------------------------
try:
    from . import tf  # noqa: E402
    from .tf import TransformBroadcaster, TransformBuffer, TransformListener  # noqa: E402
except ModuleNotFoundError:
    tf = TransformBuffer = TransformBroadcaster = TransformListener = None

try:
    from . import drivers  # noqa: E402
    from .drivers import RobotDriver, get_driver, list_drivers, register_driver  # noqa: E402
except ModuleNotFoundError:
    drivers = RobotDriver = get_driver = list_drivers = register_driver = None

try:
    from . import real, sim  # noqa: E402
    from .real import RealRobot  # noqa: E402
    from .real.so101 import So101Robot  # noqa: E402
    from .sim import MockRobot, SimRobot  # noqa: E402

    So101 = So101Robot
except ModuleNotFoundError:
    SimRobot = MockRobot = RealRobot = So101Robot = So101 = sim = real = None

try:
    from . import envs, nn, rl  # noqa: E402
except ModuleNotFoundError:
    nn = rl = envs = None

try:
    from . import teleop  # noqa: E402
except ModuleNotFoundError:
    teleop = None

try:
    from . import collection  # noqa: E402
except ModuleNotFoundError:
    collection = None

__all__ = [
    # Version
    "__version__",
    "VERSION",
    # Core protocol
    "Agent",
    "LoadedPolicy",
    "MotorCommands",
    "ObservationSpec",
    "Robot",
    "RobotBase",
    "RobotConfig",
    "Session",
    "SessionStats",
    "Skill",
    "SkillRegistry",
    "URDF",
    # Built-in robot configs
    "SO101_CONFIG",
    "GO2_CONFIG",
    "G1_CONFIG",
    "INNATE_CONFIG",
    "deploy",
    "inspect_policy",
    "jit",
    "load_config",
    "load_policy",
    "make_observation",
    "node",
    "policy",
    "push_policy",
    "robot",
    "run",
    "runtime",
    "skill",
    "unpad_action",
    "utils",
    "workflow",
    # Transforms & drivers
    "TransformBroadcaster",
    "TransformBuffer",
    "TransformListener",
    "RobotDriver",
    "drivers",
    "get_driver",
    "list_drivers",
    "register_driver",
    "tf",
    # Hardware backends
    "MockRobot",
    "RealRobot",
    "SimRobot",
    "So101",
    "So101Robot",
    "real",
    "sim",
    # Optional ML
    "envs",
    "nn",
    "rl",
    # Teleop
    "teleop",
    # Collection
    "collection",
    # Rust types
    "ControlLoopHandle",
    "ControlLoopStats",
    "Go2",
    "Go2Config",
    "Go2State",
    "ImuState",
    "MOTOR_NAMES",
    "MockSimBackend",
    "MotorCmd",
    "MotorState",
    "PhysicsConfig",
    "Pid",
    "PidConfig",
    "Quaternion",
    "SimConfig",
    "SimState",
    "Topic",
    "Transform",
    "LowPassFilter",
    "motor_idx",
    "motor_index_by_name",
    "run_control_loop",
]
