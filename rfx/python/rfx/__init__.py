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

from . import jit, utils
from .config import CameraConfig, JointConfig, RobotConfig, load_config
from .observation import ObservationSpec, make_observation, unpad_action
from .robot import Robot, RobotBase
from .session import Session, SessionStats, run
from .urdf import URDF

# Optional runtime dependencies (torch/camera stacks) are not required for
# lightweight API/skill usage, so guard these imports.
try:
    from . import real, sim
    from .real import RealRobot
    from .sim import MockRobot, SimRobot
except ModuleNotFoundError:
    SimRobot = None
    MockRobot = None
    RealRobot = None
    sim = None
    real = None

try:
    from .teleop import (
        ArmPairConfig,
        BimanualSo101Session,
        CameraStreamConfig,
        InprocTransport,
        JitPolicyConfig,
        JitterBenchmarkResult,
        LeRobotExportConfig,
        LeRobotPackageWriter,
        LeRobotRecorder,
        LoopTimingStats,
        RecordedEpisode,
        RustSubscription,
        RustTransport,
        Subscription,
        TeleopSessionConfig,
        TransportConfig,
        TransportEnvelope,
        assert_jitter_budget,
        create_transport,
        run_jitter_benchmark,
        rust_transport_available,
    )
except ModuleNotFoundError:
    ArmPairConfig = None
    BimanualSo101Session = None
    CameraStreamConfig = None
    InprocTransport = None
    JitPolicyConfig = None
    JitterBenchmarkResult = None
    LeRobotExportConfig = None
    LeRobotPackageWriter = None
    LeRobotRecorder = None
    LoopTimingStats = None
    RecordedEpisode = None
    RustSubscription = None
    RustTransport = None
    Subscription = None
    TeleopSessionConfig = None
    TransportConfig = None
    TransportEnvelope = None
    assert_jitter_budget = None
    create_transport = None
    rust_transport_available = None
    run_jitter_benchmark = None

# ============================================================================
# rfx v1 API (backward compatible)
# ============================================================================

from .agent import Agent
from .decorators import MotorCommands, control_loop, policy
from .providers import (
    available_providers,
    use,
)
from .providers import (
    enabled as provider_enabled,
)
from .providers import (
    require as require_provider,
)
from .sdk import UniversalRobot, VelocityCommand
from .sdk import connect as connect_robot
from .skills import Skill, SkillRegistry, skill

try:
    from . import envs, nn, rl
except ModuleNotFoundError:
    nn = None
    rl = None
    envs = None

__all__ = [
    # v2 API
    "URDF",
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
    "Session",
    "SessionStats",
    "run",
    "sim",
    "real",
    "utils",
    "jit",
    "ArmPairConfig",
    "BimanualSo101Session",
    "CameraStreamConfig",
    "InprocTransport",
    "JitPolicyConfig",
    "JitterBenchmarkResult",
    "LeRobotExportConfig",
    "LeRobotPackageWriter",
    "LeRobotRecorder",
    "LoopTimingStats",
    "RecordedEpisode",
    "RustSubscription",
    "RustTransport",
    "Subscription",
    "TeleopSessionConfig",
    "TransportConfig",
    "TransportEnvelope",
    "assert_jitter_budget",
    "create_transport",
    "rust_transport_available",
    "run_jitter_benchmark",
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
    "connect_robot",
    "UniversalRobot",
    "VelocityCommand",
    "use",
    "available_providers",
    "require_provider",
    "provider_enabled",
]
