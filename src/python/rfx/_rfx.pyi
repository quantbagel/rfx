"""
Type stubs for the pi Rust extension module.

This file provides type hints for IDE autocomplete and static type checking.
"""

from typing import Any, Callable, List, Optional, Tuple, overload

# Version info
__version__: str
VERSION: str

# =============================================================================
# Math Types
# =============================================================================

class Quaternion:
    """
    A unit quaternion representing a 3D rotation.

    Quaternions provide a compact, singularity-free representation of 3D rotations.
    All quaternions are automatically normalized.

    Example:
        >>> q = Quaternion.from_euler(0.0, 0.1, 0.0)  # 0.1 rad pitch
        >>> v_rotated = q.rotate_vector([1.0, 0.0, 0.0])
        >>> roll, pitch, yaw = q.to_euler()
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, w: float, x: float, y: float, z: float) -> None: ...
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Create a quaternion from components (w, x, y, z)."""
        ...

    @staticmethod
    def identity() -> "Quaternion":
        """Create identity quaternion (no rotation)."""
        ...

    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float) -> "Quaternion":
        """
        Create from Euler angles (roll, pitch, yaw) in radians.

        Args:
            roll: Rotation around X axis (radians)
            pitch: Rotation around Y axis (radians)
            yaw: Rotation around Z axis (radians)
        """
        ...

    @staticmethod
    def from_axis_angle(axis: Tuple[float, float, float], angle: float) -> "Quaternion":
        """
        Create from axis-angle representation.

        Args:
            axis: Unit vector defining rotation axis [x, y, z]
            angle: Rotation angle in radians
        """
        ...

    def to_euler(self) -> Tuple[float, float, float]:
        """
        Get Euler angles (roll, pitch, yaw) in radians.

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        ...

    def rotate_vector(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Rotate a 3D vector by this quaternion.

        Args:
            v: Vector to rotate [x, y, z]

        Returns:
            Rotated vector [x, y, z]
        """
        ...

    def slerp(self, other: "Quaternion", t: float) -> "Quaternion":
        """
        Spherical linear interpolation between quaternions.

        Args:
            other: Target quaternion
            t: Interpolation factor (0.0 = self, 1.0 = other)

        Returns:
            Interpolated quaternion
        """
        ...

    def inverse(self) -> "Quaternion":
        """Get the inverse (conjugate) quaternion."""
        ...

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Multiply with another quaternion (compose rotations)."""
        ...

    @property
    def w(self) -> float:
        """Scalar component."""
        ...

    @property
    def x(self) -> float:
        """X component of vector part."""
        ...

    @property
    def y(self) -> float:
        """Y component of vector part."""
        ...

    @property
    def z(self) -> float:
        """Z component of vector part."""
        ...

    @property
    def roll(self) -> float:
        """Roll angle in radians."""
        ...

    @property
    def pitch(self) -> float:
        """Pitch angle in radians."""
        ...

    @property
    def yaw(self) -> float:
        """Yaw angle in radians."""
        ...

class Transform:
    """
    A rigid body transformation in 3D space (SE3).

    Combines a position (translation) and orientation (rotation) to represent
    a complete 6-DOF pose.

    Example:
        >>> t = Transform.from_euler([1.0, 2.0, 0.5], 0.0, 0.1, 0.0)
        >>> point_world = t.transform_point([0.0, 0.0, 0.0])
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Quaternion] = None,
    ) -> None: ...
    def __init__(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Quaternion] = None,
    ) -> None:
        """Create a transform from position and orientation."""
        ...

    @staticmethod
    def identity() -> "Transform":
        """Create identity transform (no translation or rotation)."""
        ...

    @staticmethod
    def from_position(x: float, y: float, z: float) -> "Transform":
        """Create a transform with translation only."""
        ...

    @staticmethod
    def from_euler(
        position: Tuple[float, float, float],
        roll: float,
        pitch: float,
        yaw: float,
    ) -> "Transform":
        """
        Create from position and Euler angles.

        Args:
            position: [x, y, z] position
            roll: Rotation around X axis (radians)
            pitch: Rotation around Y axis (radians)
            yaw: Rotation around Z axis (radians)
        """
        ...

    def compose(self, other: "Transform") -> "Transform":
        """Compose with another transform (this * other)."""
        ...

    def inverse(self) -> "Transform":
        """Get the inverse transform."""
        ...

    def transform_point(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Transform a 3D point (applies rotation then translation)."""
        ...

    def transform_vector(self, vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Transform a 3D vector (rotation only, no translation)."""
        ...

    def lerp(self, other: "Transform", t: float) -> "Transform":
        """Linear interpolation between transforms."""
        ...

    def __mul__(self, other: "Transform") -> "Transform":
        """Compose transforms (same as compose)."""
        ...

    @property
    def position(self) -> Tuple[float, float, float]:
        """Position [x, y, z]."""
        ...

    @property
    def orientation(self) -> Quaternion:
        """Orientation quaternion."""
        ...

    @property
    def x(self) -> float:
        """X position."""
        ...

    @property
    def y(self) -> float:
        """Y position."""
        ...

    @property
    def z(self) -> float:
        """Z position."""
        ...

    @property
    def roll(self) -> float:
        """Roll angle in radians."""
        ...

    @property
    def pitch(self) -> float:
        """Pitch angle in radians."""
        ...

    @property
    def yaw(self) -> float:
        """Yaw angle in radians."""
        ...

class LowPassFilter:
    """
    Low-pass filter for signal smoothing.

    Uses exponential moving average: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]

    Example:
        >>> lpf = LowPassFilter.from_cutoff(10.0, 500.0)  # 10Hz cutoff at 500Hz sample rate
        >>> smoothed = lpf.update(raw_value)
    """

    def __init__(self, alpha: float) -> None:
        """
        Create a filter with the given smoothing factor.

        Args:
            alpha: Smoothing factor (0-1). Higher = less smoothing.

        Raises:
            ValueError: If alpha is not in [0, 1]
        """
        ...

    @staticmethod
    def from_cutoff(cutoff_hz: float, sample_rate_hz: float) -> "LowPassFilter":
        """
        Create from cutoff frequency and sample rate.

        Args:
            cutoff_hz: Cutoff frequency in Hz
            sample_rate_hz: Sample rate in Hz
        """
        ...

    def update(self, value: float) -> float:
        """
        Update the filter with a new value.

        Args:
            value: New input value

        Returns:
            Filtered output value
        """
        ...

    def reset(self) -> None:
        """Reset the filter state."""
        ...

    @property
    def value(self) -> float:
        """Current filtered value."""
        ...

    @property
    def alpha(self) -> float:
        """Smoothing factor."""
        ...

# =============================================================================
# Communication Types
# =============================================================================

class Topic:
    """
    A publish-subscribe topic for inter-component communication.

    Topics allow loose coupling between components. Publishers send messages
    without knowing who (if anyone) is listening.

    Example:
        >>> topic = Topic("imu_data")
        >>> topic.publish('{"roll": 0.1, "pitch": 0.2}')
    """

    def __init__(self, name: str) -> None:
        """Create a new topic with the given name."""
        ...

    def publish(self, message: str) -> None:
        """
        Publish a message to the topic.

        Args:
            message: JSON-encoded message string
        """
        ...

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        ...

    @property
    def name(self) -> str:
        """Topic name."""
        ...

# =============================================================================
# Control Types
# =============================================================================

class PidConfig:
    """
    PID controller configuration.

    Example:
        >>> config = PidConfig(kp=10.0, ki=0.1, kd=1.0)
        >>> config = config.with_limits(-100.0, 100.0)
        >>> config = config.with_integral_limit(50.0)
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0) -> None:
        """
        Create PID configuration with gains.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        ...

    def with_limits(self, min: float, max: float) -> "PidConfig":
        """Set output limits."""
        ...

    def with_integral_limit(self, limit: float) -> "PidConfig":
        """Set integral windup limit."""
        ...

    @property
    def kp(self) -> float:
        """Proportional gain."""
        ...

    @property
    def ki(self) -> float:
        """Integral gain."""
        ...

    @property
    def kd(self) -> float:
        """Derivative gain."""
        ...

class Pid:
    """
    PID controller with anti-windup and derivative filtering.

    Example:
        >>> pid = Pid.pid(10.0, 0.1, 1.0)
        >>> output = pid.update(setpoint=1.0, measurement=0.5, dt=0.002)
    """

    def __init__(self, config: PidConfig) -> None:
        """Create a PID controller with the given configuration."""
        ...

    @staticmethod
    def p(kp: float) -> "Pid":
        """Create a P-only controller."""
        ...

    @staticmethod
    def pi(kp: float, ki: float) -> "Pid":
        """Create a PI controller."""
        ...

    @staticmethod
    def pd(kp: float, kd: float) -> "Pid":
        """Create a PD controller."""
        ...

    @staticmethod
    def pid(kp: float, ki: float, kd: float) -> "Pid":
        """Create a full PID controller."""
        ...

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """
        Update the controller with a new measurement.

        Args:
            setpoint: Desired value
            measurement: Current measured value
            dt: Time step in seconds

        Returns:
            Control output
        """
        ...

    def reset(self) -> None:
        """Reset controller state (integral term, etc.)."""
        ...

    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        """Update the controller gains."""
        ...

class ControlLoopStats:
    """Statistics for a control loop."""

    @property
    def iterations(self) -> int:
        """Number of completed iterations."""
        ...

    @property
    def overruns(self) -> int:
        """Number of timing overruns."""
        ...

    @property
    def avg_iteration_time_ms(self) -> float:
        """Average iteration time in milliseconds."""
        ...

    @property
    def max_iteration_time_ms(self) -> float:
        """Maximum iteration time in milliseconds."""
        ...

class ControlLoopHandle:
    """
    Handle to a running control loop.

    Example:
        >>> handle = run_control_loop(500.0, callback)
        >>> while handle.is_running():
        ...     print(handle.stats())
        >>> handle.stop()
    """

    def is_running(self) -> bool:
        """Check if the loop is still running."""
        ...

    def stop(self) -> None:
        """Request the loop to stop."""
        ...

    def stats(self) -> Optional[ControlLoopStats]:
        """Get current loop statistics."""
        ...

def run_control_loop(
    rate_hz: float,
    callback: Callable[[int, float], bool],
    name: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> ControlLoopStats:
    """
    Run a control loop at the specified rate.

    The callback receives (iteration, dt) and should return True to continue
    or False to stop.

    Args:
        rate_hz: Target loop rate in Hz
        callback: Function called each iteration: callback(iteration, dt) -> bool
        name: Optional name for logging
        max_iterations: Optional limit on iterations

    Returns:
        Statistics from the completed loop

    Example:
        >>> def control_callback(iteration: int, dt: float) -> bool:
        ...     # Read sensors, compute control, send commands
        ...     return iteration < 1000
        >>> stats = run_control_loop(500.0, control_callback)
    """
    ...

# =============================================================================
# Hardware Types
# =============================================================================

class Go2Config:
    """
    Configuration for connecting to a Unitree Go2 robot.

    Example:
        >>> config = Go2Config("192.168.123.161")
        >>> config = config.with_edu_mode()  # Enable low-level control
        >>> robot = Go2.connect(config)
    """

    def __init__(self, ip_address: str = "192.168.123.161") -> None:
        """
        Create configuration for connecting to a Go2.

        Args:
            ip_address: Robot's IP address (default: 192.168.123.161)
        """
        ...

    def with_edu_mode(self) -> "Go2Config":
        """Enable EDU mode for low-level motor control."""
        ...

    def with_interface(self, interface: str) -> "Go2Config":
        """
        Set the network interface to use.

        Args:
            interface: Network interface name (e.g., "eth0", "enp0s31f6")
        """
        ...

    @property
    def ip_address(self) -> str:
        """Robot IP address."""
        ...

    @property
    def edu_mode(self) -> bool:
        """Whether EDU mode is enabled."""
        ...

class ImuState:
    """IMU sensor state from the robot."""

    @property
    def quaternion(self) -> Tuple[float, float, float, float]:
        """Orientation as quaternion [w, x, y, z]."""
        ...

    @property
    def gyroscope(self) -> Tuple[float, float, float]:
        """Angular velocity [x, y, z] in rad/s."""
        ...

    @property
    def accelerometer(self) -> Tuple[float, float, float]:
        """Linear acceleration [x, y, z] in m/s^2."""
        ...

    @property
    def rpy(self) -> Tuple[float, float, float]:
        """Euler angles [roll, pitch, yaw] in radians."""
        ...

    @property
    def roll(self) -> float:
        """Roll angle in radians."""
        ...

    @property
    def pitch(self) -> float:
        """Pitch angle in radians."""
        ...

    @property
    def yaw(self) -> float:
        """Yaw angle in radians."""
        ...

    @property
    def roll_deg(self) -> float:
        """Roll angle in degrees."""
        ...

    @property
    def pitch_deg(self) -> float:
        """Pitch angle in degrees."""
        ...

    @property
    def yaw_deg(self) -> float:
        """Yaw angle in degrees."""
        ...

class MotorState:
    """State of a single motor."""

    @property
    def q(self) -> float:
        """Joint position in radians."""
        ...

    @property
    def dq(self) -> float:
        """Joint velocity in rad/s."""
        ...

    @property
    def tau_est(self) -> float:
        """Estimated torque in Nm."""
        ...

    @property
    def temperature(self) -> int:
        """Motor temperature in Celsius."""
        ...

class MotorCmd:
    """
    Command to send to a motor.

    Example:
        >>> cmd = MotorCmd.position(0.5, kp=20.0, kd=0.5)
        >>> cmd = MotorCmd.damping(kd=5.0)  # Pure damping mode
    """

    def __init__(self, q: float = 0.0, kp: float = 20.0, kd: float = 0.5) -> None:
        """Create a position command."""
        ...

    @staticmethod
    def position(q: float, kp: float, kd: float) -> "MotorCmd":
        """
        Create a position control command.

        Args:
            q: Target position in radians
            kp: Position gain
            kd: Damping gain
        """
        ...

    @staticmethod
    def damping(kd: float) -> "MotorCmd":
        """
        Create a pure damping command (position tracking disabled).

        Args:
            kd: Damping gain
        """
        ...

    @property
    def q(self) -> float:
        """Target position."""
        ...

    @property
    def dq(self) -> float:
        """Target velocity."""
        ...

    @property
    def tau(self) -> float:
        """Feedforward torque."""
        ...

    @property
    def kp(self) -> float:
        """Position gain."""
        ...

    @property
    def kd(self) -> float:
        """Damping gain."""
        ...

class Go2State:
    """
    Complete state of the Go2 robot.

    Example:
        >>> state = robot.state()
        >>> print(f"Roll: {state.imu.roll_deg:.1f}°")
        >>> positions = state.joint_positions()
    """

    @property
    def tick(self) -> int:
        """Robot tick counter."""
        ...

    @property
    def timestamp(self) -> float:
        """Timestamp in seconds."""
        ...

    @property
    def imu(self) -> ImuState:
        """IMU sensor state."""
        ...

    @property
    def position(self) -> Tuple[float, float, float]:
        """Estimated position [x, y, z] in meters."""
        ...

    @property
    def velocity(self) -> Tuple[float, float, float]:
        """Estimated velocity [vx, vy, vz] in m/s."""
        ...

    @property
    def foot_contact(self) -> Tuple[bool, bool, bool, bool]:
        """Foot contact states [FR, FL, RR, RL]."""
        ...

    def joint_positions(self) -> Tuple[float, ...]:
        """Get all 12 joint positions in radians."""
        ...

    def joint_velocities(self) -> Tuple[float, ...]:
        """Get all 12 joint velocities in rad/s."""
        ...

    def motor(self, index: int) -> MotorState:
        """
        Get state of a specific motor.

        Args:
            index: Motor index (0-11)

        Raises:
            ValueError: If index is out of range
        """
        ...

class Go2:
    """
    Interface to the Unitree Go2 quadruped robot.

    Example:
        >>> robot = Go2.connect("192.168.123.161")
        >>> robot.stand()
        >>> robot.walk(0.3, 0.0, 0.0)  # Walk forward at 0.3 m/s
        >>> robot.sit()
        >>> robot.disconnect()
    """

    @staticmethod
    def connect(
        config: Optional[Go2Config] = None,
        ip_address: Optional[str] = None,
    ) -> "Go2":
        """
        Connect to a Go2 robot.

        Args:
            config: Full configuration object
            ip_address: Simple IP address (creates default config)

        Returns:
            Connected Go2 instance

        Raises:
            RuntimeError: If connection fails
        """
        ...

    def is_connected(self) -> bool:
        """Check if still connected to the robot."""
        ...

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        ...

    def state(self) -> Go2State:
        """Get current robot state."""
        ...

    def walk(self, vx: float, vy: float, vyaw: float) -> None:
        """
        Command the robot to walk.

        Args:
            vx: Forward velocity in m/s
            vy: Lateral velocity in m/s (positive = left)
            vyaw: Yaw rate in rad/s (positive = counter-clockwise)
        """
        ...

    def stand(self) -> None:
        """Command the robot to stand."""
        ...

    def sit(self) -> None:
        """Command the robot to sit down."""
        ...

    def set_motor_position(
        self,
        motor_idx: int,
        position: float,
        kp: float,
        kd: float,
    ) -> None:
        """
        Set a single motor position (EDU mode only).

        Args:
            motor_idx: Motor index (0-11)
            position: Target position in radians
            kp: Position gain
            kd: Damping gain
        """
        ...

    def set_motor_positions(
        self,
        positions: Tuple[float, ...],
        kp: float,
        kd: float,
    ) -> None:
        """
        Set all motor positions (EDU mode only).

        Args:
            positions: 12 target positions in radians
            kp: Position gain for all motors
            kd: Damping gain for all motors
        """
        ...

# =============================================================================
# Motor Constants
# =============================================================================

class motor_idx:
    """Motor indices for the Go2 quadruped."""

    # Front Right leg
    FR_HIP: int
    FR_THIGH: int
    FR_CALF: int

    # Front Left leg
    FL_HIP: int
    FL_THIGH: int
    FL_CALF: int

    # Rear Right leg
    RR_HIP: int
    RR_THIGH: int
    RR_CALF: int

    # Rear Left leg
    RL_HIP: int
    RL_THIGH: int
    RL_CALF: int

    NUM_MOTORS: int

MOTOR_NAMES: List[str]
"""Motor names in order: ['FR_hip', 'FR_thigh', 'FR_calf', ...]"""

def motor_index_by_name(name: str) -> Optional[int]:
    """
    Get motor index by name.

    Args:
        name: Motor name (e.g., "FR_hip", "FL_thigh")

    Returns:
        Motor index (0-11) or None if not found
    """
    ...

# =============================================================================
# Simulation Types
# =============================================================================

class PhysicsConfig:
    """
    Physics simulation configuration.

    Example:
        >>> config = PhysicsConfig(dt=0.002, substeps=4)
        >>> config = PhysicsConfig.accurate()  # High-fidelity preset
    """

    def __init__(self, dt: float = 0.002, substeps: int = 4) -> None:
        """
        Create physics configuration.

        Args:
            dt: Time step in seconds
            substeps: Number of physics substeps per step
        """
        ...

    @staticmethod
    def fast() -> "PhysicsConfig":
        """Create a fast simulation preset."""
        ...

    @staticmethod
    def accurate() -> "PhysicsConfig":
        """Create a high-accuracy simulation preset."""
        ...

    @property
    def dt(self) -> float:
        """Time step in seconds."""
        ...

    @property
    def substeps(self) -> int:
        """Number of substeps per step."""
        ...

    @property
    def gravity(self) -> Tuple[float, float, float]:
        """Gravity vector [x, y, z] in m/s^2."""
        ...

class SimConfig:
    """
    Simulation backend configuration.

    Example:
        >>> config = SimConfig.mock()  # For testing
        >>> config = SimConfig.mujoco().with_num_envs(16)  # Parallel envs
    """

    def __init__(self, backend: str = "mock", num_envs: int = 1) -> None:
        """
        Create simulation configuration.

        Args:
            backend: Backend name ("mock", "mujoco", "isaac_sim", "genesis")
            num_envs: Number of parallel environments
        """
        ...

    @staticmethod
    def mock() -> "SimConfig":
        """Create mock simulation config (for testing)."""
        ...

    @staticmethod
    def isaac_sim() -> "SimConfig":
        """Create NVIDIA Isaac Sim configuration."""
        ...

    @staticmethod
    def genesis() -> "SimConfig":
        """Create Genesis simulation configuration."""
        ...

    @staticmethod
    def mujoco() -> "SimConfig":
        """Create MuJoCo simulation configuration."""
        ...

    def with_num_envs(self, n: int) -> "SimConfig":
        """Set number of parallel environments."""
        ...

    @property
    def backend(self) -> str:
        """Backend name."""
        ...

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        ...

class SimState:
    """
    State returned from simulation step.

    Example:
        >>> state, done = sim.step(actions)
        >>> if state.terminated:
        ...     state = sim.reset()
    """

    @property
    def sim_time(self) -> float:
        """Simulation time in seconds."""
        ...

    @property
    def terminated(self) -> bool:
        """Whether episode terminated (fell, out of bounds, etc.)."""
        ...

    @property
    def truncated(self) -> bool:
        """Whether episode was truncated (time limit)."""
        ...

    @property
    def reward(self) -> float:
        """Reward value (for RL)."""
        ...

    def joint_positions(self) -> List[float]:
        """Get joint positions from robot state."""
        ...

    def joint_velocities(self) -> List[float]:
        """Get joint velocities from robot state."""
        ...

class MockSimBackend:
    """
    Mock simulation backend for testing.

    Provides a simple physics simulation without requiring external
    simulation engines.

    Example:
        >>> sim = MockSimBackend()
        >>> state = sim.reset()
        >>> for _ in range(1000):
        ...     state, done = sim.step([0.0] * 12)
        ...     if done:
        ...         state = sim.reset()
    """

    def __init__(self, config: Optional[SimConfig] = None) -> None:
        """Create a mock simulation backend."""
        ...

    def name(self) -> str:
        """Get backend name."""
        ...

    def reset(self) -> SimState:
        """Reset simulation to initial state."""
        ...

    def step(self, actions: List[float]) -> Tuple[SimState, bool]:
        """
        Step the simulation.

        Args:
            actions: Joint position targets (12 values)

        Returns:
            Tuple of (new state, done flag)
        """
        ...

    def state(self) -> SimState:
        """Get current state without stepping."""
        ...

    def sim_time(self) -> float:
        """Get current simulation time."""
        ...

# =============================================================================
# Channel Types
# =============================================================================

class Sender:
    """
    Sender half of a typed channel.

    Channels provide thread-safe communication between components.

    Example:
        >>> tx, rx = channel(10)  # Bounded channel with capacity 10
        >>> tx.send('{"cmd": "walk"}')
        >>> msg = rx.recv()
    """

    def send(self, message: str) -> None:
        """
        Send a message, blocking until space is available.

        Args:
            message: JSON-encoded message string

        Raises:
            RuntimeError: If channel is closed
        """
        ...

    def try_send(self, message: str) -> bool:
        """
        Try to send without blocking.

        Args:
            message: JSON-encoded message string

        Returns:
            True if sent, False if channel is full
        """
        ...

    def is_empty(self) -> bool:
        """Check if the channel is empty."""
        ...

    def is_full(self) -> bool:
        """Check if the channel is full."""
        ...

    def __len__(self) -> int:
        """Number of messages in the channel."""
        ...

    @property
    def capacity(self) -> Optional[int]:
        """Channel capacity (None for unbounded)."""
        ...

class Receiver:
    """
    Receiver half of a typed channel.

    Example:
        >>> tx, rx = channel(10)
        >>> tx.send('{"data": 42}')
        >>> msg = rx.recv()  # Blocks until message available
        >>> msg_or_none = rx.try_recv()  # Non-blocking
    """

    def recv(self) -> str:
        """
        Receive a message, blocking until one is available.

        Returns:
            The received message string

        Raises:
            RuntimeError: If channel is closed
        """
        ...

    def try_recv(self) -> Optional[str]:
        """
        Try to receive without blocking.

        Returns:
            The message if available, None if channel is empty
        """
        ...

    def recv_timeout(self, timeout_secs: float) -> Optional[str]:
        """
        Receive with a timeout.

        Args:
            timeout_secs: Maximum time to wait in seconds

        Returns:
            The message if received, None if timed out
        """
        ...

    def latest(self) -> Optional[str]:
        """
        Get the latest message, discarding older ones.

        Returns:
            The most recent message, or None if empty
        """
        ...

    def drain(self) -> List[str]:
        """
        Drain all available messages.

        Returns:
            List of all messages in the channel
        """
        ...

    def is_empty(self) -> bool:
        """Check if the channel is empty."""
        ...

    def __len__(self) -> int:
        """Number of messages in the channel."""
        ...

def channel(capacity: int) -> Tuple[Sender, Receiver]:
    """
    Create a bounded channel with the specified capacity.

    Args:
        capacity: Maximum number of messages the channel can hold

    Returns:
        Tuple of (Sender, Receiver)

    Example:
        >>> tx, rx = channel(100)
        >>> tx.send('{"type": "command", "action": "stand"}')
    """
    ...

def unbounded_channel() -> Tuple[Sender, Receiver]:
    """
    Create an unbounded channel.

    Returns:
        Tuple of (Sender, Receiver)

    Note:
        Unbounded channels can grow without limit. Use bounded channels
        for backpressure in production systems.
    """
    ...

# =============================================================================
# Stream Types
# =============================================================================

class Stream:
    """
    A typed data stream with blocking/non-blocking receive.

    Streams wrap channels with additional functionality like
    getting only the latest value (useful for sensor data).

    Example:
        >>> tx, rx = channel(10)
        >>> stream = stream_from_receiver(rx)
        >>> # In a loop:
        >>> latest = stream.latest()  # Get most recent, discard old
    """

    def next(self) -> Optional[str]:
        """
        Receive the next value, blocking until available.

        Returns:
            The next message, or None if stream is stopped
        """
        ...

    def next_timeout(self, timeout_secs: float) -> Optional[str]:
        """
        Receive with a timeout.

        Args:
            timeout_secs: Maximum time to wait in seconds

        Returns:
            The message if received, None if timed out or stopped
        """
        ...

    def try_next(self) -> Optional[str]:
        """
        Try to receive without blocking.

        Returns:
            The message if available, None otherwise
        """
        ...

    def latest(self) -> Optional[str]:
        """
        Get the latest value, discarding older ones.

        This is useful for sensor data where you only care about
        the most recent reading.

        Returns:
            The most recent message, or None if empty
        """
        ...

    def stop(self) -> None:
        """Stop the stream. After stopping, all receive methods return None."""
        ...

    def is_active(self) -> bool:
        """Check if the stream is still active."""
        ...

def stream_from_receiver(receiver: Receiver) -> Stream:
    """
    Create a stream from a receiver.

    Args:
        receiver: The Receiver to wrap

    Returns:
        A new Stream instance
    """
    ...
