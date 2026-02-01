//! Python type bindings
//!
//! Wrapper types that expose rfx-core functionality to Python.

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use std::sync::Arc;

// ============================================================================
// Math Bindings
// ============================================================================

/// A unit quaternion representing a 3D rotation
#[pyclass(name = "Quaternion")]
#[derive(Clone)]
pub struct PyQuaternion {
    inner: rfx_core::math::Quaternion,
}

#[pymethods]
impl PyQuaternion {
    #[new]
    #[pyo3(signature = (w = 1.0, x = 0.0, y = 0.0, z = 0.0))]
    fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: rfx_core::math::Quaternion::new(w, x, y, z),
        }
    }

    /// Create identity quaternion (no rotation)
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: rfx_core::math::Quaternion::identity(),
        }
    }

    /// Create from Euler angles (roll, pitch, yaw) in radians
    #[staticmethod]
    fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            inner: rfx_core::math::Quaternion::from_euler(roll, pitch, yaw),
        }
    }

    /// Create from axis-angle representation
    #[staticmethod]
    fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        Self {
            inner: rfx_core::math::Quaternion::from_axis_angle(axis, angle),
        }
    }

    /// Get Euler angles (roll, pitch, yaw) in radians
    fn to_euler(&self) -> (f64, f64, f64) {
        self.inner.to_euler()
    }

    /// Rotate a 3D vector
    fn rotate_vector(&self, v: [f64; 3]) -> [f64; 3] {
        self.inner.rotate_vector(v)
    }

    /// Spherical linear interpolation
    fn slerp(&self, other: &PyQuaternion, t: f64) -> PyQuaternion {
        PyQuaternion {
            inner: self.inner.slerp(&other.inner, t),
        }
    }

    /// Get the inverse quaternion
    fn inverse(&self) -> PyQuaternion {
        PyQuaternion {
            inner: self.inner.inverse(),
        }
    }

    /// Multiply with another quaternion
    fn __mul__(&self, other: &PyQuaternion) -> PyQuaternion {
        PyQuaternion {
            inner: self.inner.multiply(&other.inner),
        }
    }

    #[getter]
    fn w(&self) -> f64 {
        self.inner.w
    }

    #[getter]
    fn x(&self) -> f64 {
        self.inner.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.inner.y
    }

    #[getter]
    fn z(&self) -> f64 {
        self.inner.z
    }

    #[getter]
    fn roll(&self) -> f64 {
        self.inner.roll()
    }

    #[getter]
    fn pitch(&self) -> f64 {
        self.inner.pitch()
    }

    #[getter]
    fn yaw(&self) -> f64 {
        self.inner.yaw()
    }

    fn __repr__(&self) -> String {
        format!(
            "Quaternion(w={:.4}, x={:.4}, y={:.4}, z={:.4})",
            self.inner.w, self.inner.x, self.inner.y, self.inner.z
        )
    }
}

/// A rigid body transformation in 3D space (SE3)
#[pyclass(name = "Transform")]
#[derive(Clone)]
pub struct PyTransform {
    inner: rfx_core::math::Transform,
}

#[pymethods]
impl PyTransform {
    #[new]
    #[pyo3(signature = (position = None, orientation = None))]
    fn new(position: Option<[f64; 3]>, orientation: Option<PyQuaternion>) -> Self {
        let pos = position.unwrap_or([0.0, 0.0, 0.0]);
        let orient = orientation
            .map(|q| q.inner)
            .unwrap_or_else(rfx_core::math::Quaternion::identity);
        Self {
            inner: rfx_core::math::Transform::new(pos, orient),
        }
    }

    /// Create identity transform
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: rfx_core::math::Transform::identity(),
        }
    }

    /// Create from position only
    #[staticmethod]
    fn from_position(x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: rfx_core::math::Transform::from_position(x, y, z),
        }
    }

    /// Create from Euler angles and position
    #[staticmethod]
    fn from_euler(position: [f64; 3], roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            inner: rfx_core::math::Transform::from_euler(position, roll, pitch, yaw),
        }
    }

    /// Compose with another transform
    fn compose(&self, other: &PyTransform) -> PyTransform {
        PyTransform {
            inner: self.inner.compose(&other.inner),
        }
    }

    /// Get the inverse transform
    fn inverse(&self) -> PyTransform {
        PyTransform {
            inner: self.inner.inverse(),
        }
    }

    /// Transform a 3D point
    fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        self.inner.transform_point(point)
    }

    /// Transform a 3D vector (rotation only)
    fn transform_vector(&self, vector: [f64; 3]) -> [f64; 3] {
        self.inner.transform_vector(vector)
    }

    /// Linear interpolation
    fn lerp(&self, other: &PyTransform, t: f64) -> PyTransform {
        PyTransform {
            inner: self.inner.lerp(&other.inner, t),
        }
    }

    fn __mul__(&self, other: &PyTransform) -> PyTransform {
        self.compose(other)
    }

    #[getter]
    fn position(&self) -> [f64; 3] {
        self.inner.position
    }

    #[getter]
    fn orientation(&self) -> PyQuaternion {
        PyQuaternion {
            inner: self.inner.orientation,
        }
    }

    #[getter]
    fn x(&self) -> f64 {
        self.inner.x()
    }

    #[getter]
    fn y(&self) -> f64 {
        self.inner.y()
    }

    #[getter]
    fn z(&self) -> f64 {
        self.inner.z()
    }

    #[getter]
    fn roll(&self) -> f64 {
        self.inner.roll()
    }

    #[getter]
    fn pitch(&self) -> f64 {
        self.inner.pitch()
    }

    #[getter]
    fn yaw(&self) -> f64 {
        self.inner.yaw()
    }

    fn __repr__(&self) -> String {
        format!(
            "Transform(pos=[{:.3}, {:.3}, {:.3}], rpy=[{:.3}, {:.3}, {:.3}])",
            self.inner.x(),
            self.inner.y(),
            self.inner.z(),
            self.inner.roll(),
            self.inner.pitch(),
            self.inner.yaw()
        )
    }
}

/// Low-pass filter for signal smoothing
#[pyclass(name = "LowPassFilter")]
pub struct PyLowPassFilter {
    inner: rfx_core::math::LowPassFilter,
}

#[pymethods]
impl PyLowPassFilter {
    #[new]
    fn new(alpha: f64) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err("alpha must be between 0 and 1"));
        }
        Ok(Self {
            inner: rfx_core::math::LowPassFilter::new(alpha),
        })
    }

    /// Create from cutoff frequency and sample rate
    #[staticmethod]
    fn from_cutoff(cutoff_hz: f64, sample_rate_hz: f64) -> Self {
        Self {
            inner: rfx_core::math::LowPassFilter::from_cutoff(cutoff_hz, sample_rate_hz),
        }
    }

    /// Update with a new value
    fn update(&mut self, value: f64) -> f64 {
        use rfx_core::math::Filter;
        self.inner.update(value)
    }

    /// Reset the filter
    fn reset(&mut self) {
        use rfx_core::math::Filter;
        self.inner.reset();
    }

    /// Get current filtered value
    #[getter]
    fn value(&self) -> f64 {
        use rfx_core::math::Filter;
        self.inner.value()
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }
}

// ============================================================================
// Communication Bindings
// ============================================================================

/// A publish-subscribe topic
#[pyclass(name = "Topic")]
pub struct PyTopic {
    inner: rfx_core::comm::Topic<String>,
}

#[pymethods]
impl PyTopic {
    #[new]
    fn new(name: &str) -> Self {
        Self {
            inner: rfx_core::comm::Topic::new(name),
        }
    }

    /// Publish a message (as JSON string)
    fn publish(&self, message: &str) {
        self.inner.publish(message.to_string());
    }

    /// Get subscriber count
    #[getter]
    fn subscriber_count(&self) -> usize {
        self.inner.subscriber_count()
    }

    /// Get topic name
    #[getter]
    fn name(&self) -> String {
        self.inner.name()
    }
}

// ============================================================================
// Control Bindings
// ============================================================================

/// PID controller configuration
#[pyclass(name = "PidConfig")]
#[derive(Clone)]
pub struct PyPidConfig {
    pub(crate) inner: rfx_core::control::PidConfig,
}

#[pymethods]
impl PyPidConfig {
    #[new]
    #[pyo3(signature = (kp = 1.0, ki = 0.0, kd = 0.0))]
    fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            inner: rfx_core::control::PidConfig::new(kp, ki, kd),
        }
    }

    /// Set output limits
    fn with_limits(&self, min: f64, max: f64) -> Self {
        Self {
            inner: self.inner.with_limits(min, max),
        }
    }

    /// Set integral windup limit
    fn with_integral_limit(&self, limit: f64) -> Self {
        Self {
            inner: self.inner.with_integral_limit(limit),
        }
    }

    #[getter]
    fn kp(&self) -> f64 {
        self.inner.kp
    }

    #[getter]
    fn ki(&self) -> f64 {
        self.inner.ki
    }

    #[getter]
    fn kd(&self) -> f64 {
        self.inner.kd
    }

    fn __repr__(&self) -> String {
        format!(
            "PidConfig(kp={}, ki={}, kd={})",
            self.inner.kp, self.inner.ki, self.inner.kd
        )
    }
}

/// PID controller
#[pyclass(name = "Pid")]
pub struct PyPid {
    inner: rfx_core::control::Pid,
}

#[pymethods]
impl PyPid {
    #[new]
    fn new(config: PyPidConfig) -> Self {
        Self {
            inner: rfx_core::control::Pid::new(config.inner),
        }
    }

    /// Create a P-only controller
    #[staticmethod]
    fn p(kp: f64) -> Self {
        Self {
            inner: rfx_core::control::Pid::p(kp),
        }
    }

    /// Create a PI controller
    #[staticmethod]
    fn pi(kp: f64, ki: f64) -> Self {
        Self {
            inner: rfx_core::control::Pid::pi(kp, ki),
        }
    }

    /// Create a PD controller
    #[staticmethod]
    fn pd(kp: f64, kd: f64) -> Self {
        Self {
            inner: rfx_core::control::Pid::pd(kp, kd),
        }
    }

    /// Create a full PID controller
    #[staticmethod]
    fn pid(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            inner: rfx_core::control::Pid::pid(kp, ki, kd),
        }
    }

    /// Update the controller
    fn update(&mut self, setpoint: f64, measurement: f64, dt: f64) -> f64 {
        self.inner.update(setpoint, measurement, dt)
    }

    /// Reset the controller state
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Set gains
    fn set_gains(&mut self, kp: f64, ki: f64, kd: f64) {
        self.inner.set_gains(kp, ki, kd);
    }
}

/// Control loop statistics
#[pyclass(name = "ControlLoopStats")]
pub struct PyControlLoopStats {
    #[pyo3(get)]
    pub iterations: u64,
    #[pyo3(get)]
    pub overruns: u64,
    #[pyo3(get)]
    pub avg_iteration_time_ms: f64,
    #[pyo3(get)]
    pub max_iteration_time_ms: f64,
}

impl From<rfx_core::control::ControlLoopStats> for PyControlLoopStats {
    fn from(stats: rfx_core::control::ControlLoopStats) -> Self {
        Self {
            iterations: stats.iterations,
            overruns: stats.overruns,
            avg_iteration_time_ms: stats.avg_iteration_time.as_secs_f64() * 1000.0,
            max_iteration_time_ms: stats.max_iteration_time.as_secs_f64() * 1000.0,
        }
    }
}

#[pymethods]
impl PyControlLoopStats {
    fn __repr__(&self) -> String {
        format!(
            "ControlLoopStats(iterations={}, overruns={}, avg_time={:.3}ms)",
            self.iterations, self.overruns, self.avg_iteration_time_ms
        )
    }
}

/// Handle to a running control loop
#[pyclass(name = "ControlLoopHandle")]
pub struct PyControlLoopHandle {
    inner: Option<rfx_core::control::ControlLoopHandle>,
}

#[pymethods]
impl PyControlLoopHandle {
    /// Check if the loop is running
    fn is_running(&self) -> bool {
        self.inner.as_ref().is_some_and(|h| h.is_running())
    }

    /// Stop the control loop
    fn stop(&self) {
        if let Some(ref h) = self.inner {
            h.stop();
        }
    }

    /// Get current statistics
    fn stats(&self) -> Option<PyControlLoopStats> {
        self.inner.as_ref().map(|h| PyControlLoopStats::from(h.stats()))
    }
}

// ============================================================================
// Hardware Bindings
// ============================================================================

/// Go2 robot configuration
#[pyclass(name = "Go2Config")]
#[derive(Clone)]
pub struct PyGo2Config {
    pub(crate) inner: rfx_core::hardware::go2::Go2Config,
}

#[pymethods]
impl PyGo2Config {
    #[new]
    #[pyo3(signature = (ip_address = "192.168.123.161"))]
    fn new(ip_address: &str) -> Self {
        Self {
            inner: rfx_core::hardware::go2::Go2Config::new(ip_address),
        }
    }

    /// Enable EDU mode for low-level motor control
    fn with_edu_mode(&self) -> Self {
        Self {
            inner: self.inner.clone().with_edu_mode(),
        }
    }

    /// Set network interface
    fn with_interface(&self, interface: &str) -> Self {
        Self {
            inner: self.inner.clone().with_interface(interface),
        }
    }

    #[getter]
    fn ip_address(&self) -> &str {
        &self.inner.ip_address
    }

    #[getter]
    fn edu_mode(&self) -> bool {
        self.inner.edu_mode
    }

    fn __repr__(&self) -> String {
        format!(
            "Go2Config(ip='{}', edu_mode={})",
            self.inner.ip_address, self.inner.edu_mode
        )
    }
}

/// IMU state
#[pyclass(name = "ImuState")]
#[derive(Clone)]
pub struct PyImuState {
    inner: rfx_core::hardware::go2::ImuState,
}

#[pymethods]
impl PyImuState {
    #[getter]
    fn quaternion(&self) -> [f32; 4] {
        self.inner.quaternion
    }

    #[getter]
    fn gyroscope(&self) -> [f32; 3] {
        self.inner.gyroscope
    }

    #[getter]
    fn accelerometer(&self) -> [f32; 3] {
        self.inner.accelerometer
    }

    #[getter]
    fn rpy(&self) -> [f32; 3] {
        self.inner.rpy
    }

    #[getter]
    fn roll(&self) -> f32 {
        self.inner.roll()
    }

    #[getter]
    fn pitch(&self) -> f32 {
        self.inner.pitch()
    }

    #[getter]
    fn yaw(&self) -> f32 {
        self.inner.yaw()
    }

    #[getter]
    fn roll_deg(&self) -> f32 {
        self.inner.roll_deg()
    }

    #[getter]
    fn pitch_deg(&self) -> f32 {
        self.inner.pitch_deg()
    }

    #[getter]
    fn yaw_deg(&self) -> f32 {
        self.inner.yaw_deg()
    }

    fn __repr__(&self) -> String {
        format!(
            "ImuState(roll={:.2}°, pitch={:.2}°, yaw={:.2}°)",
            self.inner.roll_deg(),
            self.inner.pitch_deg(),
            self.inner.yaw_deg()
        )
    }
}

/// Motor state
#[pyclass(name = "MotorState")]
#[derive(Clone)]
pub struct PyMotorState {
    inner: rfx_core::hardware::go2::MotorState,
}

#[pymethods]
impl PyMotorState {
    #[getter]
    fn q(&self) -> f32 {
        self.inner.q
    }

    #[getter]
    fn dq(&self) -> f32 {
        self.inner.dq
    }

    #[getter]
    fn tau_est(&self) -> f32 {
        self.inner.tau_est
    }

    #[getter]
    fn temperature(&self) -> i8 {
        self.inner.temperature
    }

    fn __repr__(&self) -> String {
        format!(
            "MotorState(q={:.3}, dq={:.3}, tau={:.3})",
            self.inner.q, self.inner.dq, self.inner.tau_est
        )
    }
}

/// Motor command
#[pyclass(name = "MotorCmd")]
#[derive(Clone)]
pub struct PyMotorCmd {
    pub(crate) inner: rfx_core::hardware::go2::MotorCmd,
}

#[pymethods]
impl PyMotorCmd {
    #[new]
    #[pyo3(signature = (q = 0.0, kp = 20.0, kd = 0.5))]
    fn new(q: f32, kp: f32, kd: f32) -> Self {
        Self {
            inner: rfx_core::hardware::go2::MotorCmd::position(q, kp, kd),
        }
    }

    /// Create a position command
    #[staticmethod]
    fn position(q: f32, kp: f32, kd: f32) -> Self {
        Self {
            inner: rfx_core::hardware::go2::MotorCmd::position(q, kp, kd),
        }
    }

    /// Create a damping command
    #[staticmethod]
    fn damping(kd: f32) -> Self {
        Self {
            inner: rfx_core::hardware::go2::MotorCmd::damping(kd),
        }
    }

    #[getter]
    fn q(&self) -> f32 {
        self.inner.q
    }

    #[getter]
    fn dq(&self) -> f32 {
        self.inner.dq
    }

    #[getter]
    fn tau(&self) -> f32 {
        self.inner.tau
    }

    #[getter]
    fn kp(&self) -> f32 {
        self.inner.kp
    }

    #[getter]
    fn kd(&self) -> f32 {
        self.inner.kd
    }
}

/// Go2 robot state
#[pyclass(name = "Go2State")]
#[derive(Clone)]
pub struct PyGo2State {
    inner: rfx_core::hardware::go2::Go2State,
}

#[pymethods]
impl PyGo2State {
    #[getter]
    fn tick(&self) -> u32 {
        self.inner.tick
    }

    #[getter]
    fn timestamp(&self) -> f64 {
        self.inner.timestamp
    }

    #[getter]
    fn imu(&self) -> PyImuState {
        PyImuState {
            inner: self.inner.imu,
        }
    }

    #[getter]
    fn position(&self) -> [f32; 3] {
        self.inner.position
    }

    #[getter]
    fn velocity(&self) -> [f32; 3] {
        self.inner.velocity
    }

    #[getter]
    fn foot_contact(&self) -> [bool; 4] {
        self.inner.foot_contact
    }

    /// Get all joint positions
    fn joint_positions(&self) -> [f32; 12] {
        self.inner.joint_positions()
    }

    /// Get all joint velocities
    fn joint_velocities(&self) -> [f32; 12] {
        self.inner.joint_velocities()
    }

    /// Get motor state by index
    fn motor(&self, index: usize) -> PyResult<PyMotorState> {
        self.inner
            .motors
            .get(index)
            .map(|m| PyMotorState { inner: *m })
            .ok_or_else(|| PyValueError::new_err(format!("Invalid motor index: {}", index)))
    }

    fn __repr__(&self) -> String {
        format!(
            "Go2State(tick={}, imu={:?})",
            self.inner.tick,
            self.inner.imu.rpy
        )
    }
}

/// Unitree Go2 robot interface
#[pyclass(name = "Go2")]
pub struct PyGo2 {
    inner: Arc<rfx_core::hardware::go2::Go2>,
}

#[pymethods]
impl PyGo2 {
    /// Connect to a Go2 robot
    #[staticmethod]
    #[pyo3(signature = (config = None, ip_address = None))]
    fn connect(py: Python<'_>, config: Option<PyGo2Config>, ip_address: Option<&str>) -> PyResult<Self> {
        let cfg = match (config, ip_address) {
            (Some(c), _) => c.inner,
            (None, Some(ip)) => rfx_core::hardware::go2::Go2Config::new(ip),
            (None, None) => rfx_core::hardware::go2::Go2Config::default(),
        };

        // Release GIL during connection
        let result = py.allow_threads(|| rfx_core::hardware::go2::Go2::connect(cfg));

        match result {
            Ok(go2) => Ok(Self {
                inner: Arc::new(go2),
            }),
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Check if connected
    fn is_connected(&self) -> bool {
        self.inner.is_connected()
    }

    /// Disconnect from the robot
    fn disconnect(&self) -> PyResult<()> {
        self.inner
            .disconnect()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get current robot state
    fn state(&self) -> PyGo2State {
        PyGo2State {
            inner: self.inner.go2_state(),
        }
    }

    /// Walk with given velocities
    fn walk(&self, py: Python<'_>, vx: f32, vy: f32, vyaw: f32) -> PyResult<()> {
        py.allow_threads(|| self.inner.walk(vx, vy, vyaw))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Stand
    fn stand(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.stand())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Sit
    fn sit(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.sit())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Set a single motor position (EDU mode only)
    fn set_motor_position(
        &self,
        py: Python<'_>,
        motor_idx: usize,
        position: f32,
        kp: f32,
        kd: f32,
    ) -> PyResult<()> {
        py.allow_threads(|| self.inner.set_motor_position(motor_idx, position, kp, kd))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Set all motor positions (EDU mode only)
    fn set_motor_positions(
        &self,
        py: Python<'_>,
        positions: [f32; 12],
        kp: f32,
        kd: f32,
    ) -> PyResult<()> {
        py.allow_threads(|| self.inner.set_motor_positions(&positions, kp, kd))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("Go2(connected={})", self.inner.is_connected())
    }
}

// ============================================================================
// Simulation Bindings (v2)
// ============================================================================

/// Simulation physics configuration
#[pyclass(name = "PhysicsConfig")]
#[derive(Clone)]
pub struct PyPhysicsConfig {
    pub(crate) inner: rfx_core::sim::PhysicsConfig,
}

#[pymethods]
impl PyPhysicsConfig {
    #[new]
    #[pyo3(signature = (dt = 0.002, substeps = 4))]
    fn new(dt: f64, substeps: usize) -> Self {
        Self {
            inner: rfx_core::sim::PhysicsConfig {
                dt,
                substeps,
                ..Default::default()
            },
        }
    }

    /// Create a fast simulation config
    #[staticmethod]
    fn fast() -> Self {
        Self {
            inner: rfx_core::sim::PhysicsConfig::fast(),
        }
    }

    /// Create a high-accuracy simulation config
    #[staticmethod]
    fn accurate() -> Self {
        Self {
            inner: rfx_core::sim::PhysicsConfig::accurate(),
        }
    }

    #[getter]
    fn dt(&self) -> f64 {
        self.inner.dt
    }

    #[getter]
    fn substeps(&self) -> usize {
        self.inner.substeps
    }

    #[getter]
    fn gravity(&self) -> [f64; 3] {
        self.inner.gravity
    }

    fn __repr__(&self) -> String {
        format!(
            "PhysicsConfig(dt={}, substeps={}, gravity={:?})",
            self.inner.dt, self.inner.substeps, self.inner.gravity
        )
    }
}

/// Simulation configuration
#[pyclass(name = "SimConfig")]
#[derive(Clone)]
pub struct PySimConfig {
    pub(crate) inner: rfx_core::sim::SimConfig,
}

#[pymethods]
impl PySimConfig {
    #[new]
    #[pyo3(signature = (backend = "mock", num_envs = 1))]
    fn new(backend: &str, num_envs: usize) -> Self {
        Self {
            inner: rfx_core::sim::SimConfig {
                backend: backend.to_string(),
                num_envs,
                ..Default::default()
            },
        }
    }

    /// Create a mock simulation config
    #[staticmethod]
    fn mock() -> Self {
        Self {
            inner: rfx_core::sim::SimConfig::default(),
        }
    }

    /// Create an Isaac Sim configuration
    #[staticmethod]
    fn isaac_sim() -> Self {
        Self {
            inner: rfx_core::sim::SimConfig::isaac_sim(),
        }
    }

    /// Create a Genesis configuration
    #[staticmethod]
    fn genesis() -> Self {
        Self {
            inner: rfx_core::sim::SimConfig::genesis(),
        }
    }

    /// Create a MuJoCo configuration
    #[staticmethod]
    fn mujoco() -> Self {
        Self {
            inner: rfx_core::sim::SimConfig::mujoco(),
        }
    }

    /// Set number of parallel environments
    fn with_num_envs(&self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().with_num_envs(n),
        }
    }

    #[getter]
    fn backend(&self) -> &str {
        &self.inner.backend
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs
    }

    fn __repr__(&self) -> String {
        format!(
            "SimConfig(backend='{}', num_envs={})",
            self.inner.backend, self.inner.num_envs
        )
    }
}

/// Simulation state
#[pyclass(name = "SimState")]
#[derive(Clone)]
pub struct PySimState {
    inner: rfx_core::sim::SimState,
}

#[pymethods]
impl PySimState {
    #[getter]
    fn sim_time(&self) -> f64 {
        self.inner.sim_time
    }

    #[getter]
    fn terminated(&self) -> bool {
        self.inner.terminated
    }

    #[getter]
    fn truncated(&self) -> bool {
        self.inner.truncated
    }

    #[getter]
    fn reward(&self) -> f64 {
        self.inner.reward
    }

    /// Get joint positions from robot state
    fn joint_positions(&self) -> Vec<f64> {
        self.inner.robot.joint_positions.clone()
    }

    /// Get joint velocities from robot state
    fn joint_velocities(&self) -> Vec<f64> {
        self.inner.robot.joint_velocities.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SimState(time={:.3}, terminated={}, truncated={})",
            self.inner.sim_time, self.inner.terminated, self.inner.truncated
        )
    }
}

/// Mock simulation backend for testing
#[pyclass(name = "MockSimBackend")]
pub struct PyMockSimBackend {
    inner: rfx_core::sim::MockSimBackend,
}

#[pymethods]
impl PyMockSimBackend {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<PySimConfig>) -> Self {
        let backend = match config {
            Some(c) => rfx_core::sim::MockSimBackend::with_config(c.inner),
            None => rfx_core::sim::MockSimBackend::new(),
        };
        Self { inner: backend }
    }

    /// Get backend name
    fn name(&self) -> &str {
        use rfx_core::sim::SimBackend;
        self.inner.name()
    }

    /// Reset the simulation
    fn reset(&mut self) -> PyResult<PySimState> {
        use rfx_core::sim::SimBackend;
        self.inner
            .reset()
            .map(|s| PySimState { inner: s })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Step the simulation with actions
    fn step(&mut self, actions: Vec<f32>) -> PyResult<(PySimState, bool)> {
        use rfx_core::sim::SimBackend;
        self.inner
            .step(&actions)
            .map(|r| (PySimState { inner: r.state }, r.done))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get current state
    fn state(&self) -> PySimState {
        use rfx_core::sim::SimBackend;
        PySimState {
            inner: self.inner.state(),
        }
    }

    /// Get current simulation time
    fn sim_time(&self) -> f64 {
        use rfx_core::sim::SimBackend;
        self.inner.sim_time()
    }

    fn __repr__(&self) -> String {
        use rfx_core::sim::SimBackend;
        format!("MockSimBackend(time={:.3})", self.inner.sim_time())
    }
}

// ============================================================================
// Channel Bindings
// ============================================================================

/// Sender half of a typed channel
#[pyclass(name = "Sender")]
pub struct PySender {
    inner: rfx_core::comm::Sender<String>,
}

#[pymethods]
impl PySender {
    /// Send a message (as JSON string)
    fn send(&self, message: &str) -> PyResult<()> {
        self.inner
            .send(message.to_string())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Try to send without blocking
    fn try_send(&self, message: &str) -> PyResult<bool> {
        match self.inner.try_send(message.to_string()) {
            Ok(()) => Ok(true),
            Err(rfx_core::Error::Communication(_)) => Ok(false), // Channel full
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Check if the channel is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Check if the channel is full
    fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    /// Get the number of messages in the channel
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get the channel capacity (None for unbounded)
    #[getter]
    fn capacity(&self) -> Option<usize> {
        self.inner.capacity()
    }

    fn __repr__(&self) -> String {
        format!("Sender(len={}, capacity={:?})", self.inner.len(), self.inner.capacity())
    }
}

/// Receiver half of a typed channel
#[pyclass(name = "Receiver")]
pub struct PyReceiver {
    inner: rfx_core::comm::Receiver<String>,
}

#[pymethods]
impl PyReceiver {
    /// Receive a message, blocking until one is available
    fn recv(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.inner.recv())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Try to receive without blocking
    fn try_recv(&self) -> PyResult<Option<String>> {
        self.inner
            .try_recv()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Receive with a timeout in seconds
    fn recv_timeout(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<Option<String>> {
        let timeout = std::time::Duration::from_secs_f64(timeout_secs);
        py.allow_threads(|| self.inner.recv_timeout(timeout))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get the latest message, discarding older ones
    fn latest(&self) -> PyResult<Option<String>> {
        self.inner
            .latest()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Drain all available messages
    fn drain(&self) -> Vec<String> {
        self.inner.drain()
    }

    /// Check if the channel is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the number of messages in the channel
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Receiver(len={})", self.inner.len())
    }
}

/// Create a bounded channel with the specified capacity
#[pyfunction]
pub fn channel(capacity: usize) -> (PySender, PyReceiver) {
    let (tx, rx) = rfx_core::comm::bounded_channel(capacity);
    (PySender { inner: tx }, PyReceiver { inner: rx })
}

/// Create an unbounded channel
#[pyfunction]
pub fn unbounded_channel() -> (PySender, PyReceiver) {
    let (tx, rx) = rfx_core::comm::unbounded_channel();
    (PySender { inner: tx }, PyReceiver { inner: rx })
}

// ============================================================================
// Stream Bindings
// ============================================================================

/// A typed data stream with optional transformations
#[pyclass(name = "Stream")]
pub struct PyStream {
    inner: rfx_core::comm::Stream<String>,
}

#[pymethods]
impl PyStream {
    /// Receive the next value
    fn next(&self, py: Python<'_>) -> Option<String> {
        py.allow_threads(|| self.inner.next())
    }

    /// Receive the next value with a timeout in seconds
    fn next_timeout(&self, py: Python<'_>, timeout_secs: f64) -> Option<String> {
        let timeout = std::time::Duration::from_secs_f64(timeout_secs);
        py.allow_threads(|| self.inner.next_timeout(timeout))
    }

    /// Try to receive without blocking
    fn try_next(&self) -> Option<String> {
        self.inner.try_next()
    }

    /// Get the latest value, discarding older ones
    fn latest(&self) -> Option<String> {
        self.inner.latest()
    }

    /// Stop the stream
    fn stop(&self) {
        self.inner.stop();
    }

    /// Check if the stream is active
    fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    fn __repr__(&self) -> String {
        format!("Stream(active={})", self.inner.is_active())
    }
}

/// Create a stream from a receiver
#[pyfunction]
pub fn stream_from_receiver(receiver: &PyReceiver) -> PyStream {
    // Note: This creates a new stream from a cloned receiver
    let rx = receiver.inner.clone();
    PyStream {
        inner: rfx_core::comm::Stream::from_receiver(rx),
    }
}
