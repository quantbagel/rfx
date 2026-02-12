//! Unitree Go2 quadruped robot interface
//!
//! This module provides the interface for communicating with the
//! Unitree Go2 robot via DDS (Data Distribution Service).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │               Go2 struct                │  High-level API
//! └─────────────────────────────────────────┘
//!                     │
//! ┌─────────────────────────────────────────┐
//! │           DdsBackend trait              │  Abstraction
//! ├───────────────────┬─────────────────────┤
//! │  DustDdsBackend   │  CycloneDdsBackend  │  Implementations
//! │  (pure Rust)      │  (native, optional) │
//! └───────────────────┴─────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use rfx_core::hardware::go2::{Go2, Go2Config};
//!
//! // Connect to the robot
//! let config = Go2Config::new("192.168.123.161");
//! let go2 = Go2::connect(config)?;
//!
//! // High-level sport mode
//! go2.walk(0.5, 0.0, 0.0)?;
//! go2.stand()?;
//!
//! // Get state
//! let state = go2.state();
//! println!("IMU: {:?}", state.imu);
//! ```

pub mod dds;
mod types;

pub use dds::{DdsBackend, DustDdsBackend};
pub use types::{
    BmsState, Go2State, ImuState, LowCmd, LowState, MotorCmd, MotorState, RobotMode, SportModeCmd,
    SportModeState,
};

#[cfg(feature = "dds-cyclone")]
pub use dds::CycloneDdsBackend;

use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use super::{motor_idx::NUM_MOTORS, Command, Robot, RobotState};
use crate::comm::Receiver;
use crate::{Error, Result};

/// Go2 robot configuration
#[derive(Debug, Clone)]
pub struct Go2Config {
    /// Robot IP address
    pub ip_address: String,
    /// Whether to use EDU mode (low-level motor control)
    pub edu_mode: bool,
    /// Command rate in Hz
    pub command_rate_hz: f64,
    /// State update rate in Hz
    pub state_rate_hz: f64,
    /// Network interface (e.g., "eth0")
    pub network_interface: Option<String>,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// DDS domain ID (default: 0)
    pub dds_domain_id: i32,
}

impl Default for Go2Config {
    fn default() -> Self {
        Self {
            ip_address: "192.168.123.161".into(),
            edu_mode: false,
            command_rate_hz: 500.0,
            state_rate_hz: 500.0,
            network_interface: None,
            connection_timeout: Duration::from_secs(5),
            dds_domain_id: 0,
        }
    }
}

impl Go2Config {
    /// Create a new config with the given IP address
    pub fn new(ip_address: impl Into<String>) -> Self {
        Self {
            ip_address: ip_address.into(),
            ..Default::default()
        }
    }

    /// Enable EDU mode for low-level motor control
    pub fn with_edu_mode(mut self) -> Self {
        self.edu_mode = true;
        self
    }

    /// Set the network interface
    pub fn with_interface(mut self, interface: impl Into<String>) -> Self {
        self.network_interface = Some(interface.into());
        self
    }

    /// Set the DDS domain ID
    pub fn with_domain_id(mut self, domain_id: i32) -> Self {
        self.dds_domain_id = domain_id;
        self
    }
}

/// Unitree Go2 robot interface
///
/// Provides high-level and low-level control of the Go2 quadruped robot
/// via DDS communication.
///
/// # DDS Backends
///
/// The Go2 uses a DDS backend for communication:
/// - **DustDdsBackend** (default): Pure Rust implementation, no system dependencies
/// - **CycloneDdsBackend** (optional): Native CycloneDDS, requires system library
///
/// # Example
///
/// ```ignore
/// use rfx_core::hardware::go2::{Go2, Go2Config};
///
/// // Connect to the robot
/// let config = Go2Config::new("192.168.123.161");
/// let go2 = Go2::connect(config)?;
///
/// // High-level sport mode
/// go2.walk(0.5, 0.0, 0.0)?;
/// go2.stand()?;
///
/// // Or get state
/// let state = go2.go2_state();
/// println!("IMU: {:?}", state.imu);
/// ```
pub struct Go2 {
    config: Go2Config,
    state: Arc<RwLock<Go2State>>,
    backend: Arc<dyn DdsBackendExt>,
    connected: Arc<AtomicBool>,
    start_time: Instant,
}

/// Extended backend trait for internal use (adds Send + Sync)
trait DdsBackendExt: Send + Sync {
    fn publish_low_cmd(&self, cmd: &LowCmd) -> Result<()>;
    fn publish_sport_cmd(&self, cmd: &SportModeCmd) -> Result<()>;
    fn subscribe_state(&self) -> Receiver<LowState>;
    fn is_connected(&self) -> bool;
    fn disconnect(&self);
}

impl<T: DdsBackend + Sync> DdsBackendExt for T {
    fn publish_low_cmd(&self, cmd: &LowCmd) -> Result<()> {
        DdsBackend::publish_low_cmd(self, cmd)
    }

    fn publish_sport_cmd(&self, cmd: &SportModeCmd) -> Result<()> {
        DdsBackend::publish_sport_cmd(self, cmd)
    }

    fn subscribe_state(&self) -> Receiver<LowState> {
        DdsBackend::subscribe_state(self)
    }

    fn is_connected(&self) -> bool {
        DdsBackend::is_connected(self)
    }

    fn disconnect(&self) {
        DdsBackend::disconnect(self)
    }
}

impl Go2 {
    /// Connect to a Go2 robot using the default DDS backend
    ///
    /// Uses DustDdsBackend (pure Rust) by default. For native CycloneDDS,
    /// use `connect_with_backend` with `CycloneDdsBackend`.
    pub fn connect(config: Go2Config) -> Result<Self> {
        #[cfg(feature = "dds-cyclone")]
        {
            // Try CycloneDDS first if feature is enabled, fall back to dust-dds
            match CycloneDdsBackend::new(&config) {
                Ok(backend) => {
                    tracing::info!("Using CycloneDDS backend");
                    return Self::connect_with_backend(config, backend);
                }
                Err(e) => {
                    tracing::warn!("CycloneDDS backend failed, falling back to dust-dds: {}", e);
                }
            }
        }

        let backend = DustDdsBackend::new(&config)?;
        tracing::info!("Using dust-dds backend");
        Self::connect_with_backend(config, backend)
    }

    /// Connect to a Go2 robot with a specific DDS backend
    pub fn connect_with_backend<B: DdsBackend + Sync + 'static>(
        config: Go2Config,
        backend: B,
    ) -> Result<Self> {
        let state = Arc::new(RwLock::new(Go2State::default()));
        let connected = Arc::new(AtomicBool::new(false));
        let start_time = Instant::now();
        let backend = Arc::new(backend);

        // Start state update worker
        let state_clone = state.clone();
        let connected_clone = connected.clone();
        let state_rx = backend.subscribe_state();

        thread::spawn(move || {
            Self::state_worker(state_clone, connected_clone, state_rx);
        });

        // Wait for first state update or timeout
        let timeout = config.connection_timeout;
        let deadline = Instant::now() + timeout;

        while Instant::now() < deadline {
            if backend.is_connected() {
                connected.store(true, Ordering::Relaxed);
                tracing::info!(
                    "Connected to Go2 at {} (EDU mode: {})",
                    config.ip_address,
                    config.edu_mode
                );
                return Ok(Self {
                    config,
                    state,
                    backend,
                    connected,
                    start_time,
                });
            }
            thread::sleep(Duration::from_millis(50));
        }

        // Connection established via DDS, even if no state received yet
        connected.store(true, Ordering::Relaxed);
        tracing::info!(
            "Connected to Go2 at {} (waiting for state)",
            config.ip_address
        );

        Ok(Self {
            config,
            state,
            backend,
            connected,
            start_time,
        })
    }

    /// Connect with async support
    pub async fn connect_async(config: Go2Config) -> Result<Self> {
        tokio::task::spawn_blocking(move || Self::connect(config))
            .await
            .map_err(|e| Error::Connection(e.to_string()))?
    }

    /// State update worker thread
    ///
    /// Computes derived state outside the write lock, then locks briefly
    /// to swap in the new values. This minimizes contention with readers.
    fn state_worker(
        state: Arc<RwLock<Go2State>>,
        connected: Arc<AtomicBool>,
        state_rx: Receiver<LowState>,
    ) {
        while connected.load(Ordering::Relaxed) {
            // Try to receive state with timeout
            match state_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Some(low_state)) => {
                    // Pre-compute derived data outside the lock
                    let motors: [MotorState; NUM_MOTORS] = low_state.motor_state;

                    let foot_contact: [bool; 4] = std::array::from_fn(|i| {
                        low_state.foot_force[i] > 20 // Threshold for contact detection
                    });

                    // Lock briefly and swap in pre-computed values
                    let mut s = state.write();
                    s.tick = low_state.tick;
                    s.imu = low_state.imu;
                    s.battery = low_state.bms_state;
                    s.motors = motors;
                    s.foot_contact = foot_contact;
                }
                Ok(None) => {
                    // Timeout, continue loop
                }
                Err(_) => {
                    // Channel closed, exit
                    break;
                }
            }
        }
    }

    /// Get the current robot state (Go2-specific)
    pub fn go2_state(&self) -> Go2State {
        self.state.read().clone()
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed) && self.backend.is_connected()
    }

    /// Disconnect from the robot
    pub fn disconnect(&self) -> Result<()> {
        self.connected.store(false, Ordering::Relaxed);
        self.backend.disconnect();
        tracing::info!("Disconnected from Go2");
        Ok(())
    }

    /// Get the elapsed time since connection
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    // === Sport Mode Commands ===

    /// Command the robot to walk with given velocities
    ///
    /// # Arguments
    /// * `vx` - Forward velocity (m/s), positive = forward
    /// * `vy` - Lateral velocity (m/s), positive = left
    /// * `vyaw` - Rotational velocity (rad/s), positive = counter-clockwise
    pub fn walk(&self, vx: f32, vy: f32, vyaw: f32) -> Result<()> {
        if !self.is_connected() {
            return Err(Error::Connection("Not connected".into()));
        }

        let cmd = SportModeCmd::move_cmd(vx, vy, vyaw);
        self.backend.publish_sport_cmd(&cmd)
    }

    /// Command the robot to stand
    pub fn stand(&self) -> Result<()> {
        if !self.is_connected() {
            return Err(Error::Connection("Not connected".into()));
        }

        let cmd = SportModeCmd::stand();
        self.backend.publish_sport_cmd(&cmd)
    }

    /// Command the robot to sit down
    pub fn sit(&self) -> Result<()> {
        if !self.is_connected() {
            return Err(Error::Connection("Not connected".into()));
        }

        let cmd = SportModeCmd::stop_move();
        self.backend.publish_sport_cmd(&cmd)
    }

    /// Stop all movement
    pub fn stop(&self) -> Result<()> {
        self.sit()
    }

    // === Low-Level Commands (EDU mode) ===

    /// Send a low-level motor command
    ///
    /// Requires EDU mode to be enabled.
    pub fn send_low_cmd(&self, cmd: LowCmd) -> Result<()> {
        if !self.config.edu_mode {
            return Err(Error::Config("Low-level commands require EDU mode".into()));
        }
        if !self.is_connected() {
            return Err(Error::Connection("Not connected".into()));
        }

        self.backend.publish_low_cmd(&cmd)
    }

    /// Set a single motor position
    pub fn set_motor_position(
        &self,
        motor_idx: usize,
        position: f32,
        kp: f32,
        kd: f32,
    ) -> Result<()> {
        if motor_idx >= NUM_MOTORS {
            return Err(Error::InvalidState(format!(
                "Invalid motor index: {}",
                motor_idx
            )));
        }

        let mut cmd = LowCmd::default();
        cmd.motor_cmd[motor_idx] = MotorCmd {
            mode: 0x01,
            q: position,
            dq: 0.0,
            tau: 0.0,
            kp,
            kd,
        };

        self.send_low_cmd(cmd)
    }

    /// Set all motor positions
    pub fn set_motor_positions(
        &self,
        positions: &[f32; NUM_MOTORS],
        kp: f32,
        kd: f32,
    ) -> Result<()> {
        let mut cmd = LowCmd::default();
        for (i, &pos) in positions.iter().enumerate() {
            cmd.motor_cmd[i] = MotorCmd {
                mode: 0x01,
                q: pos,
                dq: 0.0,
                tau: 0.0,
                kp,
                kd,
            };
        }
        self.send_low_cmd(cmd)
    }

    /// Create an async state stream
    pub fn states(&self) -> StateStream {
        StateStream {
            state: self.state.clone(),
            connected: self.connected.clone(),
        }
    }

    /// Get the DDS backend state receiver directly
    ///
    /// This provides raw access to state updates for high-frequency control.
    pub fn raw_state_receiver(&self) -> Receiver<LowState> {
        self.backend.subscribe_state()
    }
}

impl Robot for Go2 {
    fn state(&self) -> RobotState {
        let go2_state = self.state.read();
        RobotState {
            pose: crate::math::Transform::from_euler(
                [
                    go2_state.position[0] as f64,
                    go2_state.position[1] as f64,
                    go2_state.position[2] as f64,
                ],
                go2_state.imu.rpy[0] as f64,
                go2_state.imu.rpy[1] as f64,
                go2_state.imu.rpy[2] as f64,
            ),
            joint_positions: go2_state.motors.iter().map(|m| m.q as f64).collect(),
            joint_velocities: go2_state.motors.iter().map(|m| m.dq as f64).collect(),
            joint_torques: go2_state.motors.iter().map(|m| m.tau_est as f64).collect(),
            timestamp: go2_state.timestamp,
        }
    }

    fn send_command(&self, cmd: Command) -> Result<()> {
        if !self.config.edu_mode {
            return Err(Error::Config("Generic commands require EDU mode".into()));
        }

        let mut low_cmd = LowCmd::default();

        if let Some(positions) = cmd.positions {
            let kp = cmd.kp.unwrap_or_else(|| {
                let mut av = arrayvec::ArrayVec::new();
                for _ in 0..NUM_MOTORS {
                    av.push(20.0);
                }
                av
            });
            let kd = cmd.kd.unwrap_or_else(|| {
                let mut av = arrayvec::ArrayVec::new();
                for _ in 0..NUM_MOTORS {
                    av.push(0.5);
                }
                av
            });

            for (i, pos) in positions.iter().enumerate().take(NUM_MOTORS) {
                low_cmd.motor_cmd[i].mode = 0x01;
                low_cmd.motor_cmd[i].q = *pos as f32;
                low_cmd.motor_cmd[i].kp = kp.get(i).copied().unwrap_or(20.0) as f32;
                low_cmd.motor_cmd[i].kd = kd.get(i).copied().unwrap_or(0.5) as f32;
            }
        }

        if let Some(velocities) = cmd.velocities {
            for (i, vel) in velocities.iter().enumerate().take(NUM_MOTORS) {
                low_cmd.motor_cmd[i].dq = *vel as f32;
            }
        }

        if let Some(torques) = cmd.torques {
            for (i, tau) in torques.iter().enumerate().take(NUM_MOTORS) {
                low_cmd.motor_cmd[i].tau = *tau as f32;
            }
        }

        self.send_low_cmd(low_cmd)
    }

    fn num_joints(&self) -> usize {
        NUM_MOTORS
    }

    fn name(&self) -> &str {
        "UnitreeGo2"
    }

    fn is_ready(&self) -> bool {
        self.is_connected()
    }

    fn emergency_stop(&self) -> Result<()> {
        // Send damping command for safe stop
        if self.config.edu_mode {
            let cmd = LowCmd::damping(5.0);
            self.send_low_cmd(cmd)
        } else {
            self.stand()
        }
    }

    fn reset(&self) -> Result<()> {
        self.stand()
    }
}

impl Drop for Go2 {
    fn drop(&mut self) {
        let _ = self.disconnect();
    }
}

/// Async iterator for state updates
pub struct StateStream {
    state: Arc<RwLock<Go2State>>,
    connected: Arc<AtomicBool>,
}

impl StateStream {
    /// Get the next state (async-friendly polling)
    pub async fn next(&self) -> Option<Go2State> {
        if !self.connected.load(Ordering::Relaxed) {
            return None;
        }

        // Wait for state updates at 500Hz
        tokio::time::sleep(Duration::from_millis(2)).await;
        Some(self.state.read().clone())
    }

    /// Get current state without waiting
    pub fn current(&self) -> Option<Go2State> {
        if !self.connected.load(Ordering::Relaxed) {
            return None;
        }
        Some(self.state.read().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_go2_config() {
        let config = Go2Config::new("192.168.1.100")
            .with_edu_mode()
            .with_interface("eth0")
            .with_domain_id(5);

        assert_eq!(config.ip_address, "192.168.1.100");
        assert!(config.edu_mode);
        assert_eq!(config.network_interface, Some("eth0".into()));
        assert_eq!(config.dds_domain_id, 5);
    }

    #[test]
    fn test_go2_state_new() {
        let state = Go2State::new();
        assert_eq!(state.motors.len(), NUM_MOTORS);
    }

    #[test]
    fn test_motor_cmd_position() {
        let cmd = MotorCmd::position(1.0, 20.0, 0.5);
        assert_eq!(cmd.mode, 0x01);
        assert_eq!(cmd.q, 1.0);
        assert_eq!(cmd.kp, 20.0);
        assert_eq!(cmd.kd, 0.5);
    }

    #[test]
    fn test_low_cmd_damping() {
        let cmd = LowCmd::damping(3.0);
        for motor in &cmd.motor_cmd {
            assert_eq!(motor.mode, 0x0A);
            assert_eq!(motor.kd, 3.0);
        }
    }

    #[test]
    fn test_sport_mode_cmd() {
        let cmd = SportModeCmd::move_cmd(0.5, 0.1, 0.0);
        assert_eq!(cmd.velocity[0], 0.5);
        assert_eq!(cmd.velocity[1], 0.1);
        assert_eq!(cmd.gait_type, 1);
    }
}
