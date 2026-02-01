//! DDS backend abstraction trait
//!
//! Defines the interface that all DDS backends must implement for
//! communicating with the Go2 robot.

use crate::comm::Receiver;
use crate::Result;

use super::super::{Go2Config, LowCmd, LowState, SportModeCmd};

/// DDS backend abstraction for Go2 communication
///
/// This trait defines the interface that allows different DDS implementations
/// to be used interchangeably. The default implementation uses dust-dds (pure Rust),
/// with an optional CycloneDDS backend for maximum compatibility.
///
/// # Example
///
/// ```ignore
/// use rfx_core::hardware::go2::dds::{DdsBackend, DustDdsBackend};
///
/// let config = Go2Config::default();
/// let backend = DustDdsBackend::new(&config)?;
///
/// // Subscribe to robot state
/// let state_rx = backend.subscribe_state();
///
/// // Publish commands
/// backend.publish_sport_cmd(&SportModeCmd::stand())?;
/// ```
pub trait DdsBackend: Send + 'static {
    /// Create a new DDS backend with the given configuration
    fn new(config: &Go2Config) -> Result<Self>
    where
        Self: Sized;

    /// Publish a low-level motor command
    ///
    /// This is used for direct motor control in EDU mode.
    /// Commands are published to `rt/lowcmd` at 500Hz.
    fn publish_low_cmd(&self, cmd: &LowCmd) -> Result<()>;

    /// Publish a sport mode command
    ///
    /// This is used for high-level control (walking, standing, etc).
    /// Commands are published to `rt/api/sport/request`.
    fn publish_sport_cmd(&self, cmd: &SportModeCmd) -> Result<()>;

    /// Subscribe to robot state updates
    ///
    /// Returns a receiver that provides `LowState` updates from `rt/lowstate`.
    /// The robot publishes state at 500Hz.
    fn subscribe_state(&self) -> Receiver<LowState>;

    /// Check if the backend is connected to the robot
    fn is_connected(&self) -> bool;

    /// Disconnect from the robot and clean up resources
    fn disconnect(&self);
}

/// DDS Quality of Service settings for Go2 communication
#[derive(Debug, Clone)]
pub struct DdsQos {
    /// Reliability policy
    pub reliable: bool,
    /// History depth (number of samples to keep)
    pub history_depth: usize,
    /// Deadline in microseconds (0 = no deadline)
    pub deadline_us: u64,
}

impl Default for DdsQos {
    fn default() -> Self {
        Self {
            reliable: true,
            history_depth: 1,
            deadline_us: 0,
        }
    }
}

impl DdsQos {
    /// QoS for real-time state streaming (500Hz)
    pub fn realtime_state() -> Self {
        Self {
            reliable: false, // Best-effort for low latency
            history_depth: 1,
            deadline_us: 2000, // 2ms deadline for 500Hz
        }
    }

    /// QoS for command publishing
    pub fn command() -> Self {
        Self {
            reliable: true,
            history_depth: 1,
            deadline_us: 2000,
        }
    }
}

/// DDS topic names for Go2 communication
pub mod topics {
    /// Low-level state topic (500Hz, subscribe)
    pub const LOW_STATE: &str = "rt/lowstate";

    /// Low-level command topic (500Hz, publish, EDU mode)
    pub const LOW_CMD: &str = "rt/lowcmd";

    /// Sport mode request topic (publish)
    pub const SPORT_REQUEST: &str = "rt/api/sport/request";

    /// Sport mode state topic (subscribe)
    pub const SPORT_STATE: &str = "rt/sportmodestate";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qos_defaults() {
        let qos = DdsQos::default();
        assert!(qos.reliable);
        assert_eq!(qos.history_depth, 1);
    }

    #[test]
    fn test_realtime_qos() {
        let qos = DdsQos::realtime_state();
        assert!(!qos.reliable);
        assert_eq!(qos.deadline_us, 2000);
    }
}
