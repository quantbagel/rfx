//! SO-101 configuration

/// Configuration for connecting to an SO-101 arm
#[derive(Debug, Clone)]
pub struct So101Config {
    /// Serial port path (e.g., "/dev/ttyACM0")
    pub port: String,
    /// Baud rate (default: 1,000,000)
    pub baudrate: u32,
    /// Whether this arm is a leader (for teleoperation)
    pub is_leader: bool,
}

impl So101Config {
    /// Default baud rate for SO-101 (1 Mbps)
    pub const DEFAULT_BAUDRATE: u32 = 1_000_000;

    /// Create a leader arm configuration
    ///
    /// Leader arms are used for teleoperation input and have torque disabled.
    pub fn leader(port: impl Into<String>) -> Self {
        Self {
            port: port.into(),
            baudrate: Self::DEFAULT_BAUDRATE,
            is_leader: true,
        }
    }

    /// Create a follower arm configuration
    ///
    /// Follower arms receive commands and have torque enabled.
    pub fn follower(port: impl Into<String>) -> Self {
        Self {
            port: port.into(),
            baudrate: Self::DEFAULT_BAUDRATE,
            is_leader: false,
        }
    }

    /// Set the baud rate
    pub fn with_baudrate(mut self, baudrate: u32) -> Self {
        self.baudrate = baudrate;
        self
    }
}

impl Default for So101Config {
    fn default() -> Self {
        Self::follower("/dev/ttyACM0")
    }
}
