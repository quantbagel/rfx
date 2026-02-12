//! rfx-core: Core Rust library for the rfx robotics framework
//!
//! A minimal, high-performance robotics framework inspired by tinygrad's simplicity
//! and DimensionalOS's skill-based architecture.
//!
//! # Modules
//!
//! - [`math`] - Transform, quaternion, and filter utilities
//! - [`comm`] - Communication primitives (topics, streams, channels)
//! - [`control`] - Control loops, PID, state machines
//! - [`hardware`] - Hardware abstraction and Go2 client
//! - [`neural`] - Observation/action space definitions (neural network code in Python)
//! - [`sim`] - Simulation backend traits
//!
//! # Architecture
//!
//! ```text
//! Python (rfx)                         Rust (rfx-core)
//! ┌──────────────┐                    ┌──────────────┐
//! │   tinygrad   │                    │   Control    │
//! │   Policies   │───actions (np)────►│    Loops     │
//! └──────────────┘                    └──────────────┘
//! ```
//!
//! Neural network code (training + inference) lives in Python using tinygrad.
//! Rust handles real-time control, hardware communication, and math utilities.

#![warn(unused_must_use)]

pub mod comm;
pub mod control;
pub mod hardware;
pub mod math;
pub mod neural;
pub mod sim;

// Re-exports for convenience
pub use comm::{Channel, Stream, Topic};
pub use control::{ControlLoop, ControlLoopHandle, Pid, PidConfig};
pub use hardware::{Command, Robot, RobotState, Simulator};
pub use math::{Filter, LowPassFilter, Quaternion, Transform};
pub use neural::{ActionSpace, ObservationSpace, Space, SpaceType};
pub use sim::{SimBackend, SimConfig};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Error types for rfx-core
///
/// All errors should be handled appropriately. Use pattern matching
/// to handle specific error cases, or use `?` to propagate errors.
///
/// # Example
/// ```ignore
/// match robot.connect() {
///     Ok(conn) => { /* use connection */ },
///     Err(Error::Connection(msg)) => eprintln!("Connection failed: {}", msg),
///     Err(Error::Timeout(msg)) => eprintln!("Timed out: {}", msg),
///     Err(e) => return Err(e),
/// }
/// ```
#[derive(Debug, thiserror::Error)]
#[must_use = "errors must be handled or explicitly ignored with let _ = ..."]
#[non_exhaustive]
pub enum Error {
    /// Failed to establish or maintain connection to robot.
    /// Handle by: checking network configuration, retrying with backoff.
    #[error("Connection error: {0}")]
    Connection(String),

    /// Hardware-level error from robot sensors or actuators.
    /// Handle by: checking robot status, ensuring safe state before retry.
    #[error("Hardware error: {0}")]
    Hardware(String),

    /// Error in inter-thread or inter-process communication.
    /// Handle by: checking channel status, recreating channels if needed.
    #[error("Communication error: {0}")]
    Communication(String),

    /// Control loop timing or execution error.
    /// Handle by: reducing loop rate, profiling callback, checking system load.
    #[error("Control loop error: {0}")]
    ControlLoop(String),

    /// Invalid configuration parameter.
    /// Handle by: validating config before use, checking parameter ranges.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Operation timed out waiting for response or condition.
    /// Handle by: increasing timeout, checking if operation is blocked.
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Communication channel was closed unexpectedly.
    /// Handle by: checking sender/receiver status, recreating channel.
    #[error("Channel closed")]
    ChannelClosed,

    /// Channel is full (backpressure).
    /// Handle by: draining the receiver, increasing buffer size, or slowing the sender.
    #[error("Channel full")]
    ChannelFull,

    /// Operation attempted in invalid state (e.g., commanding disconnected robot).
    /// Handle by: checking current state before operations, using state machine.
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Hardware(format!("I/O error: {}", e))
    }
}

/// Result type alias for rfx-core operations
pub type Result<T> = std::result::Result<T, Error>;
