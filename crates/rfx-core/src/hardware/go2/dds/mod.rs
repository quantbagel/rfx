//! DDS communication backends for Go2 robot
//!
//! This module provides DDS (Data Distribution Service) backends for
//! communicating with the Unitree Go2 quadruped robot.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │           DdsBackend trait              │
//! ├───────────────────┬─────────────────────┤
//! │  DustDdsBackend   │  CycloneDdsBackend  │
//! │  (pure Rust)      │  (native, optional) │
//! └───────────────────┴─────────────────────┘
//! ```
//!
//! # Topics
//!
//! The Go2 uses the following DDS topics:
//! - `rt/lowstate` - Robot state at 500Hz (subscribe)
//! - `rt/lowcmd` - Low-level motor commands at 500Hz (publish, EDU mode)
//! - `rt/api/sport/request` - Sport mode commands (publish)

mod backend;
mod crc;
mod messages;

pub mod dust;

#[cfg(feature = "dds-cyclone")]
pub mod cyclone;

pub use backend::{DdsBackend, DdsQos, topics};
pub use crc::compute_crc;
pub use messages::{
    BmsStateDds, ImuStateDds, LowCmdDds, LowStateDds, MotorCmdDds, MotorStateDds,
    SportModeRequestDds,
};

// Re-export the default backend
pub use dust::DustDdsBackend;

#[cfg(feature = "dds-cyclone")]
pub use cyclone::CycloneDdsBackend;
