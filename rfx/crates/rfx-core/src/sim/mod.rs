//! Simulation backend abstraction (v2)
//!
//! This module provides traits and utilities for integrating with various
//! physics simulators. The same pi code can run on real hardware or in
//! simulation by swapping the backend.
//!
//! # Supported Backends
//!
//! - **Isaac Sim**: NVIDIA's physics simulator with GPU acceleration
//! - **Genesis**: Lightweight Python-based simulator
//! - **MuJoCo**: DeepMind's physics engine
//! - **Mock**: For testing without simulation
//!
//! # Example
//!
//! ```ignore
//! use rfx_core::sim::{SimBackend, IsaacSimBackend};
//!
//! // Create simulator
//! let mut sim = IsaacSimBackend::connect("localhost:8080")?;
//!
//! // Same API as real robot
//! sim.reset()?;
//! loop {
//!     let state = sim.state();
//!     let action = policy.forward(&state.observation())?;
//!     sim.step(&action)?;
//! }
//! ```

mod backend;
mod config;

pub use backend::{SimBackend, SimInfo, SimState, StepResult};
pub use config::{PhysicsConfig, RenderConfig, SimBackendType, SimConfig};

// Backend implementations (feature-gated in future)
mod mock;
pub use mock::MockSimBackend;
