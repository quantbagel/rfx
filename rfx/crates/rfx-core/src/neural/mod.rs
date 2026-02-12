//! Neural network spaces for robot policies
//!
//! This module provides observation and action space definitions for neural
//! network policies. The actual neural network code lives in Python (tinygrad),
//! while these Rust types define the interfaces.
//!
//! # Example
//!
//! ```
//! use rfx_core::neural::{ObservationSpace, ActionSpace};
//!
//! // Get standard Go2 spaces
//! let obs_space = ObservationSpace::go2_standard();
//! let action_space = ActionSpace::go2_standard();
//!
//! assert_eq!(obs_space.total_dim, 48);
//! assert_eq!(action_space.dim(), 12);
//! ```

mod spaces;

pub use spaces::{ActionSpace, DType, ObservationSpace, Space, SpaceType};
