//! SO-101 state types

use std::time::Instant;

/// Number of joints in the SO-101 arm
pub const NUM_JOINTS: usize = 6;

/// Joint names for the SO-101 arm
pub const JOINT_NAMES: [&str; NUM_JOINTS] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
];

/// Home position for each joint (in raw servo units)
pub const HOME_POSITIONS: [u16; NUM_JOINTS] = [2048, 2048, 2048, 2048, 2048, 2048];

/// State of the SO-101 arm
#[derive(Debug, Clone)]
pub struct So101State {
    /// Joint positions in radians
    pub joint_positions: [f32; NUM_JOINTS],
    /// Joint velocities in radians/second
    pub joint_velocities: [f32; NUM_JOINTS],
    /// Timestamp when the state was captured
    pub timestamp: f64,
    /// Whether the arm is connected
    pub connected: bool,
}

impl So101State {
    /// Create a new state with default values
    pub fn new() -> Self {
        Self {
            joint_positions: [0.0; NUM_JOINTS],
            joint_velocities: [0.0; NUM_JOINTS],
            timestamp: 0.0,
            connected: false,
        }
    }

    /// Get joint positions as a slice
    pub fn joint_positions(&self) -> &[f32] {
        &self.joint_positions
    }

    /// Get joint velocities as a slice
    pub fn joint_velocities(&self) -> &[f32] {
        &self.joint_velocities
    }
}

impl Default for So101State {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing helper for calculating velocities
#[derive(Debug, Clone)]
pub(crate) struct TimingState {
    pub last_positions: [f32; NUM_JOINTS],
    pub last_read: Option<Instant>,
}

impl TimingState {
    pub fn new() -> Self {
        Self {
            last_positions: [0.0; NUM_JOINTS],
            last_read: None,
        }
    }
}

impl Default for TimingState {
    fn default() -> Self {
        Self::new()
    }
}
