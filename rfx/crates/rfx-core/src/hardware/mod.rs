//! Hardware abstraction and robot interfaces
//!
//! Provides traits for robot hardware abstraction and implementations
//! for the Unitree Go2 quadruped robot and SO-101 robotic arm.

#[cfg(feature = "hardware-go2")]
pub mod go2;
#[cfg(feature = "hardware-so101")]
pub mod so101;
mod traits;

#[cfg(feature = "hardware-go2")]
pub use go2::{
    dds::{DdsBackend, DustDdsBackend},
    Go2, Go2Config, Go2State, ImuState, LowCmd, LowState, MotorCmd, MotorState, SportModeCmd,
};
#[cfg(feature = "hardware-so101")]
pub use so101::{So101, So101Config, So101State};

#[cfg(all(feature = "hardware-go2", feature = "dds-cyclone"))]
pub use go2::dds::CycloneDdsBackend;
pub use traits::{Command, Robot, RobotState, Simulator};

/// Motor indices for the Go2 quadruped
pub mod motor_idx {
    // Front Right leg
    pub const FR_HIP: usize = 0;
    pub const FR_THIGH: usize = 1;
    pub const FR_CALF: usize = 2;

    // Front Left leg
    pub const FL_HIP: usize = 3;
    pub const FL_THIGH: usize = 4;
    pub const FL_CALF: usize = 5;

    // Rear Right leg
    pub const RR_HIP: usize = 6;
    pub const RR_THIGH: usize = 7;
    pub const RR_CALF: usize = 8;

    // Rear Left leg
    pub const RL_HIP: usize = 9;
    pub const RL_THIGH: usize = 10;
    pub const RL_CALF: usize = 11;

    pub const NUM_MOTORS: usize = 12;
}

/// Joint position limits for Go2 (in radians)
///
/// All limits include a 5% safety margin from hardware limits.
/// Source: Unitree Go2 SDK v4.2
pub mod joint_limits {
    /// Hip joint: Controls leg spread (lateral movement)
    ///
    /// The hip joint rotates the leg around the body's longitudinal axis,
    /// allowing the robot to spread or adduct its legs.
    pub mod hip {
        /// Minimum hip angle (-49.4°) - leg fully adducted (pulled in)
        pub const MIN: f64 = -0.863;
        /// Maximum hip angle (+49.4°) - leg fully abducted (spread out)
        pub const MAX: f64 = 0.863;
        /// Neutral standing position
        pub const NEUTRAL: f64 = 0.0;
    }

    /// Thigh joint: Controls leg forward/backward swing
    ///
    /// The thigh joint rotates around the hip's lateral axis,
    /// providing the main forward/backward leg motion.
    pub mod thigh {
        /// Minimum thigh angle (-39.3°) - leg extended backward
        pub const MIN: f64 = -0.686;
        /// Maximum thigh angle (161.5°) - leg pulled up against body
        pub const MAX: f64 = 2.818;
        /// Neutral standing position (approximately 45°)
        pub const NEUTRAL: f64 = 0.785;
    }

    /// Calf joint: Controls knee bend
    ///
    /// The calf joint controls the knee, allowing the leg to extend
    /// or fold. Note: uses negative angles (more negative = more bent).
    pub mod calf {
        /// Minimum calf angle (-161.5°) - knee fully bent
        pub const MIN: f64 = -2.818;
        /// Maximum calf angle (-50.9°) - knee nearly straight
        pub const MAX: f64 = -0.888;
        /// Neutral standing position (approximately -90°)
        pub const NEUTRAL: f64 = -1.571;
    }

    // Legacy flat constants for backward compatibility
    #[deprecated(note = "use joint_limits::hip::MIN instead")]
    pub const HIP_MIN: f64 = hip::MIN;
    #[deprecated(note = "use joint_limits::hip::MAX instead")]
    pub const HIP_MAX: f64 = hip::MAX;
    #[deprecated(note = "use joint_limits::thigh::MIN instead")]
    pub const THIGH_MIN: f64 = thigh::MIN;
    #[deprecated(note = "use joint_limits::thigh::MAX instead")]
    pub const THIGH_MAX: f64 = thigh::MAX;
    #[deprecated(note = "use joint_limits::calf::MIN instead")]
    pub const CALF_MIN: f64 = calf::MIN;
    #[deprecated(note = "use joint_limits::calf::MAX instead")]
    pub const CALF_MAX: f64 = calf::MAX;
}

/// Motor names for the Go2
pub const MOTOR_NAMES: [&str; 12] = [
    "FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf", "RR_hip", "RR_thigh",
    "RR_calf", "RL_hip", "RL_thigh", "RL_calf",
];

/// Get motor index by name
pub fn motor_index_by_name(name: &str) -> Option<usize> {
    MOTOR_NAMES.iter().position(|&n| n == name)
}
