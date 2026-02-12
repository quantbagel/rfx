//! Hardware abstraction traits
//!
//! Defines traits for robot hardware and simulators to enable
//! the same code to run on real hardware or in simulation.

use arrayvec::ArrayVec;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::math::Transform;
use crate::Result;

/// Maximum number of joints supported by the generic robot state
pub const MAX_JOINTS: usize = 24;

/// Generic robot state
#[non_exhaustive]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RobotState {
    /// Base pose in world frame
    pub pose: Transform,
    /// Joint positions (radians)
    pub joint_positions: ArrayVec<f64, MAX_JOINTS>,
    /// Joint velocities (rad/s)
    pub joint_velocities: ArrayVec<f64, MAX_JOINTS>,
    /// Joint torques (Nm)
    pub joint_torques: ArrayVec<f64, MAX_JOINTS>,
    /// Timestamp in seconds
    pub timestamp: f64,
}

impl RobotState {
    /// Create a new robot state with the given number of joints
    pub fn new(num_joints: usize) -> Self {
        assert!(num_joints <= MAX_JOINTS, "too many joints");
        let mut joint_positions = ArrayVec::new();
        let mut joint_velocities = ArrayVec::new();
        let mut joint_torques = ArrayVec::new();
        for _ in 0..num_joints {
            joint_positions.push(0.0);
            joint_velocities.push(0.0);
            joint_torques.push(0.0);
        }
        Self {
            pose: Transform::identity(),
            joint_positions,
            joint_velocities,
            joint_torques,
            timestamp: 0.0,
        }
    }

    /// Get a joint position by index
    #[must_use]
    pub fn joint_position(&self, index: usize) -> Option<f64> {
        self.joint_positions.get(index).copied()
    }

    /// Get a joint velocity by index
    #[must_use]
    pub fn joint_velocity(&self, index: usize) -> Option<f64> {
        self.joint_velocities.get(index).copied()
    }
}

/// Generic robot command
#[non_exhaustive]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Command {
    /// Target joint positions (radians)
    pub positions: Option<ArrayVec<f64, MAX_JOINTS>>,
    /// Target joint velocities (rad/s)
    pub velocities: Option<ArrayVec<f64, MAX_JOINTS>>,
    /// Feedforward torques (Nm)
    pub torques: Option<ArrayVec<f64, MAX_JOINTS>>,
    /// Position gains (kp)
    pub kp: Option<ArrayVec<f64, MAX_JOINTS>>,
    /// Velocity gains (kd)
    pub kd: Option<ArrayVec<f64, MAX_JOINTS>>,
}

impl Command {
    /// Create a position command
    pub fn position(positions: &[f64]) -> Self {
        let mut av = ArrayVec::new();
        av.try_extend_from_slice(positions)
            .expect("too many joints");
        Self {
            positions: Some(av),
            ..Default::default()
        }
    }

    /// Create a position command with gains
    pub fn position_with_gains(positions: &[f64], kp: &[f64], kd: &[f64]) -> Self {
        let mut pos_av = ArrayVec::new();
        pos_av
            .try_extend_from_slice(positions)
            .expect("too many joints");
        let mut kp_av = ArrayVec::new();
        kp_av.try_extend_from_slice(kp).expect("too many joints");
        let mut kd_av = ArrayVec::new();
        kd_av.try_extend_from_slice(kd).expect("too many joints");
        Self {
            positions: Some(pos_av),
            kp: Some(kp_av),
            kd: Some(kd_av),
            ..Default::default()
        }
    }

    /// Create a velocity command
    pub fn velocity(velocities: &[f64]) -> Self {
        let mut av = ArrayVec::new();
        av.try_extend_from_slice(velocities)
            .expect("too many joints");
        Self {
            velocities: Some(av),
            ..Default::default()
        }
    }

    /// Create a torque command
    pub fn torque(torques: &[f64]) -> Self {
        let mut av = ArrayVec::new();
        av.try_extend_from_slice(torques).expect("too many joints");
        Self {
            torques: Some(av),
            ..Default::default()
        }
    }

    /// Set default gains for all joints
    pub fn with_default_gains(mut self, kp: f64, kd: f64, num_joints: usize) -> Self {
        let mut kp_av = ArrayVec::new();
        for _ in 0..num_joints {
            kp_av.push(kp);
        }
        let mut kd_av = ArrayVec::new();
        for _ in 0..num_joints {
            kd_av.push(kd);
        }
        self.kp = Some(kp_av);
        self.kd = Some(kd_av);
        self
    }
}

/// Trait for robot hardware abstraction
///
/// This trait defines the interface that all robot implementations must provide,
/// enabling the same high-level code to work with different robots or simulators.
pub trait Robot: Send + Sync {
    /// Get the current robot state
    fn state(&self) -> RobotState;

    /// Send a command to the robot
    fn send_command(&self, cmd: Command) -> Result<()>;

    /// Get the number of joints
    fn num_joints(&self) -> usize;

    /// Get the robot name/type
    fn name(&self) -> &str;

    /// Check if the robot is connected/ready
    #[must_use]
    fn is_ready(&self) -> bool;

    /// Emergency stop
    fn emergency_stop(&self) -> Result<()>;

    /// Reset to a safe state
    fn reset(&self) -> Result<()>;
}

/// Trait for simulation backends
///
/// Extends the Robot trait with simulation-specific functionality.
pub trait Simulator: Robot {
    /// Step the simulation forward by the given time
    fn step(&mut self, dt: f64);

    /// Reset the simulation to initial state
    fn reset_sim(&mut self);

    /// Set the simulation time step
    fn set_timestep(&mut self, dt: f64);

    /// Get the current simulation time
    fn sim_time(&self) -> f64;

    /// Enable/disable rendering (if applicable)
    fn set_rendering(&mut self, enabled: bool);

    /// Check if the simulation is running in real-time
    #[must_use]
    fn is_realtime(&self) -> bool;

    /// Set real-time mode
    fn set_realtime(&mut self, enabled: bool);
}

/// A mock robot for testing
#[derive(Debug, Default)]
pub struct MockRobot {
    state: RwLock<RobotState>,
    num_joints: usize,
    ready: std::sync::atomic::AtomicBool,
}

impl MockRobot {
    /// Create a new mock robot
    pub fn new(num_joints: usize) -> Self {
        Self {
            state: RwLock::new(RobotState::new(num_joints)),
            num_joints,
            ready: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Set the robot state
    pub fn set_state(&self, state: RobotState) {
        *self.state.write() = state;
    }

    /// Set the ready status
    pub fn set_ready(&self, ready: bool) {
        self.ready
            .store(ready, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Robot for MockRobot {
    fn state(&self) -> RobotState {
        self.state.read().clone()
    }

    fn send_command(&self, cmd: Command) -> Result<()> {
        let mut state = self.state.write();
        if let Some(positions) = cmd.positions {
            state.joint_positions = positions;
        }
        Ok(())
    }

    fn num_joints(&self) -> usize {
        self.num_joints
    }

    fn name(&self) -> &str {
        "MockRobot"
    }

    fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn emergency_stop(&self) -> Result<()> {
        self.set_ready(false);
        Ok(())
    }

    fn reset(&self) -> Result<()> {
        *self.state.write() = RobotState::new(self.num_joints);
        self.set_ready(true);
        Ok(())
    }
}

impl Simulator for MockRobot {
    fn step(&mut self, dt: f64) {
        let mut state = self.state.write();
        state.timestamp += dt;
    }

    fn reset_sim(&mut self) {
        *self.state.write() = RobotState::new(self.num_joints);
    }

    fn set_timestep(&mut self, _dt: f64) {
        // No-op for mock
    }

    fn sim_time(&self) -> f64 {
        self.state.read().timestamp
    }

    fn set_rendering(&mut self, _enabled: bool) {
        // No-op for mock
    }

    fn is_realtime(&self) -> bool {
        false
    }

    fn set_realtime(&mut self, _enabled: bool) {
        // No-op for mock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robot_state() {
        let state = RobotState::new(12);
        assert_eq!(state.joint_positions.len(), 12);
        assert_eq!(state.joint_position(0), Some(0.0));
        assert_eq!(state.joint_position(20), None);
    }

    #[test]
    fn test_command() {
        let cmd = Command::position(&[1.0, 2.0, 3.0]);
        let mut expected = ArrayVec::<f64, MAX_JOINTS>::new();
        expected.try_extend_from_slice(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(cmd.positions, Some(expected));
        assert!(cmd.velocities.is_none());
    }

    #[test]
    fn test_mock_robot() {
        let robot = MockRobot::new(6);
        assert!(robot.is_ready());
        assert_eq!(robot.num_joints(), 6);

        let cmd = Command::position(&[1.0; 6]);
        robot.send_command(cmd).unwrap();

        let state = robot.state();
        let mut expected = ArrayVec::<f64, MAX_JOINTS>::new();
        expected.try_extend_from_slice(&[1.0; 6]).unwrap();
        assert_eq!(state.joint_positions, expected);
    }
}
