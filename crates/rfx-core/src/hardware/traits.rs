//! Hardware abstraction traits
//!
//! Defines traits for robot hardware and simulators to enable
//! the same code to run on real hardware or in simulation.

use serde::{Deserialize, Serialize};

use crate::math::Transform;
use crate::Result;

/// Generic robot state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RobotState {
    /// Base pose in world frame
    pub pose: Transform,
    /// Joint positions (radians)
    pub joint_positions: Vec<f64>,
    /// Joint velocities (rad/s)
    pub joint_velocities: Vec<f64>,
    /// Joint torques (Nm)
    pub joint_torques: Vec<f64>,
    /// Timestamp in seconds
    pub timestamp: f64,
}

impl RobotState {
    /// Create a new robot state with the given number of joints
    pub fn new(num_joints: usize) -> Self {
        Self {
            pose: Transform::identity(),
            joint_positions: vec![0.0; num_joints],
            joint_velocities: vec![0.0; num_joints],
            joint_torques: vec![0.0; num_joints],
            timestamp: 0.0,
        }
    }

    /// Get a joint position by index
    pub fn joint_position(&self, index: usize) -> Option<f64> {
        self.joint_positions.get(index).copied()
    }

    /// Get a joint velocity by index
    pub fn joint_velocity(&self, index: usize) -> Option<f64> {
        self.joint_velocities.get(index).copied()
    }
}

/// Generic robot command
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Command {
    /// Target joint positions (radians)
    pub positions: Option<Vec<f64>>,
    /// Target joint velocities (rad/s)
    pub velocities: Option<Vec<f64>>,
    /// Feedforward torques (Nm)
    pub torques: Option<Vec<f64>>,
    /// Position gains (kp)
    pub kp: Option<Vec<f64>>,
    /// Velocity gains (kd)
    pub kd: Option<Vec<f64>>,
}

impl Command {
    /// Create a position command
    pub fn position(positions: Vec<f64>) -> Self {
        Self {
            positions: Some(positions),
            ..Default::default()
        }
    }

    /// Create a position command with gains
    pub fn position_with_gains(positions: Vec<f64>, kp: Vec<f64>, kd: Vec<f64>) -> Self {
        Self {
            positions: Some(positions),
            kp: Some(kp),
            kd: Some(kd),
            ..Default::default()
        }
    }

    /// Create a velocity command
    pub fn velocity(velocities: Vec<f64>) -> Self {
        Self {
            velocities: Some(velocities),
            ..Default::default()
        }
    }

    /// Create a torque command
    pub fn torque(torques: Vec<f64>) -> Self {
        Self {
            torques: Some(torques),
            ..Default::default()
        }
    }

    /// Set default gains for all joints
    pub fn with_default_gains(mut self, kp: f64, kd: f64, num_joints: usize) -> Self {
        self.kp = Some(vec![kp; num_joints]);
        self.kd = Some(vec![kd; num_joints]);
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
    fn is_realtime(&self) -> bool;

    /// Set real-time mode
    fn set_realtime(&mut self, enabled: bool);
}

/// A mock robot for testing
#[derive(Debug, Default)]
pub struct MockRobot {
    state: std::sync::RwLock<RobotState>,
    num_joints: usize,
    ready: std::sync::atomic::AtomicBool,
}

impl MockRobot {
    /// Create a new mock robot
    pub fn new(num_joints: usize) -> Self {
        Self {
            state: std::sync::RwLock::new(RobotState::new(num_joints)),
            num_joints,
            ready: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Set the robot state
    pub fn set_state(&self, state: RobotState) {
        *self.state.write().unwrap() = state;
    }

    /// Set the ready status
    pub fn set_ready(&self, ready: bool) {
        self.ready
            .store(ready, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Robot for MockRobot {
    fn state(&self) -> RobotState {
        self.state.read().unwrap().clone()
    }

    fn send_command(&self, cmd: Command) -> Result<()> {
        let mut state = self.state.write().unwrap();
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
        *self.state.write().unwrap() = RobotState::new(self.num_joints);
        self.set_ready(true);
        Ok(())
    }
}

impl Simulator for MockRobot {
    fn step(&mut self, dt: f64) {
        let mut state = self.state.write().unwrap();
        state.timestamp += dt;
    }

    fn reset_sim(&mut self) {
        *self.state.write().unwrap() = RobotState::new(self.num_joints);
    }

    fn set_timestep(&mut self, _dt: f64) {
        // No-op for mock
    }

    fn sim_time(&self) -> f64 {
        self.state.read().unwrap().timestamp
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
        let cmd = Command::position(vec![1.0, 2.0, 3.0]);
        assert_eq!(cmd.positions, Some(vec![1.0, 2.0, 3.0]));
        assert!(cmd.velocities.is_none());
    }

    #[test]
    fn test_mock_robot() {
        let robot = MockRobot::new(6);
        assert!(robot.is_ready());
        assert_eq!(robot.num_joints(), 6);

        let cmd = Command::position(vec![1.0; 6]);
        robot.send_command(cmd).unwrap();

        let state = robot.state();
        assert_eq!(state.joint_positions, vec![1.0; 6]);
    }
}
