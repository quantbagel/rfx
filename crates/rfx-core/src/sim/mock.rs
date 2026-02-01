//! Mock simulation backend for testing

use crate::hardware::RobotState;
use crate::math::Transform;
use crate::Result;

use super::{SimBackend, SimConfig, SimInfo, SimState, StepResult};

/// A mock simulation backend for testing without a real simulator
///
/// Provides a simple physics approximation for testing control code.
#[derive(Debug)]
pub struct MockSimBackend {
    config: SimConfig,
    state: SimState,
    joint_positions: [f32; 12],
    joint_velocities: [f32; 12],
}

impl MockSimBackend {
    /// Create a new mock backend with default config
    pub fn new() -> Self {
        Self::with_config(SimConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: SimConfig) -> Self {
        let mut backend = Self {
            config,
            state: SimState::default(),
            joint_positions: [0.0; 12],
            joint_velocities: [0.0; 12],
        };
        backend.update_robot_state();
        backend
    }

    fn update_robot_state(&mut self) {
        self.state.robot = RobotState {
            pose: Transform::identity(),
            joint_positions: self.joint_positions.iter().map(|&p| p as f64).collect(),
            joint_velocities: self.joint_velocities.iter().map(|&v| v as f64).collect(),
            joint_torques: vec![0.0; 12],
            timestamp: self.state.sim_time,
        };
    }

    fn simple_physics_step(&mut self, actions: &[f32], dt: f64) {
        // Very simplified physics: PD control to target positions
        let kp = 20.0f32;
        let kd = 0.5f32;

        for i in 0..12.min(actions.len()) {
            let target = actions[i];
            let pos_error = target - self.joint_positions[i];
            let vel = self.joint_velocities[i];

            // Simple PD
            let accel = kp * pos_error - kd * vel;

            // Integrate
            self.joint_velocities[i] += accel * dt as f32;
            self.joint_positions[i] += self.joint_velocities[i] * dt as f32;

            // Clamp positions to reasonable range
            self.joint_positions[i] = self.joint_positions[i].clamp(-3.14, 3.14);
            self.joint_velocities[i] = self.joint_velocities[i].clamp(-20.0, 20.0);
        }
    }

    fn check_termination(&self) -> bool {
        // Simple termination: check if robot has fallen
        // In a real sim, this would check height, orientation, etc.
        false
    }
}

impl Default for MockSimBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SimBackend for MockSimBackend {
    fn name(&self) -> &str {
        "mock"
    }

    fn config(&self) -> &SimConfig {
        &self.config
    }

    fn reset(&mut self) -> Result<SimState> {
        // Reset to default standing position
        self.joint_positions = [0.0; 12];
        self.joint_velocities = [0.0; 12];
        self.state = SimState {
            sim_time: 0.0,
            terminated: false,
            truncated: false,
            reward: 0.0,
            info: SimInfo::default(),
            ..Default::default()
        };
        self.update_robot_state();
        Ok(self.state.clone())
    }

    fn step(&mut self, actions: &[f32]) -> Result<StepResult> {
        let dt = self.config.physics.dt;

        // Run physics substeps
        for _ in 0..self.config.physics.substeps {
            self.simple_physics_step(actions, dt / self.config.physics.substeps as f64);
        }

        // Update time
        self.state.sim_time += dt;

        // Update robot state
        self.update_robot_state();

        // Check termination
        self.state.terminated = self.check_termination();
        let done = self.state.terminated || self.state.truncated;

        Ok(StepResult {
            state: self.state.clone(),
            done,
        })
    }

    fn state(&self) -> SimState {
        self.state.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_backend_creation() {
        let backend = MockSimBackend::new();
        assert_eq!(backend.name(), "mock");
        assert_eq!(backend.state().sim_time, 0.0);
    }

    #[test]
    fn test_mock_backend_reset() {
        let mut backend = MockSimBackend::new();
        backend.step(&[0.5; 12]).unwrap();
        assert!(backend.state().sim_time > 0.0);

        backend.reset().unwrap();
        assert_eq!(backend.state().sim_time, 0.0);
    }

    #[test]
    fn test_mock_backend_step() {
        let mut backend = MockSimBackend::new();
        let actions = [0.5f32; 12];

        let result = backend.step(&actions).unwrap();
        assert!(!result.done);
        assert!(result.state.sim_time > 0.0);

        // Joints should move toward target
        for _ in 0..200 {
            backend.step(&actions).unwrap();
        }

        // After many steps, should be close to target
        let state = backend.state();
        assert!(state.robot.joint_positions[0] > 0.3);
    }
}
