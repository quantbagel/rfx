//! Simulation backend trait

use crate::hardware::RobotState;
use crate::Result;
use serde::{Deserialize, Serialize};

use super::SimConfig;

/// State returned from a simulation step
#[derive(Debug, Clone, Default)]
pub struct SimState {
    /// Robot state (joint positions, velocities, etc.)
    pub robot: RobotState,
    /// Simulation time in seconds
    pub sim_time: f64,
    /// Episode terminated (fell over, out of bounds, etc.)
    pub terminated: bool,
    /// Episode truncated (time limit reached)
    pub truncated: bool,
    /// Reward (if using RL)
    pub reward: f64,
    /// Additional info
    pub info: SimInfo,
}

/// Additional simulation info
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimInfo {
    /// Number of contact points
    pub contact_count: usize,
    /// Total contact force magnitude
    pub contact_force: f64,
    /// Distance traveled
    pub distance: f64,
    /// Energy consumed
    pub energy: f64,
}

/// Result from stepping the simulation
#[derive(Debug, Clone)]
pub struct StepResult {
    /// New state after step
    pub state: SimState,
    /// Whether the episode is done (terminated or truncated)
    pub done: bool,
}

/// Trait for simulation backends
///
/// Implementations provide the bridge between pi and various physics simulators.
pub trait SimBackend: Send + Sync {
    /// Get the backend name
    fn name(&self) -> &str;

    /// Get the current configuration
    fn config(&self) -> &SimConfig;

    /// Reset the simulation to initial state
    fn reset(&mut self) -> Result<SimState>;

    /// Step the simulation with the given actions
    ///
    /// # Arguments
    /// * `actions` - Joint position targets (12 values for Go2)
    fn step(&mut self, actions: &[f32]) -> Result<StepResult>;

    /// Get current state without stepping
    fn state(&self) -> SimState;

    /// Get current simulation time
    fn sim_time(&self) -> f64 {
        self.state().sim_time
    }

    /// Check if rendering is enabled
    fn is_rendering(&self) -> bool {
        self.config().render.enabled
    }

    /// Enable/disable rendering
    fn set_rendering(&mut self, _enabled: bool) {
        // Default: no-op
    }

    /// Render a frame (if rendering is enabled)
    fn render(&mut self) -> Result<()> {
        Ok(())
    }

    /// Close the simulation
    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// Get the number of parallel environments
    fn num_envs(&self) -> usize {
        self.config().num_envs
    }

    /// Step multiple environments in parallel
    fn step_batch(&mut self, actions: &[Vec<f32>]) -> Result<Vec<StepResult>> {
        // Default: step sequentially
        actions.iter().map(|a| self.step(a)).collect()
    }

    /// Reset specific environments
    fn reset_envs(&mut self, _env_indices: &[usize]) -> Result<Vec<SimState>> {
        // Default: reset all
        Ok(vec![self.reset()?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sim_state() {
        let state = SimState::default();
        assert_eq!(state.sim_time, 0.0);
        assert!(!state.terminated);
    }

    #[test]
    fn test_step_result() {
        let result = StepResult {
            state: SimState::default(),
            done: false,
        };
        assert!(!result.done);
    }
}
