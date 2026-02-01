//! Observation and action space definitions
//!
//! Defines the structure of inputs (observations) and outputs (actions)
//! for neural network policies.

use serde::{Deserialize, Serialize};

/// Type of space (continuous, discrete, etc.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpaceType {
    /// Continuous values in a range
    Box {
        low: Vec<f32>,
        high: Vec<f32>,
        shape: Vec<usize>,
    },
    /// Discrete integer values
    Discrete {
        n: usize,
    },
    /// Multiple discrete values
    MultiDiscrete {
        nvec: Vec<usize>,
    },
    /// Binary values
    MultiBinary {
        n: usize,
    },
}

/// A generic space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Space {
    /// Name of the space
    pub name: String,
    /// Type and bounds
    pub space_type: SpaceType,
    /// Data type (float32, float16, int32, etc.)
    pub dtype: String,
}

impl Space {
    /// Create a continuous box space
    pub fn box_space(name: impl Into<String>, shape: Vec<usize>, low: f32, high: f32) -> Self {
        let size: usize = shape.iter().product();
        Self {
            name: name.into(),
            space_type: SpaceType::Box {
                low: vec![low; size],
                high: vec![high; size],
                shape,
            },
            dtype: "float32".into(),
        }
    }

    /// Create a discrete space
    pub fn discrete(name: impl Into<String>, n: usize) -> Self {
        Self {
            name: name.into(),
            space_type: SpaceType::Discrete { n },
            dtype: "int32".into(),
        }
    }

    /// Get the total size of the space
    pub fn size(&self) -> usize {
        match &self.space_type {
            SpaceType::Box { shape, .. } => shape.iter().product(),
            SpaceType::Discrete { n } => *n,
            SpaceType::MultiDiscrete { nvec } => nvec.len(),
            SpaceType::MultiBinary { n } => *n,
        }
    }

    /// Get the shape of the space
    pub fn shape(&self) -> Vec<usize> {
        match &self.space_type {
            SpaceType::Box { shape, .. } => shape.clone(),
            SpaceType::Discrete { .. } => vec![1],
            SpaceType::MultiDiscrete { nvec } => vec![nvec.len()],
            SpaceType::MultiBinary { n } => vec![*n],
        }
    }

    /// Check if a value is within bounds
    pub fn contains(&self, value: &[f32]) -> bool {
        match &self.space_type {
            SpaceType::Box { low, high, .. } => {
                if value.len() != low.len() {
                    return false;
                }
                value
                    .iter()
                    .zip(low.iter().zip(high.iter()))
                    .all(|(v, (l, h))| *v >= *l && *v <= *h)
            }
            SpaceType::Discrete { n } => {
                value.len() == 1 && value[0] >= 0.0 && (value[0] as usize) < *n
            }
            _ => true, // Simplified for now
        }
    }

    /// Clip values to be within bounds
    pub fn clip(&self, value: &mut [f32]) {
        if let SpaceType::Box { low, high, .. } = &self.space_type {
            for (v, (l, h)) in value.iter_mut().zip(low.iter().zip(high.iter())) {
                *v = v.clamp(*l, *h);
            }
        }
    }
}

/// Observation space for a policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationSpace {
    /// Component spaces (e.g., "joint_pos", "joint_vel", "imu")
    pub components: Vec<Space>,
    /// Total observation dimension
    pub total_dim: usize,
}

impl ObservationSpace {
    /// Create a new observation space from components
    pub fn new(components: Vec<Space>) -> Self {
        let total_dim = components.iter().map(|s| s.size()).sum();
        Self {
            components,
            total_dim,
        }
    }

    /// Create a simple flat observation space
    pub fn flat(dim: usize, low: f32, high: f32) -> Self {
        Self::new(vec![Space::box_space("observation", vec![dim], low, high)])
    }

    /// Standard Go2 observation space (48-dim)
    pub fn go2_standard() -> Self {
        Self::new(vec![
            Space::box_space("joint_pos", vec![12], -3.14, 3.14),
            Space::box_space("joint_vel", vec![12], -20.0, 20.0),
            Space::box_space("base_ang_vel", vec![3], -10.0, 10.0),
            Space::box_space("projected_gravity", vec![3], -1.0, 1.0),
            Space::box_space("commands", vec![3], -1.0, 1.0),
            Space::box_space("last_actions", vec![12], -1.0, 1.0),
            Space::box_space("clock", vec![3], -1.0, 1.0), // sin/cos phase
        ])
    }
}

/// Action space for a policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpace {
    /// The action space definition
    pub space: Space,
    /// Whether actions are delta (relative) or absolute
    pub is_delta: bool,
    /// Scale factor for actions
    pub scale: f32,
}

impl ActionSpace {
    /// Create a new action space
    pub fn new(space: Space) -> Self {
        Self {
            space,
            is_delta: true,
            scale: 1.0,
        }
    }

    /// Create a standard Go2 action space (12 joint position targets)
    pub fn go2_standard() -> Self {
        Self {
            space: Space::box_space("actions", vec![12], -1.0, 1.0),
            is_delta: true,
            scale: 0.5, // Scale normalized actions to joint range
        }
    }

    /// Set whether actions are delta (relative) or absolute
    pub fn with_delta(mut self, is_delta: bool) -> Self {
        self.is_delta = is_delta;
        self
    }

    /// Set the action scale
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Get the action dimension
    pub fn dim(&self) -> usize {
        self.space.size()
    }

    /// Scale and clip actions
    pub fn process(&self, actions: &mut [f32]) {
        for a in actions.iter_mut() {
            *a *= self.scale;
        }
        self.space.clip(actions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_space() {
        let space = Space::box_space("test", vec![3], -1.0, 1.0);
        assert_eq!(space.size(), 3);
        assert!(space.contains(&[0.0, 0.5, -0.5]));
        assert!(!space.contains(&[2.0, 0.0, 0.0]));
    }

    #[test]
    fn test_observation_space() {
        let obs = ObservationSpace::go2_standard();
        assert_eq!(obs.total_dim, 48);
    }

    #[test]
    fn test_action_space() {
        let action = ActionSpace::go2_standard();
        assert_eq!(action.dim(), 12);

        let mut actions = vec![0.5f32; 12];
        action.process(&mut actions);
        assert!((actions[0] - 0.25).abs() < 0.001); // 0.5 * 0.5 scale
    }
}
