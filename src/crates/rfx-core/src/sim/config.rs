//! Simulation configuration

use serde::{Deserialize, Serialize};

/// Physics simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    /// Simulation timestep in seconds
    pub dt: f64,
    /// Number of physics substeps per step
    pub substeps: usize,
    /// Gravity vector [x, y, z] in m/s²
    pub gravity: [f64; 3],
    /// Enable contact dynamics
    pub enable_contacts: bool,
    /// Contact stiffness
    pub contact_stiffness: f64,
    /// Contact damping
    pub contact_damping: f64,
    /// Friction coefficient
    pub friction: f64,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            dt: 0.002,   // 500 Hz
            substeps: 4, // Effective 2000 Hz physics
            gravity: [0.0, 0.0, -9.81],
            enable_contacts: true,
            contact_stiffness: 1e4,
            contact_damping: 1e2,
            friction: 1.0,
        }
    }
}

impl PhysicsConfig {
    /// Create a fast simulation config (lower accuracy)
    pub fn fast() -> Self {
        Self {
            dt: 0.005,
            substeps: 2,
            ..Default::default()
        }
    }

    /// Create a high-accuracy simulation config
    pub fn accurate() -> Self {
        Self {
            dt: 0.001,
            substeps: 8,
            ..Default::default()
        }
    }

    /// Set the timestep
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Set gravity
    pub fn with_gravity(mut self, gx: f64, gy: f64, gz: f64) -> Self {
        self.gravity = [gx, gy, gz];
        self
    }
}

/// Rendering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    /// Enable rendering
    pub enabled: bool,
    /// Window width
    pub width: u32,
    /// Window height
    pub height: u32,
    /// Target FPS
    pub fps: u32,
    /// Enable shadows
    pub shadows: bool,
    /// Camera position [x, y, z]
    pub camera_pos: [f64; 3],
    /// Camera target [x, y, z]
    pub camera_target: [f64; 3],
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            width: 1280,
            height: 720,
            fps: 60,
            shadows: true,
            camera_pos: [2.0, 2.0, 1.5],
            camera_target: [0.0, 0.0, 0.3],
        }
    }
}

impl RenderConfig {
    /// Create a headless config (no rendering)
    pub fn headless() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Enable rendering with default settings
    pub fn with_rendering() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }
}

/// Simulation backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimBackendType {
    Mock,
    MuJoCo,
    IsaacSim,
    Genesis,
}

impl Default for SimBackendType {
    fn default() -> Self {
        Self::Mock
    }
}

impl std::fmt::Display for SimBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mock => write!(f, "mock"),
            Self::MuJoCo => write!(f, "mujoco"),
            Self::IsaacSim => write!(f, "isaac_sim"),
            Self::Genesis => write!(f, "genesis"),
        }
    }
}

/// Overall simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    /// Simulator backend type
    pub backend: SimBackendType,
    /// Physics configuration
    pub physics: PhysicsConfig,
    /// Rendering configuration
    pub render: RenderConfig,
    /// Robot URDF/USD path
    pub robot_asset: Option<String>,
    /// Environment/terrain type
    pub terrain: TerrainConfig,
    /// Number of parallel environments
    pub num_envs: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            backend: SimBackendType::Mock,
            physics: PhysicsConfig::default(),
            render: RenderConfig::default(),
            robot_asset: None,
            terrain: TerrainConfig::default(),
            num_envs: 1,
            seed: 42,
        }
    }
}

impl SimConfig {
    /// Create an Isaac Sim configuration
    pub fn isaac_sim() -> Self {
        Self {
            backend: SimBackendType::IsaacSim,
            physics: PhysicsConfig::accurate(),
            render: RenderConfig::with_rendering(),
            ..Default::default()
        }
    }

    /// Create a Genesis configuration
    pub fn genesis() -> Self {
        Self {
            backend: SimBackendType::Genesis,
            physics: PhysicsConfig::default(),
            render: RenderConfig::headless(),
            ..Default::default()
        }
    }

    /// Create a MuJoCo configuration
    pub fn mujoco() -> Self {
        Self {
            backend: SimBackendType::MuJoCo,
            physics: PhysicsConfig::accurate(),
            ..Default::default()
        }
    }

    /// Set number of parallel environments
    pub fn with_num_envs(mut self, n: usize) -> Self {
        self.num_envs = n;
        self
    }

    /// Set robot asset path
    pub fn with_robot(mut self, path: impl Into<String>) -> Self {
        self.robot_asset = Some(path.into());
        self
    }
}

/// Terrain/environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainConfig {
    /// Terrain type
    pub terrain_type: TerrainType,
    /// Terrain size [width, length]
    pub size: [f64; 2],
    /// Height variation (for rough terrain)
    pub roughness: f64,
    /// Slope angle in degrees (for slopes)
    pub slope_angle: f64,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            terrain_type: TerrainType::Flat,
            size: [10.0, 10.0],
            roughness: 0.0,
            slope_angle: 0.0,
        }
    }
}

/// Types of terrain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerrainType {
    /// Flat ground plane
    Flat,
    /// Rough terrain with height variations
    Rough,
    /// Sloped terrain
    Slope,
    /// Stairs
    Stairs,
    /// Custom heightmap
    Heightmap,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_config() {
        let config = PhysicsConfig::default();
        assert_eq!(config.dt, 0.002);
        assert_eq!(config.gravity[2], -9.81);
    }

    #[test]
    fn test_sim_config() {
        let config = SimConfig::isaac_sim()
            .with_num_envs(4096)
            .with_robot("go2.usd");

        assert_eq!(config.backend, SimBackendType::IsaacSim);
        assert_eq!(config.num_envs, 4096);
        assert!(config.render.enabled);
    }
}
