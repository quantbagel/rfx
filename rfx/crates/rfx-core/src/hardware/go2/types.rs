//! Type definitions for Go2 communication
//!
//! These types mirror the Unitree SDK structures for DDS communication.

use serde::{Deserialize, Serialize};

use super::super::motor_idx::NUM_MOTORS;

/// IMU sensor state
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ImuState {
    /// Quaternion orientation [w, x, y, z]
    pub quaternion: [f32; 4],
    /// Angular velocity [x, y, z] (rad/s)
    pub gyroscope: [f32; 3],
    /// Linear acceleration [x, y, z] (m/s²)
    pub accelerometer: [f32; 3],
    /// Roll, pitch, yaw angles (rad)
    pub rpy: [f32; 3],
    /// Temperature (°C)
    pub temperature: i8,
}

impl ImuState {
    /// Get roll angle in radians
    pub fn roll(&self) -> f32 {
        self.rpy[0]
    }

    /// Get pitch angle in radians
    pub fn pitch(&self) -> f32 {
        self.rpy[1]
    }

    /// Get yaw angle in radians
    pub fn yaw(&self) -> f32 {
        self.rpy[2]
    }

    /// Get roll angle in degrees
    pub fn roll_deg(&self) -> f32 {
        self.rpy[0].to_degrees()
    }

    /// Get pitch angle in degrees
    pub fn pitch_deg(&self) -> f32 {
        self.rpy[1].to_degrees()
    }

    /// Get yaw angle in degrees
    pub fn yaw_deg(&self) -> f32 {
        self.rpy[2].to_degrees()
    }
}

/// Single motor state
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MotorState {
    /// Motor mode
    pub mode: u8,
    /// Current position (rad)
    pub q: f32,
    /// Current velocity (rad/s)
    pub dq: f32,
    /// Commanded position (rad)
    pub q_raw: f32,
    /// Commanded velocity (rad/s)
    pub dq_raw: f32,
    /// Commanded torque (Nm)
    pub tau_raw: f32,
    /// Estimated torque (Nm)
    pub tau_est: f32,
    /// Motor temperature (°C)
    pub temperature: i8,
    /// Motor lost connection flag
    pub lost: u32,
}

/// Single motor command
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MotorCmd {
    /// Motor mode (0x00 = idle, 0x01 = position, 0x0A = damping)
    pub mode: u8,
    /// Target position (rad)
    pub q: f32,
    /// Target velocity (rad/s)
    pub dq: f32,
    /// Feedforward torque (Nm)
    pub tau: f32,
    /// Position gain
    pub kp: f32,
    /// Velocity gain
    pub kd: f32,
}

impl MotorCmd {
    /// Create a position command
    pub fn position(q: f32, kp: f32, kd: f32) -> Self {
        Self {
            mode: 0x01,
            q,
            dq: 0.0,
            tau: 0.0,
            kp,
            kd,
        }
    }

    /// Create a damping command (safe mode)
    pub fn damping(kd: f32) -> Self {
        Self {
            mode: 0x0A,
            q: 0.0,
            dq: 0.0,
            tau: 0.0,
            kp: 0.0,
            kd,
        }
    }

    /// Create an idle command
    pub fn idle() -> Self {
        Self::default()
    }
}

/// Low-level state from the robot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowState {
    /// Message counter
    pub tick: u32,
    /// IMU state
    pub imu: ImuState,
    /// Motor states for all 12 joints
    pub motor_state: [MotorState; NUM_MOTORS],
    /// Battery state of charge (%)
    pub bms_state: BmsState,
    /// Foot force sensors [FR, FL, RR, RL]
    pub foot_force: [i16; 4],
    /// Foot force sensors (estimated)
    pub foot_force_est: [i16; 4],
    /// Wireless remote state
    #[serde(
        serialize_with = "crate::hardware::go2::types::serialize_bytes_40",
        deserialize_with = "crate::hardware::go2::types::deserialize_bytes_40",
        default = "crate::hardware::go2::types::default_bytes_40"
    )]
    pub wireless_remote: [u8; 40],
}

fn default_bytes_40() -> [u8; 40] {
    [0u8; 40]
}

fn serialize_bytes_40<S: serde::Serializer>(bytes: &[u8; 40], s: S) -> Result<S::Ok, S::Error> {
    s.serialize_bytes(bytes)
}

fn deserialize_bytes_40<'de, D: serde::Deserializer<'de>>(d: D) -> Result<[u8; 40], D::Error> {
    let v: Vec<u8> = serde::Deserialize::deserialize(d)?;
    v.try_into().map_err(|v: Vec<u8>| {
        serde::de::Error::custom(format!("expected 40 bytes, got {}", v.len()))
    })
}

impl Default for LowState {
    fn default() -> Self {
        Self {
            tick: 0,
            imu: ImuState::default(),
            motor_state: [MotorState::default(); NUM_MOTORS],
            bms_state: BmsState::default(),
            foot_force: [0; 4],
            foot_force_est: [0; 4],
            wireless_remote: [0; 40],
        }
    }
}

/// Low-level command to the robot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowCmd {
    /// Message counter
    pub tick: u32,
    /// Motor commands for all 12 joints
    pub motor_cmd: [MotorCmd; NUM_MOTORS],
}

impl Default for LowCmd {
    fn default() -> Self {
        Self {
            tick: 0,
            motor_cmd: [MotorCmd::default(); NUM_MOTORS],
        }
    }
}

impl LowCmd {
    /// Set all motors to damping mode
    pub fn damping(kd: f32) -> Self {
        Self {
            motor_cmd: [MotorCmd::damping(kd); NUM_MOTORS],
            ..Default::default()
        }
    }

    /// Set all motors to idle
    pub fn idle() -> Self {
        Self::default()
    }
}

/// Battery Management System state
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BmsState {
    /// Battery version
    pub version_high: u8,
    pub version_low: u8,
    /// Battery status
    pub status: u8,
    /// State of charge (%)
    pub soc: u8,
    /// Current (mA)
    pub current: i32,
    /// Voltage (mV)
    pub voltage: u16,
    /// Temperature (°C * 10)
    pub temperature: i16,
    /// Cycle count
    pub cycle: u16,
}

impl BmsState {
    /// Get battery percentage
    pub fn percentage(&self) -> u8 {
        self.soc
    }

    /// Get voltage in volts
    pub fn voltage_v(&self) -> f32 {
        self.voltage as f32 / 1000.0
    }

    /// Get current in amps
    pub fn current_a(&self) -> f32 {
        self.current as f32 / 1000.0
    }

    /// Get temperature in celsius
    pub fn temperature_c(&self) -> f32 {
        self.temperature as f32 / 10.0
    }
}

/// High-level Go2 state (combines multiple sources)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Go2State {
    /// Message tick
    pub tick: u32,
    /// Timestamp in seconds
    pub timestamp: f64,
    /// IMU state
    pub imu: ImuState,
    /// Motor states
    pub motors: [MotorState; NUM_MOTORS],
    /// Battery state
    pub battery: BmsState,
    /// Estimated position [x, y, z] in world frame
    pub position: [f32; 3],
    /// Estimated velocity [vx, vy, vz] in world frame
    pub velocity: [f32; 3],
    /// Foot contact states [FR, FL, RR, RL]
    pub foot_contact: [bool; 4],
    /// Current robot mode
    pub mode: RobotMode,
    /// Sport mode state (if available)
    pub sport_mode: Option<SportModeState>,
}

impl Go2State {
    /// Create a new default state
    pub fn new() -> Self {
        Self {
            motors: [MotorState::default(); NUM_MOTORS],
            ..Default::default()
        }
    }

    /// Get joint positions as array
    pub fn joint_positions(&self) -> [f32; NUM_MOTORS] {
        std::array::from_fn(|i| self.motors[i].q)
    }

    /// Get joint velocities as array
    pub fn joint_velocities(&self) -> [f32; NUM_MOTORS] {
        std::array::from_fn(|i| self.motors[i].dq)
    }
}

/// Robot operating mode
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobotMode {
    #[default]
    Idle,
    Standing,
    Walking,
    Running,
    Climbing,
    LowLevel,
}

/// Sport mode state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SportModeState {
    /// Gait type
    pub gait_type: u8,
    /// Progress in current gait cycle (0-1)
    pub progress: f32,
    /// Body height (m)
    pub body_height: f32,
    /// Current velocity command [vx, vy, vyaw]
    pub velocity_cmd: [f32; 3],
}

/// Sport mode command
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SportModeCmd {
    /// Command mode
    pub mode: u8,
    /// Gait type (0=idle, 1=trot, 2=run, etc.)
    pub gait_type: u8,
    /// Speed level (0-2)
    pub speed_level: u8,
    /// Foot raise height (m)
    pub foot_raise_height: f32,
    /// Body height (m)
    pub body_height: f32,
    /// Position command [x, y, yaw]
    pub position: [f32; 3],
    /// Velocity command [vx, vy, vyaw]
    pub velocity: [f32; 3],
    /// Euler angles [roll, pitch, yaw]
    pub euler: [f32; 3],
}

impl SportModeCmd {
    /// Create a move command
    pub fn move_cmd(vx: f32, vy: f32, vyaw: f32) -> Self {
        Self {
            mode: 1,
            gait_type: 1, // Trot
            velocity: [vx, vy, vyaw],
            ..Default::default()
        }
    }

    /// Create a stand command
    pub fn stand() -> Self {
        Self {
            mode: 1,
            gait_type: 0,
            ..Default::default()
        }
    }

    /// Create a stop move command
    pub fn stop_move() -> Self {
        Self {
            mode: 0,
            ..Default::default()
        }
    }

    /// Set body height
    pub fn with_body_height(mut self, height: f32) -> Self {
        self.body_height = height;
        self
    }

    /// Set foot raise height
    pub fn with_foot_height(mut self, height: f32) -> Self {
        self.foot_raise_height = height;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motor_cmd_position() {
        let cmd = MotorCmd::position(1.0, 20.0, 0.5);
        assert_eq!(cmd.mode, 0x01);
        assert_eq!(cmd.q, 1.0);
        assert_eq!(cmd.kp, 20.0);
        assert_eq!(cmd.kd, 0.5);
    }

    #[test]
    fn test_motor_cmd_damping() {
        let cmd = MotorCmd::damping(5.0);
        assert_eq!(cmd.mode, 0x0A);
        assert_eq!(cmd.kd, 5.0);
    }

    #[test]
    fn test_low_cmd_damping() {
        let cmd = LowCmd::damping(3.0);
        for motor in &cmd.motor_cmd {
            assert_eq!(motor.mode, 0x0A);
            assert_eq!(motor.kd, 3.0);
        }
    }

    #[test]
    fn test_imu_degrees() {
        let imu = ImuState {
            rpy: [0.1, 0.2, 0.3],
            ..Default::default()
        };
        assert!((imu.roll_deg() - 5.729578).abs() < 0.001);
    }

    #[test]
    fn test_sport_mode_cmd() {
        let cmd = SportModeCmd::move_cmd(0.5, 0.1, 0.0);
        assert_eq!(cmd.velocity[0], 0.5);
        assert_eq!(cmd.velocity[1], 0.1);
        assert_eq!(cmd.gait_type, 1);
    }

    #[test]
    fn test_go2_state_new() {
        let state = Go2State::new();
        assert_eq!(state.motors.len(), NUM_MOTORS);
    }
}
