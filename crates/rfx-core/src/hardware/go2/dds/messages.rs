//! DDS wire-format message structures
//!
//! These structures match the Unitree SDK2 wire format for DDS communication.
//! The Go2 uses CDR (Common Data Representation) serialization.

use super::super::types::{BmsState, ImuState, LowCmd, LowState, MotorCmd, MotorState};
use super::crc::compute_crc;
use crate::hardware::motor_idx::NUM_MOTORS;

/// Number of motors in the DDS wire format (includes arm motors)
pub const DDS_NUM_MOTORS: usize = 20;

/// IMU state in DDS wire format
#[derive(Debug, Clone, Copy, Default)]
pub struct ImuStateDds {
    pub quaternion: [f32; 4],
    pub gyroscope: [f32; 3],
    pub accelerometer: [f32; 3],
    pub rpy: [f32; 3],
    pub temperature: i8,
}

impl From<ImuStateDds> for ImuState {
    fn from(dds: ImuStateDds) -> Self {
        Self {
            quaternion: dds.quaternion,
            gyroscope: dds.gyroscope,
            accelerometer: dds.accelerometer,
            rpy: dds.rpy,
            temperature: dds.temperature,
        }
    }
}

impl From<&ImuState> for ImuStateDds {
    fn from(state: &ImuState) -> Self {
        Self {
            quaternion: state.quaternion,
            gyroscope: state.gyroscope,
            accelerometer: state.accelerometer,
            rpy: state.rpy,
            temperature: state.temperature,
        }
    }
}

/// Motor state in DDS wire format
#[derive(Debug, Clone, Copy, Default)]
pub struct MotorStateDds {
    pub mode: u8,
    pub q: f32,
    pub dq: f32,
    pub ddq: f32,
    pub tau_est: f32,
    pub q_raw: f32,
    pub dq_raw: f32,
    pub ddq_raw: f32,
    pub temperature: i8,
    pub lost: u32,
    pub reserve: [u32; 2],
}

impl From<MotorStateDds> for MotorState {
    fn from(dds: MotorStateDds) -> Self {
        Self {
            mode: dds.mode,
            q: dds.q,
            dq: dds.dq,
            q_raw: dds.q_raw,
            dq_raw: dds.dq_raw,
            tau_raw: dds.ddq_raw,
            tau_est: dds.tau_est,
            temperature: dds.temperature,
            lost: dds.lost,
        }
    }
}

/// Motor command in DDS wire format
#[derive(Debug, Clone, Copy, Default)]
pub struct MotorCmdDds {
    pub mode: u8,
    pub q: f32,
    pub dq: f32,
    pub tau: f32,
    pub kp: f32,
    pub kd: f32,
    pub reserve: [u32; 3],
}

impl From<&MotorCmd> for MotorCmdDds {
    fn from(cmd: &MotorCmd) -> Self {
        Self {
            mode: cmd.mode,
            q: cmd.q,
            dq: cmd.dq,
            tau: cmd.tau,
            kp: cmd.kp,
            kd: cmd.kd,
            reserve: [0; 3],
        }
    }
}

impl From<MotorCmdDds> for MotorCmd {
    fn from(dds: MotorCmdDds) -> Self {
        Self {
            mode: dds.mode,
            q: dds.q,
            dq: dds.dq,
            tau: dds.tau,
            kp: dds.kp,
            kd: dds.kd,
        }
    }
}

/// BMS state in DDS wire format
#[derive(Debug, Clone, Copy, Default)]
pub struct BmsStateDds {
    pub version_high: u8,
    pub version_low: u8,
    pub status: u8,
    pub soc: u8,
    pub current: i32,
    pub cycle: u16,
    pub bq_ntc: [i8; 2],
    pub mcu_ntc: [i8; 2],
    pub cell_vol: [u16; 15],
}

impl From<BmsStateDds> for BmsState {
    fn from(dds: BmsStateDds) -> Self {
        let voltage: u32 = dds.cell_vol.iter().map(|&v| v as u32).sum();
        Self {
            version_high: dds.version_high,
            version_low: dds.version_low,
            status: dds.status,
            soc: dds.soc,
            current: dds.current,
            voltage: (voltage / 15) as u16,
            temperature: dds.bq_ntc[0] as i16 * 10,
            cycle: dds.cycle,
        }
    }
}

/// Low-level state in DDS wire format
#[derive(Debug, Clone)]
pub struct LowStateDds {
    pub head: [u8; 2],
    pub level_flag: u8,
    pub frame_reserve: u8,
    pub sn: [u32; 2],
    pub version: [u32; 2],
    pub bandwidth: u16,
    pub imu_state: ImuStateDds,
    pub motor_state: Vec<MotorStateDds>,
    pub bms_state: BmsStateDds,
    pub foot_force: [i16; 4],
    pub foot_force_est: [i16; 4],
    pub tick: u32,
    pub wireless_remote: Vec<u8>,
    pub bit_flag: u8,
    pub aios_state: f32,
    pub power_v: u16,
    pub reserve: Vec<u8>,
    pub crc: u32,
}

impl Default for LowStateDds {
    fn default() -> Self {
        Self {
            head: [0; 2],
            level_flag: 0,
            frame_reserve: 0,
            sn: [0; 2],
            version: [0; 2],
            bandwidth: 0,
            imu_state: ImuStateDds::default(),
            motor_state: vec![MotorStateDds::default(); DDS_NUM_MOTORS],
            bms_state: BmsStateDds::default(),
            foot_force: [0; 4],
            foot_force_est: [0; 4],
            tick: 0,
            wireless_remote: vec![0; 40],
            bit_flag: 0,
            aios_state: 0.0,
            power_v: 0,
            reserve: vec![0; 16],
            crc: 0,
        }
    }
}

impl LowStateDds {
    pub fn to_low_state(&self) -> LowState {
        let mut motor_state = [MotorState::default(); NUM_MOTORS];
        for (i, ms) in self.motor_state.iter().take(NUM_MOTORS).enumerate() {
            motor_state[i] = MotorState::from(*ms);
        }

        LowState {
            tick: self.tick,
            imu: ImuState::from(self.imu_state),
            motor_state,
            bms_state: BmsState::from(self.bms_state),
            foot_force: self.foot_force,
            foot_force_est: self.foot_force_est,
            wireless_remote: self.wireless_remote.clone(),
        }
    }
}

/// Low-level command in DDS wire format
#[derive(Debug, Clone)]
pub struct LowCmdDds {
    pub head: [u8; 2],
    pub level_flag: u8,
    pub frame_reserve: u8,
    pub sn: [u32; 2],
    pub version: [u32; 2],
    pub bandwidth: u16,
    pub motor_cmd: Vec<MotorCmdDds>,
    pub bms_cmd: BmsCmdDds,
    pub wireless_remote: Vec<u8>,
    pub led: Vec<u8>,
    pub fan: [u8; 2],
    pub gpio: u8,
    pub reserve: u32,
    pub crc: u32,
}

/// BMS command in DDS wire format
#[derive(Debug, Clone, Copy, Default)]
pub struct BmsCmdDds {
    pub off: u8,
    pub reserve: [u8; 3],
}

impl Default for LowCmdDds {
    fn default() -> Self {
        Self {
            head: [0xFE, 0xEF],
            level_flag: 0xFF,
            frame_reserve: 0,
            sn: [0; 2],
            version: [0; 2],
            bandwidth: 0,
            motor_cmd: vec![MotorCmdDds::default(); DDS_NUM_MOTORS],
            bms_cmd: BmsCmdDds::default(),
            wireless_remote: vec![0; 40],
            led: vec![0; 12],
            fan: [0; 2],
            gpio: 0,
            reserve: 0,
            crc: 0,
        }
    }
}

impl LowCmdDds {
    pub fn from_low_cmd(cmd: &LowCmd) -> Self {
        let mut dds = Self::default();
        dds.head = [0xFE, 0xEF];
        dds.level_flag = 0xFF;

        for (i, mc) in cmd.motor_cmd.iter().enumerate().take(NUM_MOTORS) {
            dds.motor_cmd[i] = MotorCmdDds::from(mc);
        }

        dds.crc = dds.compute_crc();
        dds
    }

    pub fn compute_crc(&self) -> u32 {
        let bytes = self.to_bytes_for_crc();
        compute_crc(&bytes)
    }

    fn to_bytes_for_crc(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(1024);

        bytes.extend_from_slice(&self.head);
        bytes.push(self.level_flag);
        bytes.push(self.frame_reserve);

        for &sn in &self.sn {
            bytes.extend_from_slice(&sn.to_le_bytes());
        }
        for &ver in &self.version {
            bytes.extend_from_slice(&ver.to_le_bytes());
        }
        bytes.extend_from_slice(&self.bandwidth.to_le_bytes());

        for mc in &self.motor_cmd {
            bytes.push(mc.mode);
            bytes.extend_from_slice(&mc.q.to_le_bytes());
            bytes.extend_from_slice(&mc.dq.to_le_bytes());
            bytes.extend_from_slice(&mc.tau.to_le_bytes());
            bytes.extend_from_slice(&mc.kp.to_le_bytes());
            bytes.extend_from_slice(&mc.kd.to_le_bytes());
            for &r in &mc.reserve {
                bytes.extend_from_slice(&r.to_le_bytes());
            }
        }

        bytes.push(self.bms_cmd.off);
        bytes.extend_from_slice(&self.bms_cmd.reserve);
        bytes.extend_from_slice(&self.wireless_remote);
        bytes.extend_from_slice(&self.led);
        bytes.extend_from_slice(&self.fan);
        bytes.push(self.gpio);
        bytes.extend_from_slice(&self.reserve.to_le_bytes());

        bytes
    }
}

impl From<&LowCmd> for LowCmdDds {
    fn from(cmd: &LowCmd) -> Self {
        LowCmdDds::from_low_cmd(cmd)
    }
}

/// Sport mode request in DDS wire format
#[derive(Debug, Clone)]
pub struct SportModeRequestDds {
    pub header: RequestHeaderDds,
    pub parameter: String,
}

#[derive(Debug, Clone, Default)]
pub struct RequestHeaderDds {
    pub identity: IdentityDds,
    pub api_id: i32,
}

#[derive(Debug, Clone, Default)]
pub struct IdentityDds {
    pub id: i64,
    pub api_id: i32,
}

impl SportModeRequestDds {
    pub fn move_cmd(vx: f32, vy: f32, vyaw: f32) -> Self {
        let param = format!(r#"{{"x":{},"y":{},"z":{}}}"#, vx, vy, vyaw);
        Self {
            header: RequestHeaderDds {
                identity: IdentityDds { id: 0, api_id: 1008 },
                api_id: 1008,
            },
            parameter: param,
        }
    }

    pub fn stand() -> Self {
        Self {
            header: RequestHeaderDds {
                identity: IdentityDds { id: 0, api_id: 1004 },
                api_id: 1004,
            },
            parameter: String::new(),
        }
    }

    pub fn stop() -> Self {
        Self {
            header: RequestHeaderDds {
                identity: IdentityDds { id: 0, api_id: 1002 },
                api_id: 1002,
            },
            parameter: String::new(),
        }
    }

    pub fn sit() -> Self {
        Self {
            header: RequestHeaderDds {
                identity: IdentityDds { id: 0, api_id: 1003 },
                api_id: 1003,
            },
            parameter: String::new(),
        }
    }

    pub fn damp() -> Self {
        Self {
            header: RequestHeaderDds {
                identity: IdentityDds { id: 0, api_id: 1001 },
                api_id: 1001,
            },
            parameter: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motor_cmd_conversion() {
        let cmd = MotorCmd {
            mode: 0x01,
            q: 1.5,
            dq: 0.5,
            tau: 0.1,
            kp: 20.0,
            kd: 0.5,
        };

        let dds = MotorCmdDds::from(&cmd);
        assert_eq!(dds.mode, 0x01);
        assert_eq!(dds.q, 1.5);
        assert_eq!(dds.kp, 20.0);

        let back = MotorCmd::from(dds);
        assert_eq!(back.mode, cmd.mode);
        assert_eq!(back.q, cmd.q);
    }

    #[test]
    fn test_low_cmd_dds_default() {
        let cmd = LowCmdDds::default();
        assert_eq!(cmd.head, [0xFE, 0xEF]);
        assert_eq!(cmd.level_flag, 0xFF);
        assert_eq!(cmd.motor_cmd.len(), DDS_NUM_MOTORS);
    }

    #[test]
    fn test_low_cmd_crc() {
        let cmd = LowCmdDds::default();
        let crc = cmd.compute_crc();
        assert_eq!(crc, cmd.compute_crc());
    }

    #[test]
    fn test_sport_mode_request() {
        let req = SportModeRequestDds::move_cmd(0.5, 0.0, 0.0);
        assert_eq!(req.header.api_id, 1008);
        assert!(req.parameter.contains("0.5"));

        let stand = SportModeRequestDds::stand();
        assert_eq!(stand.header.api_id, 1004);
    }
}
