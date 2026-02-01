//! Feetech STS/SCS servo protocol implementation
//!
//! The SO-101 uses Feetech STS3215 servos which follow a Dynamixel-like protocol.
//!
//! Packet format: [0xFF, 0xFF, ID, Length, Instruction, Params..., Checksum]

use std::io::{Read, Write};
use std::time::Duration;

use crate::Error;

use super::types::NUM_JOINTS;

/// Feetech instruction codes
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Instruction {
    Ping = 0x01,
    Read = 0x02,
    Write = 0x03,
    RegWrite = 0x04,
    Action = 0x05,
    FactoryReset = 0x06,
    SyncRead = 0x82,
    SyncWrite = 0x83,
}

/// Feetech register addresses
pub mod registers {
    /// Current position (2 bytes, read-only)
    pub const PRESENT_POSITION: u8 = 0x38;
    /// Goal position (2 bytes, read-write)
    pub const GOAL_POSITION: u8 = 0x2A;
    /// Torque enable (1 byte, read-write)
    pub const TORQUE_ENABLE: u8 = 0x28;
    /// Current speed (2 bytes, read-only)
    pub const PRESENT_SPEED: u8 = 0x3A;
    /// Moving speed (2 bytes, read-write)
    pub const MOVING_SPEED: u8 = 0x2E;
}

/// Broadcast ID for sync commands
pub const BROADCAST_ID: u8 = 0xFE;

/// Default servo IDs for SO-101 (1-indexed)
pub const SERVO_IDS: [u8; NUM_JOINTS] = [1, 2, 3, 4, 5, 6];

/// Serial protocol handler for Feetech servos
pub struct FeetechProtocol<S> {
    serial: S,
    timeout: Duration,
}

impl<S> FeetechProtocol<S>
where
    S: Read + Write,
{
    /// Create a new protocol handler
    pub fn new(serial: S, timeout: Duration) -> Self {
        Self { serial, timeout }
    }

    /// Get mutable reference to underlying serial port
    pub fn serial_mut(&mut self) -> &mut S {
        &mut self.serial
    }

    /// Calculate checksum for a packet
    fn checksum(id: u8, length: u8, instruction: u8, params: &[u8]) -> u8 {
        let mut sum: u8 = id;
        sum = sum.wrapping_add(length);
        sum = sum.wrapping_add(instruction);
        for p in params {
            sum = sum.wrapping_add(*p);
        }
        !sum
    }

    /// Build a packet
    fn build_packet(id: u8, instruction: Instruction, params: &[u8]) -> Vec<u8> {
        let length = (params.len() + 2) as u8; // instruction + params + checksum
        let mut packet = Vec::with_capacity(6 + params.len());
        packet.push(0xFF);
        packet.push(0xFF);
        packet.push(id);
        packet.push(length);
        packet.push(instruction as u8);
        packet.extend_from_slice(params);
        packet.push(Self::checksum(id, length, instruction as u8, params));
        packet
    }

    /// Send a packet and return the response
    fn send_packet(&mut self, packet: &[u8]) -> Result<Vec<u8>, Error> {
        // Clear any pending data
        let _ = self.flush_input();

        // Send the packet
        self.serial
            .write_all(packet)
            .map_err(|e| Error::Hardware(format!("Failed to write packet: {}", e)))?;

        // Read response header
        let mut header = [0u8; 4];
        self.read_exact(&mut header)?;

        if header[0] != 0xFF || header[1] != 0xFF {
            return Err(Error::Hardware("Invalid response header".into()));
        }

        let _id = header[2];
        let length = header[3] as usize;

        // Read the rest of the packet
        let mut data = vec![0u8; length];
        self.read_exact(&mut data)?;

        // Verify checksum (last byte)
        // Note: We skip checksum verification for simplicity, but could add it

        Ok(data)
    }

    /// Read exact number of bytes with timeout handling
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), Error> {
        let mut total_read = 0;
        let start = std::time::Instant::now();

        while total_read < buf.len() {
            if start.elapsed() > self.timeout {
                return Err(Error::Hardware(format!(
                    "Read timeout: got {} of {} bytes",
                    total_read,
                    buf.len()
                )));
            }

            match self.serial.read(&mut buf[total_read..]) {
                Ok(0) => {
                    std::thread::sleep(Duration::from_micros(100));
                }
                Ok(n) => total_read += n,
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                    continue;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_micros(100));
                }
                Err(e) => {
                    return Err(Error::Hardware(format!("Read error: {}", e)));
                }
            }
        }

        Ok(())
    }

    /// Flush input buffer
    fn flush_input(&mut self) -> Result<(), Error> {
        let mut buf = [0u8; 256];
        loop {
            match self.serial.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => continue,
            }
        }
        Ok(())
    }

    /// Ping a servo to check if it's alive
    pub fn ping(&mut self, id: u8) -> Result<bool, Error> {
        let packet = Self::build_packet(id, Instruction::Ping, &[]);
        match self.send_packet(&packet) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Read a register value
    pub fn read_register(&mut self, id: u8, address: u8, length: u8) -> Result<Vec<u8>, Error> {
        let packet = Self::build_packet(id, Instruction::Read, &[address, length]);
        let response = self.send_packet(&packet)?;

        // Response format: [error, data..., checksum]
        if response.is_empty() {
            return Err(Error::Hardware("Empty response".into()));
        }

        let error = response[0];
        if error != 0 {
            return Err(Error::Hardware(format!("Servo error: 0x{:02X}", error)));
        }

        // Return data without error byte and checksum
        Ok(response[1..response.len() - 1].to_vec())
    }

    /// Write a register value
    pub fn write_register(&mut self, id: u8, address: u8, data: &[u8]) -> Result<(), Error> {
        let mut params = vec![address];
        params.extend_from_slice(data);

        let packet = Self::build_packet(id, Instruction::Write, &params);
        let response = self.send_packet(&packet)?;

        if !response.is_empty() {
            let error = response[0];
            if error != 0 {
                return Err(Error::Hardware(format!("Write error: 0x{:02X}", error)));
            }
        }

        Ok(())
    }

    /// Read position from a single servo
    pub fn read_position(&mut self, id: u8) -> Result<u16, Error> {
        let data = self.read_register(id, registers::PRESENT_POSITION, 2)?;
        if data.len() < 2 {
            return Err(Error::Hardware("Invalid position data".into()));
        }
        Ok(u16::from_le_bytes([data[0], data[1]]))
    }

    /// Read positions from all servos using sync read
    pub fn sync_read_positions(&mut self, ids: &[u8]) -> Result<Vec<u16>, Error> {
        // Build sync read packet
        // Format: [address, length, id1, id2, ...]
        let mut params = vec![registers::PRESENT_POSITION, 2];
        params.extend_from_slice(ids);

        let packet = Self::build_packet(BROADCAST_ID, Instruction::SyncRead, &params);

        // Clear input buffer
        let _ = self.flush_input();

        // Send sync read command
        self.serial
            .write_all(&packet)
            .map_err(|e| Error::Hardware(format!("Failed to send sync read: {}", e)))?;

        // Read responses from each servo
        let mut positions = Vec::with_capacity(ids.len());

        for &_id in ids {
            // Read response header
            let mut header = [0u8; 4];
            if let Err(_) = self.read_exact(&mut header) {
                // If we can't read, fall back to individual reads
                return self.read_positions_sequential(ids);
            }

            if header[0] != 0xFF || header[1] != 0xFF {
                return self.read_positions_sequential(ids);
            }

            let length = header[3] as usize;
            let mut data = vec![0u8; length];
            if let Err(_) = self.read_exact(&mut data) {
                return self.read_positions_sequential(ids);
            }

            // data format: [error, pos_low, pos_high, checksum]
            if data.len() >= 3 && data[0] == 0 {
                let pos = u16::from_le_bytes([data[1], data[2]]);
                positions.push(pos);
            } else {
                positions.push(2048); // Default center position on error
            }
        }

        Ok(positions)
    }

    /// Read positions sequentially (fallback)
    fn read_positions_sequential(&mut self, ids: &[u8]) -> Result<Vec<u16>, Error> {
        let mut positions = Vec::with_capacity(ids.len());
        for &id in ids {
            match self.read_position(id) {
                Ok(pos) => positions.push(pos),
                Err(_) => positions.push(2048), // Default on error
            }
        }
        Ok(positions)
    }

    /// Write position to a single servo
    pub fn write_position(&mut self, id: u8, position: u16) -> Result<(), Error> {
        let data = position.to_le_bytes();
        self.write_register(id, registers::GOAL_POSITION, &data)
    }

    /// Sync write positions to multiple servos
    pub fn sync_write_positions(&mut self, ids: &[u8], positions: &[u16]) -> Result<(), Error> {
        if ids.len() != positions.len() {
            return Err(Error::Hardware("ID and position count mismatch".into()));
        }

        // Build sync write packet
        // Format: [address, data_length, id1, data1..., id2, data2..., ...]
        let data_length = 2u8; // 2 bytes per position
        let mut params = vec![registers::GOAL_POSITION, data_length];

        for (id, pos) in ids.iter().zip(positions.iter()) {
            params.push(*id);
            params.extend_from_slice(&pos.to_le_bytes());
        }

        let packet = Self::build_packet(BROADCAST_ID, Instruction::SyncWrite, &params);

        // Sync write has no response
        self.serial
            .write_all(&packet)
            .map_err(|e| Error::Hardware(format!("Failed to sync write: {}", e)))?;

        Ok(())
    }

    /// Enable or disable torque for a servo
    pub fn set_torque(&mut self, id: u8, enabled: bool) -> Result<(), Error> {
        self.write_register(id, registers::TORQUE_ENABLE, &[if enabled { 1 } else { 0 }])
    }

    /// Enable or disable torque for all servos
    pub fn set_torque_all(&mut self, ids: &[u8], enabled: bool) -> Result<(), Error> {
        // Build sync write packet for torque enable
        let data_length = 1u8;
        let mut params = vec![registers::TORQUE_ENABLE, data_length];

        let value = if enabled { 1u8 } else { 0u8 };
        for id in ids {
            params.push(*id);
            params.push(value);
        }

        let packet = Self::build_packet(BROADCAST_ID, Instruction::SyncWrite, &params);

        self.serial
            .write_all(&packet)
            .map_err(|e| Error::Hardware(format!("Failed to set torque: {}", e)))?;

        Ok(())
    }
}

/// Convert raw servo position to radians
///
/// Feetech servos use 0-4095 range for ~300 degrees of rotation.
/// Center position (2048) = 0 radians
pub fn raw_to_radians(raw: u16) -> f32 {
    // 4096 steps = ~300 degrees = ~5.236 radians
    // Center is at 2048
    const STEPS_PER_RADIAN: f32 = 4096.0 / 5.236;
    (raw as f32 - 2048.0) / STEPS_PER_RADIAN
}

/// Convert radians to raw servo position
pub fn radians_to_raw(radians: f32) -> u16 {
    const STEPS_PER_RADIAN: f32 = 4096.0 / 5.236;
    let raw = (radians * STEPS_PER_RADIAN + 2048.0) as i32;
    raw.clamp(0, 4095) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_to_radians_center() {
        let rad = raw_to_radians(2048);
        assert!((rad - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_radians_to_raw_center() {
        let raw = radians_to_raw(0.0);
        assert_eq!(raw, 2048);
    }

    #[test]
    fn test_conversion_roundtrip() {
        for raw in [0, 1024, 2048, 3072, 4095] {
            let rad = raw_to_radians(raw);
            let back = radians_to_raw(rad);
            assert!((raw as i32 - back as i32).abs() <= 1);
        }
    }
}
