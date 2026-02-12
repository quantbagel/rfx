//! CRC32 calculation for Unitree Go2 DDS messages
//!
//! The Go2 uses CRC32 checksums to validate command and state messages.
//! This module provides the CRC computation matching the Unitree SDK2 format.

use crc32fast::Hasher;

/// Compute CRC32 checksum for the given data
///
/// Uses the standard CRC32 polynomial (IEEE 802.3) which is what
/// the Unitree SDK2 uses for message validation.
///
/// # Arguments
///
/// * `data` - The bytes to compute the checksum for
///
/// # Returns
///
/// The 32-bit CRC checksum
///
/// # Example
///
/// ```
/// use rfx_core::hardware::go2::dds::compute_crc;
///
/// let data = b"hello";
/// let crc = compute_crc(data);
/// assert_eq!(crc, 0x3610a686);
/// ```
pub fn compute_crc(data: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

/// Verify that the CRC32 checksum matches the expected value
///
/// # Arguments
///
/// * `data` - The bytes to verify (excluding the CRC field)
/// * `expected_crc` - The expected CRC32 value
///
/// # Returns
///
/// `true` if the computed CRC matches the expected value
pub fn verify_crc(data: &[u8], expected_crc: u32) -> bool {
    compute_crc(data) == expected_crc
}

/// CRC32 context for incremental computation
///
/// Useful when building messages piece by piece.
#[derive(Default)]
pub struct Crc32 {
    hasher: Hasher,
}

impl Crc32 {
    /// Create a new CRC32 context
    pub fn new() -> Self {
        Self {
            hasher: Hasher::new(),
        }
    }

    /// Update the CRC with additional data
    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    /// Update with a u8 value
    pub fn update_u8(&mut self, value: u8) {
        self.hasher.update(&[value]);
    }

    /// Update with a u16 value (little-endian)
    pub fn update_u16(&mut self, value: u16) {
        self.hasher.update(&value.to_le_bytes());
    }

    /// Update with a u32 value (little-endian)
    pub fn update_u32(&mut self, value: u32) {
        self.hasher.update(&value.to_le_bytes());
    }

    /// Update with an i32 value (little-endian)
    pub fn update_i32(&mut self, value: i32) {
        self.hasher.update(&value.to_le_bytes());
    }

    /// Update with an f32 value (little-endian)
    pub fn update_f32(&mut self, value: f32) {
        self.hasher.update(&value.to_le_bytes());
    }

    /// Finalize and return the CRC32 value
    ///
    /// Note: This consumes the context. To continue updating,
    /// create a new context.
    pub fn finalize(self) -> u32 {
        self.hasher.finalize()
    }

    /// Clone and finalize, returning the current CRC without consuming the context
    pub fn current(&self) -> u32 {
        self.hasher.clone().finalize()
    }

    /// Reset the CRC context
    pub fn reset(&mut self) {
        self.hasher = Hasher::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_crc() {
        // Test vector from IEEE 802.3
        let data = b"123456789";
        let crc = compute_crc(data);
        assert_eq!(crc, 0xcbf43926);
    }

    #[test]
    fn test_empty_crc() {
        let data = b"";
        let crc = compute_crc(data);
        assert_eq!(crc, 0x00000000);
    }

    #[test]
    fn test_verify_crc() {
        let data = b"hello world";
        let expected_crc = compute_crc(data);
        assert!(verify_crc(data, expected_crc));
        assert!(!verify_crc(data, expected_crc + 1));
    }

    #[test]
    fn test_incremental_crc() {
        let full_data = b"hello world";
        let full_crc = compute_crc(full_data);

        let mut ctx = Crc32::new();
        ctx.update(b"hello");
        ctx.update(b" ");
        ctx.update(b"world");
        let incremental_crc = ctx.finalize();

        assert_eq!(full_crc, incremental_crc);
    }

    #[test]
    fn test_crc_numeric_types() {
        let mut ctx = Crc32::new();
        ctx.update_u8(0x42);
        ctx.update_u16(0x1234);
        ctx.update_u32(0xDEADBEEF);
        ctx.update_f32(std::f32::consts::PI);
        let crc = ctx.finalize();

        // Verify deterministic
        let mut ctx2 = Crc32::new();
        ctx2.update_u8(0x42);
        ctx2.update_u16(0x1234);
        ctx2.update_u32(0xDEADBEEF);
        ctx2.update_f32(std::f32::consts::PI);
        assert_eq!(crc, ctx2.finalize());
    }

    #[test]
    fn test_current_does_not_consume() {
        let mut ctx = Crc32::new();
        ctx.update(b"hello");
        let crc1 = ctx.current();
        ctx.update(b" world");
        let crc2 = ctx.finalize();

        assert_ne!(crc1, crc2);
    }
}
