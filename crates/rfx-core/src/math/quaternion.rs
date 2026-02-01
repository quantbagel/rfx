//! Quaternion implementation for 3D rotations
//!
//! Wrapper around nalgebra's UnitQuaternion with robotics-friendly APIs.

use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// A unit quaternion representing a 3D rotation
///
/// Uses Hamilton convention (w, x, y, z) where w is the scalar part.
/// Always normalized to unit length.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    /// Scalar component (w)
    pub w: f64,
    /// X component
    pub x: f64,
    /// Y component
    pub y: f64,
    /// Z component
    pub z: f64,
}

impl Quaternion {
    /// Create a new quaternion from components (automatically normalized)
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        let mut q = Self { w, x, y, z };
        q.normalize();
        q
    }

    /// Identity quaternion (no rotation)
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create from Euler angles (roll, pitch, yaw) in radians
    ///
    /// Uses ZYX convention (yaw around Z, then pitch around Y, then roll around X)
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        let uq = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        Self::from_nalgebra(uq)
    }

    /// Create from axis-angle representation
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        let axis_vec = Vector3::new(axis[0], axis[1], axis[2]);
        if let Some(unit_axis) = nalgebra::Unit::try_new(axis_vec, 1e-10) {
            let uq = UnitQuaternion::from_axis_angle(&unit_axis, angle);
            Self::from_nalgebra(uq)
        } else {
            Self::identity()
        }
    }

    /// Create from a rotation matrix
    pub fn from_rotation_matrix(matrix: &super::Matrix3) -> Self {
        let rot = nalgebra::Rotation3::from_matrix_unchecked(*matrix);
        let uq = UnitQuaternion::from_rotation_matrix(&rot);
        Self::from_nalgebra(uq)
    }

    /// Convert from nalgebra's UnitQuaternion
    pub fn from_nalgebra(uq: UnitQuaternion<f64>) -> Self {
        Self {
            w: uq.w,
            x: uq.i,
            y: uq.j,
            z: uq.k,
        }
    }

    /// Convert to nalgebra's UnitQuaternion
    pub fn to_nalgebra(&self) -> UnitQuaternion<f64> {
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(self.w, self.x, self.y, self.z))
    }

    /// Get Euler angles (roll, pitch, yaw) in radians
    pub fn to_euler(&self) -> (f64, f64, f64) {
        self.to_nalgebra().euler_angles()
    }

    /// Get roll angle in radians
    pub fn roll(&self) -> f64 {
        self.to_euler().0
    }

    /// Get pitch angle in radians
    pub fn pitch(&self) -> f64 {
        self.to_euler().1
    }

    /// Get yaw angle in radians
    pub fn yaw(&self) -> f64 {
        self.to_euler().2
    }

    /// Get axis-angle representation
    pub fn to_axis_angle(&self) -> ([f64; 3], f64) {
        let uq = self.to_nalgebra();
        if let Some((axis, angle)) = uq.axis_angle() {
            ([axis[0], axis[1], axis[2]], angle)
        } else {
            ([0.0, 0.0, 1.0], 0.0)
        }
    }

    /// Get the rotation matrix
    pub fn to_rotation_matrix(&self) -> super::Matrix3 {
        *self.to_nalgebra().to_rotation_matrix().matrix()
    }

    /// Multiply two quaternions (compose rotations)
    pub fn multiply(&self, other: &Quaternion) -> Quaternion {
        let uq = self.to_nalgebra() * other.to_nalgebra();
        Self::from_nalgebra(uq)
    }

    /// Get the inverse (conjugate) quaternion
    pub fn inverse(&self) -> Quaternion {
        Self::from_nalgebra(self.to_nalgebra().inverse())
    }

    /// Rotate a 3D vector
    pub fn rotate_vector(&self, v: [f64; 3]) -> [f64; 3] {
        let vec = Vector3::new(v[0], v[1], v[2]);
        let rotated = self.to_nalgebra() * vec;
        [rotated[0], rotated[1], rotated[2]]
    }

    /// Spherical linear interpolation between two quaternions
    pub fn slerp(&self, other: &Quaternion, t: f64) -> Quaternion {
        let uq = self.to_nalgebra().slerp(&other.to_nalgebra(), t);
        Self::from_nalgebra(uq)
    }

    /// Normalize the quaternion to unit length
    fn normalize(&mut self) {
        let norm = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if norm > 1e-10 {
            self.w /= norm;
            self.x /= norm;
            self.y /= norm;
            self.z /= norm;
        }
    }

    /// Squared magnitude
    pub fn norm_squared(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Dot product between two quaternions
    pub fn dot(&self, other: &Quaternion) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Angular distance between two quaternions in radians
    pub fn angular_distance(&self, other: &Quaternion) -> f64 {
        let dot = self.dot(other).abs().min(1.0);
        2.0 * dot.acos()
    }
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::identity()
    }
}

impl std::ops::Mul for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(&rhs)
    }
}

impl std::ops::Mul<&Quaternion> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: &Self) -> Self::Output {
        self.multiply(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_identity() {
        let q = Quaternion::identity();
        assert_relative_eq!(q.w, 1.0);
        assert_relative_eq!(q.x, 0.0);
        assert_relative_eq!(q.y, 0.0);
        assert_relative_eq!(q.z, 0.0);
    }

    #[test]
    fn test_from_euler() {
        let q = Quaternion::from_euler(0.0, 0.0, 0.0);
        let identity = Quaternion::identity();
        assert_relative_eq!(q.dot(&identity).abs(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euler_roundtrip() {
        let (roll, pitch, yaw) = (0.1, 0.2, 0.3);
        let q = Quaternion::from_euler(roll, pitch, yaw);
        let (r2, p2, y2) = q.to_euler();
        assert_relative_eq!(roll, r2, epsilon = 1e-10);
        assert_relative_eq!(pitch, p2, epsilon = 1e-10);
        assert_relative_eq!(yaw, y2, epsilon = 1e-10);
    }

    #[test]
    fn test_rotate_vector() {
        // 90 degree rotation around Z axis
        let q = Quaternion::from_euler(0.0, 0.0, PI / 2.0);
        let v = [1.0, 0.0, 0.0];
        let rotated = q.rotate_vector(v);
        assert_relative_eq!(rotated[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse() {
        let q = Quaternion::from_euler(0.1, 0.2, 0.3);
        let q_inv = q.inverse();
        let result = q * q_inv;
        let identity = Quaternion::identity();
        assert_relative_eq!(result.dot(&identity).abs(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_slerp() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::from_euler(0.0, 0.0, PI);
        let q_mid = q1.slerp(&q2, 0.5);
        let (_, _, yaw) = q_mid.to_euler();
        assert_relative_eq!(yaw.abs(), PI / 2.0, epsilon = 1e-10);
    }
}
