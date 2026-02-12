//! SE(3) Transform for 3D rigid body transformations
//!
//! Represents position and orientation in 3D space, commonly used for
//! robot link frames, sensor poses, and coordinate transformations.

use super::{Matrix4, Quaternion};
use serde::{Deserialize, Serialize};

/// A rigid body transformation in 3D space (SE3)
///
/// Represents both position (translation) and orientation (rotation).
/// Can be composed with other transforms and used to transform points/vectors.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    /// Position (translation) in 3D space
    pub position: [f64; 3],
    /// Orientation as a unit quaternion
    pub orientation: Quaternion,
}

impl Transform {
    /// Create a new transform from position and orientation
    #[inline]
    pub const fn new(position: [f64; 3], orientation: Quaternion) -> Self {
        Self {
            position,
            orientation,
        }
    }

    /// Identity transform (no translation, no rotation)
    #[inline]
    pub const fn identity() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            orientation: Quaternion::identity(),
        }
    }

    /// Create from position only (identity rotation)
    #[inline]
    pub const fn from_position(x: f64, y: f64, z: f64) -> Self {
        Self {
            position: [x, y, z],
            orientation: Quaternion::identity(),
        }
    }

    /// Create from orientation only (zero position)
    #[inline]
    pub const fn from_orientation(orientation: Quaternion) -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            orientation,
        }
    }

    /// Create from Euler angles (roll, pitch, yaw) and position
    pub fn from_euler(position: [f64; 3], roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            position,
            orientation: Quaternion::from_euler(roll, pitch, yaw),
        }
    }

    /// Create from a 4x4 homogeneous transformation matrix
    pub fn from_matrix(matrix: &Matrix4) -> Self {
        let position = [matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]];
        let rotation_matrix = matrix.fixed_view::<3, 3>(0, 0).into_owned();
        let orientation = Quaternion::from_rotation_matrix(&rotation_matrix);
        Self {
            position,
            orientation,
        }
    }

    /// Convert to a 4x4 homogeneous transformation matrix
    pub fn to_matrix(&self) -> Matrix4 {
        let rot = self.orientation.to_rotation_matrix();
        Matrix4::new(
            rot[(0, 0)],
            rot[(0, 1)],
            rot[(0, 2)],
            self.position[0],
            rot[(1, 0)],
            rot[(1, 1)],
            rot[(1, 2)],
            self.position[1],
            rot[(2, 0)],
            rot[(2, 1)],
            rot[(2, 2)],
            self.position[2],
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Compose two transforms: self * other
    ///
    /// The result represents applying `other` first, then `self`.
    #[inline]
    pub fn compose(&self, other: &Transform) -> Transform {
        let new_orientation = self.orientation.multiply(&other.orientation);
        let rotated_pos = self.orientation.rotate_vector(other.position);
        let new_position = [
            self.position[0] + rotated_pos[0],
            self.position[1] + rotated_pos[1],
            self.position[2] + rotated_pos[2],
        ];
        Transform::new(new_position, new_orientation)
    }

    /// Get the inverse transform
    #[inline]
    pub fn inverse(&self) -> Transform {
        let inv_orientation = self.orientation.inverse();
        let rotated_pos = inv_orientation.rotate_vector(self.position);
        Transform::new(
            [-rotated_pos[0], -rotated_pos[1], -rotated_pos[2]],
            inv_orientation,
        )
    }

    /// Transform a 3D point
    #[inline]
    pub fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        let rotated = self.orientation.rotate_vector(point);
        [
            rotated[0] + self.position[0],
            rotated[1] + self.position[1],
            rotated[2] + self.position[2],
        ]
    }

    /// Transform a 3D vector (rotation only, no translation)
    #[inline]
    pub fn transform_vector(&self, vector: [f64; 3]) -> [f64; 3] {
        self.orientation.rotate_vector(vector)
    }

    /// Linear interpolation between two transforms
    pub fn lerp(&self, other: &Transform, t: f64) -> Transform {
        let position = [
            (other.position[0] - self.position[0]).mul_add(t, self.position[0]),
            (other.position[1] - self.position[1]).mul_add(t, self.position[1]),
            (other.position[2] - self.position[2]).mul_add(t, self.position[2]),
        ];
        let orientation = self.orientation.slerp(&other.orientation, t);
        Transform::new(position, orientation)
    }

    /// Squared distance between two transforms (translation only)
    #[inline]
    pub fn translation_distance_sq(&self, other: &Transform) -> f64 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        dx.mul_add(dx, dy.mul_add(dy, dz * dz))
    }

    /// Distance between two transforms (translation only)
    #[inline]
    pub fn translation_distance(&self, other: &Transform) -> f64 {
        self.translation_distance_sq(other).sqrt()
    }

    /// Angular distance between two transforms (rotation only)
    #[inline]
    pub fn angular_distance(&self, other: &Transform) -> f64 {
        self.orientation.angular_distance(&other.orientation)
    }

    /// Get x position
    #[inline]
    pub fn x(&self) -> f64 {
        self.position[0]
    }

    /// Get y position
    #[inline]
    pub fn y(&self) -> f64 {
        self.position[1]
    }

    /// Get z position
    #[inline]
    pub fn z(&self) -> f64 {
        self.position[2]
    }

    /// Get roll angle
    #[inline]
    pub fn roll(&self) -> f64 {
        self.orientation.roll()
    }

    /// Get pitch angle
    #[inline]
    pub fn pitch(&self) -> f64 {
        self.orientation.pitch()
    }

    /// Get yaw angle
    #[inline]
    pub fn yaw(&self) -> f64 {
        self.orientation.yaw()
    }

    /// Set position
    #[inline]
    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.position = [x, y, z];
    }

    /// Set orientation from Euler angles
    #[inline]
    pub fn set_euler(&mut self, roll: f64, pitch: f64, yaw: f64) {
        self.orientation = Quaternion::from_euler(roll, pitch, yaw);
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

impl std::ops::Mul for Transform {
    type Output = Transform;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&Transform> for Transform {
    type Output = Transform;

    fn mul(self, rhs: &Self) -> Self::Output {
        self.compose(rhs)
    }
}

/// Named frame for coordinate system tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    /// Name of this frame
    pub name: String,
    /// Transform from parent frame to this frame
    pub transform: Transform,
    /// Parent frame name (None for world/root frame)
    pub parent: Option<String>,
}

impl Frame {
    /// Create a new frame
    pub fn new(name: impl Into<String>, transform: Transform, parent: Option<String>) -> Self {
        Self {
            name: name.into(),
            transform,
            parent,
        }
    }

    /// Create a root/world frame
    pub fn root(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transform: Transform::identity(),
            parent: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_identity() {
        let t = Transform::identity();
        assert_eq!(t.position, [0.0, 0.0, 0.0]);
        assert_relative_eq!(t.orientation.w, 1.0);
    }

    #[test]
    fn test_transform_point() {
        let t = Transform::from_position(1.0, 2.0, 3.0);
        let p = t.transform_point([0.0, 0.0, 0.0]);
        assert_eq!(p, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_compose() {
        let t1 = Transform::from_position(1.0, 0.0, 0.0);
        let t2 = Transform::from_position(0.0, 1.0, 0.0);
        let composed = t1.compose(&t2);
        assert_relative_eq!(composed.position[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(composed.position[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse() {
        let t = Transform::from_euler([1.0, 2.0, 3.0], 0.1, 0.2, 0.3);
        let t_inv = t.inverse();
        let identity = t.compose(&t_inv);
        assert_relative_eq!(identity.position[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity.position[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity.position[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_transform() {
        let t = Transform::from_euler([0.0, 0.0, 0.0], 0.0, 0.0, PI / 2.0);
        let p = t.transform_point([1.0, 0.0, 0.0]);
        assert_relative_eq!(p[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(p[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(p[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_roundtrip() {
        let t = Transform::from_euler([1.0, 2.0, 3.0], 0.1, 0.2, 0.3);
        let matrix = t.to_matrix();
        let t2 = Transform::from_matrix(&matrix);
        assert_relative_eq!(t.position[0], t2.position[0], epsilon = 1e-10);
        assert_relative_eq!(t.position[1], t2.position[1], epsilon = 1e-10);
        assert_relative_eq!(t.position[2], t2.position[2], epsilon = 1e-10);
        assert_relative_eq!(
            t.orientation.dot(&t2.orientation).abs(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_lerp() {
        let t1 = Transform::from_position(0.0, 0.0, 0.0);
        let t2 = Transform::from_position(10.0, 10.0, 10.0);
        let t_mid = t1.lerp(&t2, 0.5);
        assert_relative_eq!(t_mid.position[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(t_mid.position[1], 5.0, epsilon = 1e-10);
        assert_relative_eq!(t_mid.position[2], 5.0, epsilon = 1e-10);
    }
}
