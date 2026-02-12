//! Math utilities for robotics: transforms, quaternions, and filters
//!
//! This module provides essential mathematical primitives for robotics applications,
//! built on top of nalgebra for performance and correctness.

mod filter;
mod quaternion;
mod transform;

pub use filter::{Filter, LowPassFilter, MovingAverageFilter};
pub use quaternion::Quaternion;
pub use transform::Transform;

/// Type alias for 3D vectors
pub type Vector3 = nalgebra::Vector3<f64>;

/// Type alias for 4x4 matrices
pub type Matrix4 = nalgebra::Matrix4<f64>;

/// Type alias for 3x3 rotation matrices
pub type Matrix3 = nalgebra::Matrix3<f64>;

/// Linearly interpolate between two values
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Normalize an angle to [-π, π]
#[inline]
pub fn normalize_angle(angle: f64) -> f64 {
    const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
    const INV_TWO_PI: f64 = 1.0 / TWO_PI;
    angle - TWO_PI * (angle * INV_TWO_PI + 0.5).floor()
}

/// Wrap angle difference to [-π, π] for shortest path
#[inline]
pub fn angle_diff(target: f64, current: f64) -> f64 {
    normalize_angle(target - current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_clamp() {
        assert_eq!(5.0f64.clamp(0.0, 10.0), 5.0);
        assert_eq!((-1.0f64).clamp(0.0, 10.0), 0.0);
        assert_eq!(15.0f64.clamp(0.0, 10.0), 10.0);
    }

    #[test]
    fn test_lerp() {
        assert_relative_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_relative_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_relative_eq!(lerp(0.0, 10.0, 1.0), 10.0);
    }

    #[test]
    fn test_normalize_angle() {
        use std::f64::consts::PI;
        assert_relative_eq!(normalize_angle(0.0), 0.0, epsilon = 1e-10);
        // PI and -PI are equivalent; the branchless formula maps PI -> -PI
        assert_relative_eq!(normalize_angle(PI).abs(), PI, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(-PI).abs(), PI, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(3.0 * PI).abs(), PI, epsilon = 1e-10);
    }
}
