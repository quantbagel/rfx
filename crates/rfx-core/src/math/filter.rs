//! Digital filters for signal processing
//!
//! Common filters used in robotics for sensor data smoothing and noise reduction.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Trait for digital filters
pub trait Filter: Send + Sync {
    /// Update the filter with a new value and return the filtered output
    fn update(&mut self, value: f64) -> f64;

    /// Reset the filter state
    fn reset(&mut self);

    /// Get the current filtered value without updating
    fn value(&self) -> f64;
}

/// First-order low-pass filter (exponential moving average)
///
/// Smooths high-frequency noise while allowing low-frequency signals through.
/// Cutoff frequency is determined by the alpha parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowPassFilter {
    /// Filter coefficient (0-1). Lower = more smoothing.
    alpha: f64,
    /// Current filtered value
    value: f64,
    /// Whether the filter has been initialized
    initialized: bool,
}

impl LowPassFilter {
    /// Create a new low-pass filter with the given alpha coefficient
    ///
    /// # Arguments
    /// * `alpha` - Filter coefficient (0.0 to 1.0). Lower values = more smoothing.
    ///
    /// # Panics
    /// Panics if alpha is not in range [0, 1]
    pub fn new(alpha: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "Alpha must be between 0 and 1"
        );
        Self {
            alpha,
            value: 0.0,
            initialized: false,
        }
    }

    /// Create a low-pass filter from cutoff frequency and sample rate
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    /// * `sample_rate_hz` - Sample rate in Hz
    pub fn from_cutoff(cutoff_hz: f64, sample_rate_hz: f64) -> Self {
        let dt = 1.0 / sample_rate_hz;
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
        let alpha = dt / (rc + dt);
        Self::new(alpha)
    }

    /// Get the alpha coefficient
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set the alpha coefficient
    pub fn set_alpha(&mut self, alpha: f64) {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "Alpha must be between 0 and 1"
        );
        self.alpha = alpha;
    }
}

impl Filter for LowPassFilter {
    fn update(&mut self, value: f64) -> f64 {
        if !self.initialized {
            self.value = value;
            self.initialized = true;
        } else {
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value;
        }
        self.value
    }

    fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
    }

    fn value(&self) -> f64 {
        self.value
    }
}

impl Default for LowPassFilter {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Moving average filter
///
/// Simple average of the last N samples. Good for removing random noise
/// while preserving sharp edges better than exponential filters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovingAverageFilter {
    /// Window size
    window_size: usize,
    /// Sample buffer
    buffer: VecDeque<f64>,
    /// Running sum for efficient calculation
    sum: f64,
}

impl MovingAverageFilter {
    /// Create a new moving average filter with the given window size
    ///
    /// # Arguments
    /// * `window_size` - Number of samples to average
    ///
    /// # Panics
    /// Panics if window_size is 0
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "Window size must be > 0");
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
            sum: 0.0,
        }
    }

    /// Get the window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Check if the filter is fully populated
    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.window_size
    }
}

impl Filter for MovingAverageFilter {
    fn update(&mut self, value: f64) -> f64 {
        // Add new value
        self.buffer.push_back(value);
        self.sum += value;

        // Remove oldest value if buffer is full
        if self.buffer.len() > self.window_size {
            if let Some(old) = self.buffer.pop_front() {
                self.sum -= old;
            }
        }

        // Return average
        self.sum / self.buffer.len() as f64
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
    }

    fn value(&self) -> f64 {
        if self.buffer.is_empty() {
            0.0
        } else {
            self.sum / self.buffer.len() as f64
        }
    }
}

impl Default for MovingAverageFilter {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Derivative filter for computing rate of change
///
/// Estimates the derivative of a signal using finite differences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivativeFilter {
    /// Previous value
    prev_value: Option<f64>,
    /// Sample period in seconds
    dt: f64,
    /// Low-pass filter for derivative smoothing
    lpf: LowPassFilter,
}

impl DerivativeFilter {
    /// Create a new derivative filter
    ///
    /// # Arguments
    /// * `sample_rate_hz` - Sample rate in Hz
    /// * `smoothing_alpha` - Low-pass filter coefficient for derivative smoothing
    pub fn new(sample_rate_hz: f64, smoothing_alpha: f64) -> Self {
        Self {
            prev_value: None,
            dt: 1.0 / sample_rate_hz,
            lpf: LowPassFilter::new(smoothing_alpha),
        }
    }

    /// Get the sample period
    pub fn dt(&self) -> f64 {
        self.dt
    }
}

impl Filter for DerivativeFilter {
    fn update(&mut self, value: f64) -> f64 {
        let derivative = match self.prev_value {
            Some(prev) => (value - prev) / self.dt,
            None => 0.0,
        };
        self.prev_value = Some(value);
        self.lpf.update(derivative)
    }

    fn reset(&mut self) {
        self.prev_value = None;
        self.lpf.reset();
    }

    fn value(&self) -> f64 {
        self.lpf.value()
    }
}

/// Vector filter for 3D data (e.g., accelerometer, gyroscope)
pub struct Vector3Filter {
    x: Box<dyn Filter>,
    y: Box<dyn Filter>,
    z: Box<dyn Filter>,
}

impl std::fmt::Debug for Vector3Filter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vector3Filter")
            .field("x_value", &self.x.value())
            .field("y_value", &self.y.value())
            .field("z_value", &self.z.value())
            .finish()
    }
}

impl Vector3Filter {
    /// Create a new vector filter using low-pass filters
    pub fn low_pass(alpha: f64) -> Self {
        Self {
            x: Box::new(LowPassFilter::new(alpha)),
            y: Box::new(LowPassFilter::new(alpha)),
            z: Box::new(LowPassFilter::new(alpha)),
        }
    }

    /// Create a new vector filter using moving average filters
    pub fn moving_average(window_size: usize) -> Self {
        Self {
            x: Box::new(MovingAverageFilter::new(window_size)),
            y: Box::new(MovingAverageFilter::new(window_size)),
            z: Box::new(MovingAverageFilter::new(window_size)),
        }
    }

    /// Update with a 3D vector and return the filtered result
    pub fn update(&mut self, v: [f64; 3]) -> [f64; 3] {
        [
            self.x.update(v[0]),
            self.y.update(v[1]),
            self.z.update(v[2]),
        ]
    }

    /// Reset all filters
    pub fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
        self.z.reset();
    }

    /// Get current filtered value
    pub fn value(&self) -> [f64; 3] {
        [self.x.value(), self.y.value(), self.z.value()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_low_pass_initialization() {
        let mut lpf = LowPassFilter::new(0.5);
        assert_relative_eq!(lpf.update(10.0), 10.0); // First value passes through
    }

    #[test]
    fn test_low_pass_smoothing() {
        let mut lpf = LowPassFilter::new(0.1);
        lpf.update(0.0);
        let val = lpf.update(10.0);
        // Should be smoothed towards 10 but not there yet
        assert!(val > 0.0 && val < 10.0);
    }

    #[test]
    fn test_low_pass_from_cutoff() {
        let lpf = LowPassFilter::from_cutoff(10.0, 100.0);
        assert!(lpf.alpha() > 0.0 && lpf.alpha() < 1.0);
    }

    #[test]
    fn test_moving_average() {
        let mut ma = MovingAverageFilter::new(3);
        assert_relative_eq!(ma.update(1.0), 1.0);
        assert_relative_eq!(ma.update(2.0), 1.5);
        assert_relative_eq!(ma.update(3.0), 2.0);
        assert_relative_eq!(ma.update(4.0), 3.0); // (2+3+4)/3
    }

    #[test]
    fn test_derivative() {
        let mut df = DerivativeFilter::new(100.0, 1.0); // No smoothing
        df.update(0.0);
        let deriv = df.update(1.0);
        // 1.0 / 0.01 = 100.0
        assert_relative_eq!(deriv, 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_filter_reset() {
        let mut lpf = LowPassFilter::new(0.5);
        lpf.update(10.0);
        lpf.update(10.0);
        lpf.reset();
        assert_relative_eq!(lpf.value(), 0.0);
    }

    #[test]
    fn test_vector_filter() {
        let mut vf = Vector3Filter::low_pass(0.5);
        let v = vf.update([1.0, 2.0, 3.0]);
        assert_relative_eq!(v[0], 1.0);
        assert_relative_eq!(v[1], 2.0);
        assert_relative_eq!(v[2], 3.0);
    }
}
