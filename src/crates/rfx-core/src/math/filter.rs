//! Digital filters for signal processing
//!
//! Common filters used in robotics for sensor data smoothing and noise reduction.

use serde::{Deserialize, Serialize};

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
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowPassFilter {
    /// Current filtered value
    value: f64,
    /// Filter coefficient (0-1). Lower = more smoothing.
    alpha: f64,
    /// Precomputed 1.0 - alpha
    one_minus_alpha: f64,
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
            value: 0.0,
            alpha,
            one_minus_alpha: 1.0 - alpha,
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
        Self {
            value: 0.0,
            alpha,
            one_minus_alpha: 1.0 - alpha,
        }
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
        self.one_minus_alpha = 1.0 - alpha;
    }
}

impl Filter for LowPassFilter {
    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        self.value = self.alpha.mul_add(value, self.one_minus_alpha * self.value);
        self.value
    }

    #[inline]
    fn reset(&mut self) {
        self.value = 0.0;
    }

    #[inline]
    fn value(&self) -> f64 {
        self.value
    }
}

impl Default for LowPassFilter {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Moving average filter backed by a fixed-size circular buffer
///
/// Simple average of the last N samples. Good for removing random noise
/// while preserving sharp edges better than exponential filters.
///
/// The const generic `N` sets the maximum window size (default 64).
/// The actual window size can be smaller via the constructor.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MovingAverageFilter<const N: usize = 64> {
    /// Running sum for efficient calculation
    sum: f64,
    /// Precomputed 1.0 / window_size
    inv_window: f64,
    /// Write index (next slot to overwrite)
    head: usize,
    /// Number of samples currently stored
    count: usize,
    /// Configured window size (<= N)
    window_size: usize,
    /// Fixed-size ring buffer
    buffer: [f64; N],
}

impl<const N: usize> MovingAverageFilter<N> {
    /// Create a new moving average filter with the given window size
    ///
    /// # Arguments
    /// * `window_size` - Number of samples to average (must be <= N)
    ///
    /// # Panics
    /// Panics if window_size is 0 or greater than N
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "Window size must be > 0");
        assert!(
            window_size <= N,
            "Window size {} exceeds buffer capacity {}",
            window_size,
            N
        );
        Self {
            sum: 0.0,
            inv_window: 1.0 / window_size as f64,
            head: 0,
            count: 0,
            window_size,
            buffer: [0.0; N],
        }
    }

    /// Get the window size
    #[inline]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Check if the filter is fully populated
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count == self.window_size
    }
}

impl<const N: usize> Filter for MovingAverageFilter<N> {
    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        // If buffer is full, subtract the oldest value being overwritten
        if self.count == self.window_size {
            self.sum -= self.buffer[self.head];
        } else {
            self.count += 1;
        }

        // Write new value at head position
        self.buffer[self.head] = value;
        self.sum += value;

        // Advance head, wrapping around the window
        self.head = (self.head + 1) % self.window_size;

        // Return average
        if self.count == self.window_size {
            self.sum * self.inv_window
        } else {
            self.sum / self.count as f64
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline]
    fn value(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
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
    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        let derivative = self
            .prev_value
            .map(|prev| (value - prev) / self.dt)
            .unwrap_or(0.0);
        self.prev_value = Some(value);
        self.lpf.update(derivative)
    }

    #[inline]
    fn reset(&mut self) {
        self.prev_value = None;
        self.lpf.reset();
    }

    #[inline]
    fn value(&self) -> f64 {
        self.lpf.value()
    }
}

/// Vector filter for 3D data (e.g., accelerometer, gyroscope)
///
/// Generic over the filter type to avoid dynamic dispatch on the hot path.
pub struct Vector3Filter<F: Filter> {
    x: F,
    y: F,
    z: F,
}

impl<F: Filter> std::fmt::Debug for Vector3Filter<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vector3Filter")
            .field("x_value", &self.x.value())
            .field("y_value", &self.y.value())
            .field("z_value", &self.z.value())
            .finish()
    }
}

impl Vector3Filter<LowPassFilter> {
    /// Create a new vector filter using low-pass filters
    pub fn low_pass(alpha: f64) -> Self {
        Self {
            x: LowPassFilter::new(alpha),
            y: LowPassFilter::new(alpha),
            z: LowPassFilter::new(alpha),
        }
    }
}

impl Vector3Filter<MovingAverageFilter> {
    /// Create a new vector filter using moving average filters
    pub fn moving_average(window_size: usize) -> Self {
        Self {
            x: MovingAverageFilter::new(window_size),
            y: MovingAverageFilter::new(window_size),
            z: MovingAverageFilter::new(window_size),
        }
    }
}

impl<F: Filter> Vector3Filter<F> {
    /// Create a vector filter from three filter instances
    pub fn from_filters(x: F, y: F, z: F) -> Self {
        Self { x, y, z }
    }

    /// Update with a 3D vector and return the filtered result
    #[inline]
    pub fn update(&mut self, v: [f64; 3]) -> [f64; 3] {
        [
            self.x.update(v[0]),
            self.y.update(v[1]),
            self.z.update(v[2]),
        ]
    }

    /// Reset all filters
    #[inline]
    pub fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
        self.z.reset();
    }

    /// Get current filtered value
    #[inline]
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
        // alpha * value + (1 - alpha) * 0.0 = 0.5 * 10.0 = 5.0
        assert_relative_eq!(lpf.update(10.0), 5.0);
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
        let mut ma = MovingAverageFilter::<64>::new(3);
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
        // alpha * value + (1 - alpha) * 0.0 = 0.5 * value
        let v = vf.update([1.0, 2.0, 3.0]);
        assert_relative_eq!(v[0], 0.5);
        assert_relative_eq!(v[1], 1.0);
        assert_relative_eq!(v[2], 1.5);
    }
}
