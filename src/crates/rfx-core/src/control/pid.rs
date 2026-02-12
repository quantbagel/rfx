//! PID Controller implementation
//!
//! A standard PID (Proportional-Integral-Derivative) controller with
//! features like integral windup protection and derivative filtering.

use serde::{Deserialize, Serialize};

/// Derivative term filtering configuration
///
/// The derivative term in a PID controller can be noisy due to
/// measurement noise and quantization. Filtering smooths the derivative
/// at the cost of phase lag.
///
/// # Example
/// ```
/// use rfx_core::control::DerivativeFilter;
///
/// // Use preset for common scenarios
/// let filter = DerivativeFilter::MODERATE;
///
/// // Or specify cutoff frequency for precise control
/// let filter = DerivativeFilter::CutoffFrequency {
///     cutoff_hz: 20.0,
///     sample_rate_hz: 500.0,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DerivativeFilter {
    /// No filtering - raw derivative passes through
    /// Use when: signal is clean, or response speed is critical
    Disabled,

    /// Low-pass filter with explicit alpha (0-1)
    /// alpha = 1.0 means no filtering, alpha = 0.0 means maximum filtering
    /// Use when: you know the exact smoothing factor needed
    LowPass {
        /// Smoothing factor (0-1). Higher = less smoothing.
        alpha: f64,
    },

    /// Low-pass filter computed from cutoff frequency
    /// Use when: you know the noise frequency or desired bandwidth
    CutoffFrequency {
        /// Cutoff frequency in Hz
        cutoff_hz: f64,
        /// Sample rate in Hz (control loop rate)
        sample_rate_hz: f64,
    },
}

impl DerivativeFilter {
    /// No filtering (alpha = 1.0)
    pub const NONE: Self = Self::Disabled;

    /// Light filtering (alpha = 0.8) - minimal lag, some smoothing
    pub const LIGHT: Self = Self::LowPass { alpha: 0.8 };

    /// Moderate filtering (alpha = 0.5) - balanced lag and smoothing
    pub const MODERATE: Self = Self::LowPass { alpha: 0.5 };

    /// Heavy filtering (alpha = 0.2) - significant smoothing, more lag
    pub const HEAVY: Self = Self::LowPass { alpha: 0.2 };

    /// Get the alpha value for this filter configuration
    pub fn alpha(&self) -> f64 {
        match self {
            Self::Disabled => 1.0,
            Self::LowPass { alpha } => alpha.clamp(0.0, 1.0),
            Self::CutoffFrequency {
                cutoff_hz,
                sample_rate_hz,
            } => {
                // Compute alpha from cutoff frequency using the formula:
                // alpha = dt / (RC + dt) where RC = 1 / (2 * pi * cutoff)
                let dt = 1.0 / sample_rate_hz;
                let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
                (dt / (rc + dt)).clamp(0.0, 1.0)
            }
        }
    }
}

impl Default for DerivativeFilter {
    fn default() -> Self {
        Self::Disabled
    }
}

/// PID controller configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PidConfig {
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Derivative gain
    pub kd: f64,
    /// Output minimum limit
    pub output_min: f64,
    /// Output maximum limit
    pub output_max: f64,
    /// Integral windup limit (f64::INFINITY for no limit)
    pub integral_limit: f64,
    /// Derivative term filtering
    pub derivative_filter: DerivativeFilter,
}

impl Default for PidConfig {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.0,
            kd: 0.0,
            output_min: f64::NEG_INFINITY,
            output_max: f64::INFINITY,
            integral_limit: f64::INFINITY,
            derivative_filter: DerivativeFilter::Disabled,
        }
    }
}

impl PidConfig {
    /// Create a new PID config with given gains
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            ..Default::default()
        }
    }

    /// Set output limits
    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.output_min = min;
        self.output_max = max;
        self
    }

    /// Set integral windup limit
    pub fn with_integral_limit(mut self, limit: f64) -> Self {
        self.integral_limit = limit;
        self
    }

    /// Set derivative filter
    ///
    /// # Example
    /// ```
    /// use rfx_core::control::{PidConfig, DerivativeFilter};
    ///
    /// let config = PidConfig::pd(1.0, 0.1)
    ///     .with_derivative_filter(DerivativeFilter::MODERATE);
    /// ```
    pub fn with_derivative_filter(mut self, filter: DerivativeFilter) -> Self {
        self.derivative_filter = filter;
        self
    }

    /// Set derivative filter from alpha value (legacy API)
    #[deprecated(note = "use with_derivative_filter(DerivativeFilter::LowPass { alpha }) instead")]
    pub fn with_derivative_filter_alpha(mut self, alpha: f64) -> Self {
        self.derivative_filter = DerivativeFilter::LowPass {
            alpha: alpha.clamp(0.0, 1.0),
        };
        self
    }

    /// Create a P-only controller
    pub fn p(kp: f64) -> Self {
        Self::new(kp, 0.0, 0.0)
    }

    /// Create a PI controller
    pub fn pi(kp: f64, ki: f64) -> Self {
        Self::new(kp, ki, 0.0)
    }

    /// Create a PD controller
    pub fn pd(kp: f64, kd: f64) -> Self {
        Self::new(kp, 0.0, kd)
    }
}

/// PID controller internal state
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PidState {
    /// Accumulated integral term
    pub integral: f64,
    /// Previous error for derivative calculation
    pub prev_error: f64,
    /// Filtered derivative value
    pub derivative_filtered: f64,
    /// Previous output
    pub prev_output: f64,
}

/// PID controller
///
/// A standard PID controller with integral windup protection and
/// derivative filtering.
///
/// # Example
/// ```
/// use rfx_core::control::{Pid, PidConfig};
///
/// let config = PidConfig::new(1.0, 0.1, 0.05)
///     .with_limits(-10.0, 10.0)
///     .with_integral_limit(5.0);
///
/// let mut pid = Pid::new(config);
///
/// // In a control loop
/// let setpoint = 1.0;
/// let measurement = 0.5;
/// let dt = 0.01; // 100Hz
///
/// let output = pid.update(setpoint, measurement, dt);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Pid {
    config: PidConfig,
    /// Cached alpha from config.derivative_filter for hot-path use
    derivative_alpha: f64,
    state: PidState,
}

impl Pid {
    /// Create a new PID controller with the given configuration
    pub fn new(config: PidConfig) -> Self {
        Self {
            derivative_alpha: config.derivative_filter.alpha(),
            config,
            state: PidState::default(),
        }
    }

    /// Create a simple P controller
    pub fn p(kp: f64) -> Self {
        Self::new(PidConfig::p(kp))
    }

    /// Create a PI controller
    pub fn pi(kp: f64, ki: f64) -> Self {
        Self::new(PidConfig::pi(kp, ki))
    }

    /// Create a PD controller
    pub fn pd(kp: f64, kd: f64) -> Self {
        Self::new(PidConfig::pd(kp, kd))
    }

    /// Create a full PID controller
    pub fn pid(kp: f64, ki: f64, kd: f64) -> Self {
        Self::new(PidConfig::new(kp, ki, kd))
    }

    /// Update the PID controller with a new measurement
    ///
    /// # Arguments
    /// * `setpoint` - Desired value
    /// * `measurement` - Current measured value
    /// * `dt` - Time step in seconds
    ///
    /// # Returns
    /// The control output
    #[inline]
    pub fn update(&mut self, setpoint: f64, measurement: f64, dt: f64) -> f64 {
        let error = setpoint - measurement;
        self.update_error(error, dt)
    }

    /// Update the PID controller with a pre-computed error
    ///
    /// # Arguments
    /// * `error` - Error (setpoint - measurement)
    /// * `dt` - Time step in seconds
    ///
    /// # Returns
    /// The control output
    #[inline]
    pub fn update_error(&mut self, error: f64, dt: f64) -> f64 {
        debug_assert!(dt > 0.0);

        // Proportional term
        let p_term = self.config.kp * error;

        // Integral term with windup protection (FMA)
        self.state.integral = error.mul_add(dt, self.state.integral);
        self.state.integral = self
            .state
            .integral
            .clamp(-self.config.integral_limit, self.config.integral_limit);
        let i_term = self.config.ki * self.state.integral;

        // Derivative term with filtering (1 FMA + 1 sub)
        let raw_derivative = (error - self.state.prev_error) / dt;
        let alpha = self.derivative_alpha;
        self.state.derivative_filtered = alpha.mul_add(
            raw_derivative - self.state.derivative_filtered,
            self.state.derivative_filtered,
        );
        let d_term = self.config.kd * self.state.derivative_filtered;

        // Calculate output
        let output =
            (p_term + i_term + d_term).clamp(self.config.output_min, self.config.output_max);

        // Update state for next iteration
        self.state.prev_error = error;
        self.state.prev_output = output;

        output
    }

    /// Reset the controller state
    pub fn reset(&mut self) {
        self.state = PidState::default();
    }

    /// Get the current state
    pub fn state(&self) -> &PidState {
        &self.state
    }

    /// Get the configuration
    pub fn config(&self) -> &PidConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: PidConfig) {
        self.derivative_alpha = config.derivative_filter.alpha();
        self.config = config;
    }

    /// Set the gains
    pub fn set_gains(&mut self, kp: f64, ki: f64, kd: f64) {
        self.config.kp = kp;
        self.config.ki = ki;
        self.config.kd = kd;
    }

    /// Get individual terms for debugging
    pub fn get_terms(&self, error: f64, dt: f64) -> (f64, f64, f64) {
        let p_term = self.config.kp * error;
        let i_term = self.config.ki * self.state.integral;
        let d_term = if dt > 0.0 {
            self.config.kd * (error - self.state.prev_error) / dt
        } else {
            0.0
        };
        (p_term, i_term, d_term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_p_controller() {
        let mut pid = Pid::p(2.0);
        let output = pid.update(10.0, 5.0, 0.01);
        // Error = 10 - 5 = 5, P term = 2 * 5 = 10
        assert_relative_eq!(output, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pi_controller() {
        let mut pid = Pid::pi(1.0, 0.5);

        // First update
        let output1 = pid.update(10.0, 5.0, 0.1);
        // Error = 5, P = 5, I = 0.5 * 5 * 0.1 = 0.25
        assert_relative_eq!(output1, 5.25, epsilon = 1e-10);

        // Second update (integral accumulates)
        let output2 = pid.update(10.0, 5.0, 0.1);
        // I = 0.5 * (0.5 + 0.5) = 0.5
        assert_relative_eq!(output2, 5.5, epsilon = 1e-10);
    }

    #[test]
    fn test_output_limits() {
        let config = PidConfig::p(10.0).with_limits(-5.0, 5.0);
        let mut pid = Pid::new(config);

        let output = pid.update(10.0, 0.0, 0.01);
        assert_relative_eq!(output, 5.0, epsilon = 1e-10); // Clamped to max
    }

    #[test]
    fn test_integral_windup() {
        let config = PidConfig::pi(1.0, 1.0).with_integral_limit(10.0);
        let mut pid = Pid::new(config);

        // Accumulate integral
        for _ in 0..100 {
            pid.update(100.0, 0.0, 0.1);
        }

        // Integral should be limited
        assert!(pid.state().integral <= 10.0);
    }

    #[test]
    fn test_reset() {
        let mut pid = Pid::pi(1.0, 1.0);
        pid.update(10.0, 5.0, 0.1);
        pid.update(10.0, 5.0, 0.1);

        assert!(pid.state().integral > 0.0);

        pid.reset();
        assert_relative_eq!(pid.state().integral, 0.0);
        assert_relative_eq!(pid.state().prev_error, 0.0);
    }
}
