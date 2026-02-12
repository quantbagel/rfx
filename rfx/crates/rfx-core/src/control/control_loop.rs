//! Fixed-rate control loop implementation
//!
//! Provides a real-time control loop that runs at a specified frequency,
//! with timing statistics and graceful shutdown support.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::{Error, Result};

/// Configuration for a control loop
#[derive(Debug, Clone)]
pub struct ControlLoopConfig {
    /// Target loop rate in Hz
    pub rate_hz: f64,
    /// Name for logging/debugging
    pub name: Arc<str>,
    /// Whether to warn on timing overruns
    pub warn_on_overrun: bool,
    /// Maximum acceptable jitter ratio before warning (e.g., 0.1 = 10%)
    pub max_jitter_ratio: f64,
}

impl Default for ControlLoopConfig {
    fn default() -> Self {
        Self {
            rate_hz: 500.0,
            name: "control_loop".into(),
            warn_on_overrun: true,
            max_jitter_ratio: 0.1,
        }
    }
}

impl ControlLoopConfig {
    /// Create a new config with the given rate
    pub fn new(rate_hz: f64) -> Self {
        Self {
            rate_hz,
            ..Default::default()
        }
    }

    /// Set the loop name
    pub fn with_name(mut self, name: impl Into<Arc<str>>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the target period
    pub fn period(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.rate_hz)
    }
}

/// Statistics for a control loop
///
/// Tracks timing information using Welford's online algorithm for
/// numerically stable variance computation.
#[derive(Debug, Clone, Copy, Default)]
pub struct ControlLoopStats {
    /// Number of loop iterations
    pub iterations: u64,
    /// Number of timing overruns
    pub overruns: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Maximum iteration time
    pub max_iteration_time: Duration,
    /// Minimum iteration time
    pub min_iteration_time: Duration,
    /// Average iteration time
    pub avg_iteration_time: Duration,
    /// Last iteration time
    pub last_iteration_time: Duration,
    // Welford's online algorithm state for variance
    welford_mean: f64,
    welford_m2: f64,
}

impl ControlLoopStats {
    pub fn update(&mut self, execution_time: Duration, target_period: Duration) {
        self.iterations += 1;
        self.total_execution_time += execution_time;
        self.last_iteration_time = execution_time;

        let time_secs = execution_time.as_secs_f64();

        if self.iterations == 1 {
            self.min_iteration_time = execution_time;
            self.max_iteration_time = execution_time;
            // Initialize Welford's algorithm
            self.welford_mean = time_secs;
            self.welford_m2 = 0.0;
        } else {
            self.min_iteration_time = self.min_iteration_time.min(execution_time);
            self.max_iteration_time = self.max_iteration_time.max(execution_time);

            // Update Welford's algorithm for online variance
            // See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
            let delta = time_secs - self.welford_mean;
            self.welford_mean += delta / self.iterations as f64;
            let delta2 = time_secs - self.welford_mean;
            self.welford_m2 += delta * delta2;
        }

        // Update avg on first iteration and every 64th to avoid per-tick division
        if self.iterations == 1 || self.iterations % 64 == 0 {
            self.avg_iteration_time = self.total_execution_time.div_f64(self.iterations as f64);
        }

        if execution_time > target_period {
            self.overruns += 1;
        }
    }

    /// Get the overrun ratio (0.0 to 1.0)
    pub fn overrun_ratio(&self) -> f64 {
        if self.iterations == 0 {
            0.0
        } else {
            self.overruns as f64 / self.iterations as f64
        }
    }

    /// Get the timing range (max - min iteration time)
    ///
    /// This is the simplest measure of timing variability but is sensitive
    /// to outliers. For a more robust measure, use [`jitter_std_dev`].
    pub fn timing_range(&self) -> Duration {
        self.max_iteration_time
            .saturating_sub(self.min_iteration_time)
    }

    /// Get timing jitter as standard deviation (in seconds)
    ///
    /// Uses Welford's online algorithm for numerically stable computation.
    /// Returns 0.0 if fewer than 2 iterations have completed.
    pub fn jitter_std_dev(&self) -> f64 {
        if self.iterations < 2 {
            0.0
        } else {
            let variance = self.welford_m2 / (self.iterations - 1) as f64;
            variance.sqrt()
        }
    }

    /// Get timing jitter as standard deviation in Duration
    pub fn jitter_std_dev_duration(&self) -> Duration {
        Duration::from_secs_f64(self.jitter_std_dev())
    }

    /// Get the coefficient of variation (CV) for timing jitter
    ///
    /// CV = std_dev / mean, providing a dimensionless measure that
    /// can be compared across different loop rates.
    ///
    /// Returns 0.0 if mean is zero or fewer than 2 iterations.
    pub fn jitter_coefficient(&self) -> f64 {
        if self.iterations < 2 || self.welford_mean == 0.0 {
            0.0
        } else {
            self.jitter_std_dev() / self.welford_mean
        }
    }

    /// Deprecated: use `timing_range()` instead
    #[deprecated(note = "use timing_range() or jitter_std_dev() instead")]
    pub fn jitter(&self) -> Duration {
        self.timing_range()
    }
}

/// Handle to a running control loop
pub struct ControlLoopHandle {
    running: Arc<AtomicBool>,
    stats: Arc<Mutex<ControlLoopStats>>,
    thread: Option<JoinHandle<Result<()>>>,
}

impl ControlLoopHandle {
    /// Check if the loop is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Get the current statistics
    pub fn stats(&self) -> ControlLoopStats {
        *self.stats.lock()
    }

    /// Stop the control loop
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Stop and wait for the loop to finish
    pub fn join(mut self) -> Result<()> {
        self.stop();
        if let Some(handle) = self.thread.take() {
            handle
                .join()
                .map_err(|_| Error::ControlLoop("Thread panicked".into()))??;
        }
        Ok(())
    }
}

/// A fixed-rate control loop
///
/// Runs a callback function at a specified frequency with timing guarantees.
///
/// # Example
/// ```ignore
/// use rfx_core::control::{ControlLoop, ControlLoopConfig};
///
/// let config = ControlLoopConfig::new(500.0).with_name("balance");
///
/// let handle = ControlLoop::spawn(config, |iteration, dt| {
///     // Control logic here
///     // Returns true to continue, false to stop
///     iteration < 1000
/// });
///
/// // Later...
/// handle.stop();
/// handle.join().unwrap();
/// ```
pub struct ControlLoop;

impl ControlLoop {
    /// Spawn a control loop in a new thread
    ///
    /// The callback receives the iteration count and actual delta time,
    /// and returns true to continue or false to stop.
    pub fn spawn<F>(config: ControlLoopConfig, mut callback: F) -> ControlLoopHandle
    where
        F: FnMut(u64, f64) -> bool + Send + 'static,
    {
        let running = Arc::new(AtomicBool::new(true));
        let stats = Arc::new(Mutex::new(ControlLoopStats::default()));

        let running_clone = running.clone();
        let stats_clone = stats.clone();
        let period = config.period();

        let thread = thread::spawn(move || {
            let mut iteration = 0u64;
            let mut last_time = Instant::now();

            while running_clone.load(Ordering::Relaxed) {
                let loop_start = Instant::now();
                let dt = loop_start.duration_since(last_time).as_secs_f64();
                last_time = loop_start;

                // Run the callback
                let should_continue = callback(iteration, dt);

                let execution_time = loop_start.elapsed();

                if !should_continue {
                    running_clone.store(false, Ordering::Relaxed);
                    break;
                }

                // Only update stats for successful iterations
                stats_clone.lock().update(execution_time, period);

                // Sleep for remaining time
                if let Some(sleep_time) = period.checked_sub(execution_time) {
                    thread::sleep(sleep_time);
                } else if config.warn_on_overrun {
                    tracing::warn!(
                        "{}: loop overrun by {:?}",
                        config.name,
                        execution_time - period
                    );
                }

                iteration += 1;
            }

            Ok(())
        });

        ControlLoopHandle {
            running,
            stats,
            thread: Some(thread),
        }
    }

    /// Run a control loop on the current thread (blocking)
    pub fn run<F>(config: ControlLoopConfig, mut callback: F) -> Result<ControlLoopStats>
    where
        F: FnMut(u64, f64) -> bool,
    {
        let period = config.period();
        let mut stats = ControlLoopStats::default();
        let mut iteration = 0u64;
        let mut last_time = Instant::now();

        loop {
            let loop_start = Instant::now();
            let dt = loop_start.duration_since(last_time).as_secs_f64();
            last_time = loop_start;

            // Run the callback
            let should_continue = callback(iteration, dt);

            let execution_time = loop_start.elapsed();

            if !should_continue {
                break;
            }

            // Only count successful iterations
            stats.update(execution_time, period);

            // Sleep for remaining time
            if let Some(sleep_time) = period.checked_sub(execution_time) {
                thread::sleep(sleep_time);
            } else if config.warn_on_overrun {
                tracing::warn!(
                    "{}: loop overrun by {:?}",
                    config.name,
                    execution_time - period
                );
            }

            iteration += 1;
        }

        Ok(stats)
    }

    /// Run a control loop with a timeout
    pub fn run_for<F>(
        config: ControlLoopConfig,
        duration: Duration,
        mut callback: F,
    ) -> Result<ControlLoopStats>
    where
        F: FnMut(u64, f64) -> bool,
    {
        let start = Instant::now();
        Self::run(config, |iter, dt| {
            if start.elapsed() >= duration {
                return false;
            }
            callback(iter, dt)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_loop_iterations() {
        let config = ControlLoopConfig::new(1000.0).with_name("test");
        let stats = ControlLoop::run(config, |iter, _dt| iter < 10).unwrap();
        assert_eq!(stats.iterations, 10);
    }

    #[test]
    fn test_control_loop_timing() {
        let config = ControlLoopConfig::new(100.0); // 10ms period
        let start = Instant::now();
        let stats = ControlLoop::run(config, |iter, _dt| iter < 5).unwrap();
        let elapsed = start.elapsed();

        // Should take approximately 50ms (5 iterations at 10ms each)
        // Allow some tolerance
        assert!(elapsed >= Duration::from_millis(40));
        assert!(elapsed <= Duration::from_millis(100));
        assert_eq!(stats.iterations, 5);
    }

    #[test]
    fn test_control_loop_spawn_and_stop() {
        let config = ControlLoopConfig::new(100.0);
        let handle = ControlLoop::spawn(config, |_iter, _dt| true);

        assert!(handle.is_running());

        // Let it run for a bit
        thread::sleep(Duration::from_millis(50));

        handle.stop();
        let stats = handle.stats();

        assert!(stats.iterations > 0);
    }

    #[test]
    fn test_control_loop_run_for() {
        let config = ControlLoopConfig::new(100.0);
        let stats =
            ControlLoop::run_for(config, Duration::from_millis(100), |_iter, _dt| true).unwrap();

        // Should have run approximately 10 iterations (wider bounds for CI tolerance)
        assert!(
            stats.iterations >= 5 && stats.iterations <= 20,
            "Expected ~10 iterations, got {}",
            stats.iterations
        );
    }
}
