//! SO-101 6-DOF robotic arm driver
//!
//! This module provides a Rust driver for the SO-101 robotic arm,
//! which uses Feetech STS3215 servos.
//!
//! # Example
//!
//! ```no_run
//! use rfx_core::hardware::so101::{So101, So101Config};
//!
//! let config = So101Config::follower("/dev/ttyACM0");
//! let mut arm = So101::connect(config)?;
//!
//! // Read current state
//! let state = arm.state();
//! println!("Positions: {:?}", state.joint_positions());
//!
//! // Move to a position
//! arm.set_positions(&[0.0; 6])?;
//!
//! // Disconnect
//! arm.disconnect()?;
//! # Ok::<(), rfx_core::Error>(())
//! ```

mod config;
mod serial;
mod types;

pub use config::So101Config;
pub use types::{So101State, JOINT_NAMES, NUM_JOINTS};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serialport::SerialPort;

use arrayvec::ArrayVec;

use crate::Error;
use serial::{radians_to_raw, raw_to_radians, FeetechProtocol, SERVO_IDS};
use types::TimingState;

/// Read loop interval (50 Hz)
const READ_INTERVAL: Duration = Duration::from_millis(20);

/// SO-101 robotic arm interface
pub struct So101 {
    /// Thread-safe state
    state: Arc<RwLock<So101State>>,
    /// Protocol handler (protected by mutex for exclusive access)
    protocol: Arc<parking_lot::Mutex<FeetechProtocol<Box<dyn SerialPort>>>>,
    /// Configuration
    config: So101Config,
    /// Flag to stop the reader thread
    running: Arc<AtomicBool>,
    /// Reader thread handle
    reader_handle: Option<JoinHandle<()>>,
}

impl So101 {
    /// Connect to an SO-101 arm
    pub fn connect(config: So101Config) -> Result<Self, Error> {
        tracing::info!(
            "Connecting to SO-101 on {} at {} baud",
            config.port,
            config.baudrate
        );

        // Open serial port
        let port = serialport::new(&config.port, config.baudrate)
            .timeout(Duration::from_millis(100))
            .open()
            .map_err(|e| Error::Hardware(format!("Failed to open port {}: {}", config.port, e)))?;

        let protocol = FeetechProtocol::new(port, Duration::from_millis(100));
        let protocol = Arc::new(parking_lot::Mutex::new(protocol));

        // Initialize state
        let state = Arc::new(RwLock::new(So101State {
            connected: true,
            ..Default::default()
        }));

        // Set up reader thread
        let running = Arc::new(AtomicBool::new(true));

        let reader_handle = {
            let state = Arc::clone(&state);
            let protocol = Arc::clone(&protocol);
            let running = Arc::clone(&running);

            thread::Builder::new()
                .name("so101-reader".into())
                .spawn(move || {
                    Self::reader_loop(state, protocol, running);
                })
                .map_err(|e| Error::Hardware(format!("Failed to spawn reader thread: {}", e)))?
        };

        let arm = Self {
            state,
            protocol,
            config,
            running,
            reader_handle: Some(reader_handle),
        };

        // Initial ping to verify connection
        {
            let mut proto = arm.protocol.lock();
            for &id in &SERVO_IDS {
                if !proto.ping(id).unwrap_or(false) {
                    tracing::warn!("Servo {} did not respond to ping", id);
                }
            }
        }

        tracing::info!("SO-101 connected successfully");
        Ok(arm)
    }

    /// Reader thread loop
    fn reader_loop(
        state: Arc<RwLock<So101State>>,
        protocol: Arc<parking_lot::Mutex<FeetechProtocol<Box<dyn SerialPort>>>>,
        running: Arc<AtomicBool>,
    ) {
        let mut timing = TimingState::new();
        let start_time = Instant::now();

        while running.load(Ordering::Relaxed) {
            let loop_start = Instant::now();

            // Read positions
            let positions = {
                let mut proto = protocol.lock();
                proto.sync_read_positions(&SERVO_IDS)
            };

            if let Ok(raw_positions) = positions {
                let now = Instant::now();
                let timestamp = start_time.elapsed().as_secs_f64();

                // Convert to radians
                let mut joint_positions = [0.0f32; NUM_JOINTS];
                for (i, &raw) in raw_positions.iter().enumerate() {
                    if i < NUM_JOINTS {
                        joint_positions[i] = raw_to_radians(raw);
                    }
                }

                // Calculate velocities
                let mut joint_velocities = [0.0f32; NUM_JOINTS];
                if let Some(last_read) = timing.last_read {
                    let dt = now.duration_since(last_read).as_secs_f32();
                    if dt > 0.0 {
                        for i in 0..NUM_JOINTS {
                            joint_velocities[i] =
                                (joint_positions[i] - timing.last_positions[i]) / dt;
                        }
                    }
                }

                // Update timing state
                timing.last_positions = joint_positions;
                timing.last_read = Some(now);

                // Update shared state
                let mut state = state.write();
                state.joint_positions = joint_positions;
                state.joint_velocities = joint_velocities;
                state.timestamp = timestamp;
            }

            // Sleep for remaining time in interval
            let elapsed = loop_start.elapsed();
            if elapsed < READ_INTERVAL {
                thread::sleep(READ_INTERVAL - elapsed);
            }
        }
    }

    /// Check if the arm is connected
    pub fn is_connected(&self) -> bool {
        self.state.read().connected
    }

    /// Get the current state
    pub fn state(&self) -> So101State {
        self.state.read().clone()
    }

    /// Read current positions (direct read, not from cached state)
    pub fn read_positions(&self) -> [f32; NUM_JOINTS] {
        let mut proto = self.protocol.lock();
        match proto.sync_read_positions(&SERVO_IDS) {
            Ok(raw) => {
                let mut positions = [0.0f32; NUM_JOINTS];
                for (i, &r) in raw.iter().enumerate().take(NUM_JOINTS) {
                    positions[i] = raw_to_radians(r);
                }
                positions
            }
            Err(e) => {
                tracing::warn!("Failed to read positions: {}", e);
                self.state.read().joint_positions
            }
        }
    }

    /// Set target positions for all joints
    pub fn set_positions(&self, positions: &[f32]) -> Result<(), Error> {
        if positions.len() != NUM_JOINTS {
            return Err(Error::Hardware(format!(
                "Expected {} positions, got {}",
                NUM_JOINTS,
                positions.len()
            )));
        }

        let mut raw_positions = ArrayVec::<u16, 8>::new();
        for &r in positions.iter().take(NUM_JOINTS) {
            raw_positions.push(radians_to_raw(r));
        }

        let mut proto = self.protocol.lock();
        proto.sync_write_positions(&SERVO_IDS, &raw_positions)
    }

    /// Enable or disable torque for all servos
    pub fn set_torque_enable(&self, enabled: bool) -> Result<(), Error> {
        let mut proto = self.protocol.lock();
        proto.set_torque_all(&SERVO_IDS, enabled)
    }

    /// Move to home position
    pub fn go_home(&self) -> Result<(), Error> {
        // HOME_POSITIONS are all 2048 (center), which maps to 0.0 radians
        self.set_positions(&[0.0f32; NUM_JOINTS])
    }

    /// Disconnect from the arm
    pub fn disconnect(&mut self) -> Result<(), Error> {
        tracing::info!("Disconnecting from SO-101");

        // Stop the reader thread
        self.running.store(false, Ordering::SeqCst);

        // Wait for reader thread to finish
        if let Some(handle) = self.reader_handle.take() {
            handle
                .join()
                .map_err(|_| Error::Hardware("Reader thread panicked".into()))?;
        }

        // Update state
        self.state.write().connected = false;

        tracing::info!("SO-101 disconnected");
        Ok(())
    }
}

impl Drop for So101 {
    fn drop(&mut self) {
        // Ensure we clean up even if disconnect wasn't called
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.reader_handle.take() {
            let _ = handle.join();
        }
    }
}
