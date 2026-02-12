//! Python bindings for the rfx robotics framework
//!
//! This crate provides PyO3 bindings to expose rfx-core functionality to Python.
//! The GIL is released immediately in all long-running operations to allow
//! Python threads to run concurrently.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

mod bindings;

use bindings::*;

/// The rfx Python module
#[pymodule]
fn _rfx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tracing for debugging
    let _ = tracing_subscriber::fmt::try_init();

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("VERSION", rfx_core::VERSION)?;

    // Math types
    m.add_class::<PyQuaternion>()?;
    m.add_class::<PyTransform>()?;
    m.add_class::<PyLowPassFilter>()?;

    // Communication types
    m.add_class::<PyTopic>()?;

    // Control types
    m.add_class::<PyPid>()?;
    m.add_class::<PyPidConfig>()?;
    m.add_class::<PyControlLoopHandle>()?;

    // Hardware types - Go2
    m.add_class::<PyGo2>()?;
    m.add_class::<PyGo2Config>()?;
    m.add_class::<PyGo2State>()?;
    m.add_class::<PyImuState>()?;
    m.add_class::<PyMotorState>()?;
    m.add_class::<PyMotorCmd>()?;

    // Hardware types - SO-101
    m.add_class::<PySo101>()?;
    m.add_class::<PySo101Config>()?;
    m.add_class::<PySo101State>()?;

    // Motor index constants
    let motor_idx = PyModule::new(m.py(), "motor_idx")?;
    motor_idx.add("FR_HIP", rfx_core::hardware::motor_idx::FR_HIP)?;
    motor_idx.add("FR_THIGH", rfx_core::hardware::motor_idx::FR_THIGH)?;
    motor_idx.add("FR_CALF", rfx_core::hardware::motor_idx::FR_CALF)?;
    motor_idx.add("FL_HIP", rfx_core::hardware::motor_idx::FL_HIP)?;
    motor_idx.add("FL_THIGH", rfx_core::hardware::motor_idx::FL_THIGH)?;
    motor_idx.add("FL_CALF", rfx_core::hardware::motor_idx::FL_CALF)?;
    motor_idx.add("RR_HIP", rfx_core::hardware::motor_idx::RR_HIP)?;
    motor_idx.add("RR_THIGH", rfx_core::hardware::motor_idx::RR_THIGH)?;
    motor_idx.add("RR_CALF", rfx_core::hardware::motor_idx::RR_CALF)?;
    motor_idx.add("RL_HIP", rfx_core::hardware::motor_idx::RL_HIP)?;
    motor_idx.add("RL_THIGH", rfx_core::hardware::motor_idx::RL_THIGH)?;
    motor_idx.add("RL_CALF", rfx_core::hardware::motor_idx::RL_CALF)?;
    motor_idx.add("NUM_MOTORS", rfx_core::hardware::motor_idx::NUM_MOTORS)?;
    m.add_submodule(&motor_idx)?;

    // Motor names
    m.add("MOTOR_NAMES", rfx_core::hardware::MOTOR_NAMES.to_vec())?;

    // Simulation types (v2)
    m.add_class::<PyPhysicsConfig>()?;
    m.add_class::<PySimConfig>()?;
    m.add_class::<PySimState>()?;
    m.add_class::<PyMockSimBackend>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(motor_index_by_name, m)?)?;
    m.add_function(wrap_pyfunction!(run_control_loop, m)?)?;

    // Channel types
    m.add_class::<PySender>()?;
    m.add_class::<PyReceiver>()?;
    m.add_function(wrap_pyfunction!(bindings::channel, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::unbounded_channel, m)?)?;

    // Stream types
    m.add_class::<PyStream>()?;
    m.add_function(wrap_pyfunction!(bindings::stream_from_receiver, m)?)?;

    Ok(())
}

/// Get motor index by name
#[pyfunction]
#[must_use]
fn motor_index_by_name(name: &str) -> PyResult<Option<usize>> {
    Ok(rfx_core::hardware::motor_index_by_name(name))
}

/// Run a control loop with a Python callback
///
/// The callback receives (iteration, dt) and should return True to continue or False to stop.
#[pyfunction]
#[pyo3(signature = (rate_hz, callback, name = None, max_iterations = None))]
fn run_control_loop(
    py: Python<'_>,
    rate_hz: f64,
    callback: PyObject,
    name: Option<&str>,
    max_iterations: Option<u64>,
) -> PyResult<PyControlLoopStats> {
    let config =
        rfx_core::control::ControlLoopConfig::new(rate_hz).with_name(name.unwrap_or("python_loop"));

    // Clone the callback with GIL before releasing it
    let callback_clone = callback.clone_ref(py);

    // Run the control loop, releasing GIL during sleep periods
    let stats = py.allow_threads(|| {
        rfx_core::control::ControlLoop::run(config, move |iter, dt| {
            // Check max iterations outside GIL
            if let Some(max) = max_iterations {
                if iter >= max {
                    return false;
                }
            }

            // Acquire GIL only when calling Python callback
            Python::with_gil(|py| match callback_clone.call1(py, (iter, dt)) {
                Ok(result) => result.is_truthy(py).unwrap_or(false),
                Err(e) => {
                    e.print(py);
                    false
                }
            })
        })
    });

    match stats {
        Ok(s) => Ok(PyControlLoopStats::from(s)),
        Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
    }
}
