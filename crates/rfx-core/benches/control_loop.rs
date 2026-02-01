//! Benchmarks for control loop and PID components
//!
//! Run with: cargo bench --bench control_loop

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rfx_core::control::{Pid, PidConfig, DerivativeFilter};
use rfx_core::math::Quaternion;

/// Benchmark PID controller update
fn bench_pid_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("PID");

    // P controller
    group.bench_function("P controller update", |b| {
        let mut pid = Pid::p(10.0);
        let dt = 0.002; // 500Hz

        b.iter(|| {
            black_box(pid.update(1.0, 0.5, dt))
        })
    });

    // PI controller
    group.bench_function("PI controller update", |b| {
        let mut pid = Pid::pi(10.0, 1.0);
        let dt = 0.002;

        b.iter(|| {
            black_box(pid.update(1.0, 0.5, dt))
        })
    });

    // Full PID controller
    group.bench_function("PID controller update", |b| {
        let mut pid = Pid::pid(10.0, 1.0, 0.5);
        let dt = 0.002;

        b.iter(|| {
            black_box(pid.update(1.0, 0.5, dt))
        })
    });

    // PID with derivative filter
    group.bench_function("PID with filter", |b| {
        let config = PidConfig::new(10.0, 1.0, 0.5)
            .with_derivative_filter(DerivativeFilter::MODERATE)
            .with_limits(-100.0, 100.0)
            .with_integral_limit(50.0);
        let mut pid = Pid::new(config);
        let dt = 0.002;

        b.iter(|| {
            black_box(pid.update(1.0, 0.5, dt))
        })
    });

    group.finish();
}

/// Benchmark PID controller with varying numbers of sequential updates
fn bench_pid_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("PID Sequence");

    for n in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("updates", n), n, |b, &n| {
            let mut pid = Pid::pid(10.0, 1.0, 0.5);
            let dt = 0.002;

            b.iter(|| {
                for i in 0..n {
                    // Simulate a decaying error
                    let setpoint = 1.0;
                    let measurement = setpoint * (1.0 - (-0.1 * i as f64).exp());
                    black_box(pid.update(setpoint, measurement, dt));
                }
                pid.reset();
            })
        });
    }

    group.finish();
}

/// Benchmark quaternion operations (common in control loops)
fn bench_quaternion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quaternion");

    // Identity creation
    group.bench_function("identity", |b| {
        b.iter(|| black_box(Quaternion::identity()))
    });

    // From Euler angles
    group.bench_function("from_euler", |b| {
        b.iter(|| black_box(Quaternion::from_euler(0.1, 0.2, 0.3)))
    });

    // To Euler angles
    group.bench_function("to_euler", |b| {
        let q = Quaternion::from_euler(0.1, 0.2, 0.3);
        b.iter(|| black_box(q.to_euler()))
    });

    // Quaternion multiplication
    group.bench_function("multiply", |b| {
        let q1 = Quaternion::from_euler(0.1, 0.2, 0.3);
        let q2 = Quaternion::from_euler(0.05, 0.1, 0.15);

        b.iter(|| black_box(q1.multiply(&q2)))
    });

    // Vector rotation
    group.bench_function("rotate_vector", |b| {
        let q = Quaternion::from_euler(0.1, 0.2, 0.3);
        let v = [1.0, 2.0, 3.0];

        b.iter(|| black_box(q.rotate_vector(v)))
    });

    // SLERP interpolation
    group.bench_function("slerp", |b| {
        let q1 = Quaternion::from_euler(0.0, 0.0, 0.0);
        let q2 = Quaternion::from_euler(0.1, 0.2, 0.3);

        b.iter(|| black_box(q1.slerp(&q2, 0.5)))
    });

    // Inverse
    group.bench_function("inverse", |b| {
        let q = Quaternion::from_euler(0.1, 0.2, 0.3);

        b.iter(|| black_box(q.inverse()))
    });

    group.finish();
}

/// Benchmark control loop overhead (without actual callback work)
fn bench_control_loop_overhead(c: &mut Criterion) {
    use std::time::Duration;

    let mut group = c.benchmark_group("Control Loop Overhead");

    // Measure the overhead of ControlLoopStats::update
    group.bench_function("stats_update", |b| {
        let mut stats = rfx_core::control::ControlLoopStats::default();
        let period = Duration::from_millis(2);
        let execution = Duration::from_micros(500);

        b.iter(|| {
            stats.update(execution, period);
        })
    });

    group.finish();
}

/// Benchmark derivative filter alpha computation
fn bench_derivative_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("Derivative Filter");

    group.bench_function("disabled alpha", |b| {
        let filter = DerivativeFilter::Disabled;
        b.iter(|| black_box(filter.alpha()))
    });

    group.bench_function("low_pass alpha", |b| {
        let filter = DerivativeFilter::LowPass { alpha: 0.5 };
        b.iter(|| black_box(filter.alpha()))
    });

    group.bench_function("cutoff_freq alpha", |b| {
        let filter = DerivativeFilter::CutoffFrequency {
            cutoff_hz: 20.0,
            sample_rate_hz: 500.0,
        };
        b.iter(|| black_box(filter.alpha()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pid_update,
    bench_pid_sequence,
    bench_quaternion,
    bench_control_loop_overhead,
    bench_derivative_filter,
);
criterion_main!(benches);
