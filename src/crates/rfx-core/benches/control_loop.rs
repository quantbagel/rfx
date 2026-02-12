//! Benchmarks for rfx-core optimized components
//!
//! Run with: cargo bench --bench control_loop

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rfx_core::control::{DerivativeFilter, Pid, PidConfig};
use rfx_core::math::{Filter, LowPassFilter, MovingAverageFilter, Quaternion, Transform};

// ── PID Benchmarks ──────────────────────────────────────────────────────────

fn bench_pid_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("PID");

    group.bench_function("P controller update", |b| {
        let mut pid = Pid::p(10.0);
        let dt = 0.002;
        b.iter(|| black_box(pid.update(1.0, 0.5, dt)))
    });

    group.bench_function("PI controller update", |b| {
        let mut pid = Pid::pi(10.0, 1.0);
        let dt = 0.002;
        b.iter(|| black_box(pid.update(1.0, 0.5, dt)))
    });

    group.bench_function("PID controller update", |b| {
        let mut pid = Pid::pid(10.0, 1.0, 0.5);
        let dt = 0.002;
        b.iter(|| black_box(pid.update(1.0, 0.5, dt)))
    });

    group.bench_function("PID with filter", |b| {
        let config = PidConfig::new(10.0, 1.0, 0.5)
            .with_derivative_filter(DerivativeFilter::MODERATE)
            .with_limits(-100.0, 100.0)
            .with_integral_limit(50.0);
        let mut pid = Pid::new(config);
        let dt = 0.002;
        b.iter(|| black_box(pid.update(1.0, 0.5, dt)))
    });

    group.finish();
}

fn bench_pid_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("PID Sequence");

    for n in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("updates", n), n, |b, &n| {
            let mut pid = Pid::pid(10.0, 1.0, 0.5);
            let dt = 0.002;
            b.iter(|| {
                for i in 0..n {
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

// ── Quaternion Benchmarks (direct Hamilton product vs nalgebra) ──────────────

fn bench_quaternion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quaternion");

    group.bench_function("identity", |b| b.iter(|| black_box(Quaternion::identity())));

    group.bench_function("from_euler", |b| {
        b.iter(|| black_box(Quaternion::from_euler(0.1, 0.2, 0.3)))
    });

    group.bench_function("to_euler", |b| {
        let q = Quaternion::from_euler(0.1, 0.2, 0.3);
        b.iter(|| black_box(q.to_euler()))
    });

    group.bench_function("multiply", |b| {
        let q1 = Quaternion::from_euler(0.1, 0.2, 0.3);
        let q2 = Quaternion::from_euler(0.05, 0.1, 0.15);
        b.iter(|| black_box(q1.multiply(&q2)))
    });

    group.bench_function("rotate_vector", |b| {
        let q = Quaternion::from_euler(0.1, 0.2, 0.3);
        let v = [1.0, 2.0, 3.0];
        b.iter(|| black_box(q.rotate_vector(v)))
    });

    group.bench_function("slerp", |b| {
        let q1 = Quaternion::from_euler(0.0, 0.0, 0.0);
        let q2 = Quaternion::from_euler(0.1, 0.2, 0.3);
        b.iter(|| black_box(q1.slerp(&q2, 0.5)))
    });

    group.bench_function("inverse", |b| {
        let q = Quaternion::from_euler(0.1, 0.2, 0.3);
        b.iter(|| black_box(q.inverse()))
    });

    // Compound: typical control loop quaternion work
    group.bench_function("compound_orientation_update", |b| {
        let current = Quaternion::from_euler(0.1, 0.2, 0.3);
        let delta = Quaternion::from_euler(0.001, 0.002, 0.003);
        let v = [0.0, 0.0, -9.81];
        b.iter(|| {
            let q = current.multiply(&delta);
            let gravity_body = q.rotate_vector(v);
            black_box((q, gravity_body))
        })
    });

    group.finish();
}

// ── Transform Benchmarks ────────────────────────────────────────────────────

fn bench_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transform");

    group.bench_function("identity", |b| b.iter(|| black_box(Transform::identity())));

    group.bench_function("compose", |b| {
        let t1 = Transform::new([1.0, 2.0, 3.0], Quaternion::from_euler(0.1, 0.2, 0.3));
        let t2 = Transform::new([0.5, 0.5, 0.5], Quaternion::from_euler(0.05, 0.1, 0.15));
        b.iter(|| black_box(t1.compose(&t2)))
    });

    group.bench_function("transform_point", |b| {
        let t = Transform::new([1.0, 2.0, 3.0], Quaternion::from_euler(0.1, 0.2, 0.3));
        let p = [4.0, 5.0, 6.0];
        b.iter(|| black_box(t.transform_point(p)))
    });

    group.bench_function("inverse", |b| {
        let t = Transform::new([1.0, 2.0, 3.0], Quaternion::from_euler(0.1, 0.2, 0.3));
        b.iter(|| black_box(t.inverse()))
    });

    group.finish();
}

// ── Filter Benchmarks (ring buffer, devirtualized) ──────────────────────────

fn bench_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("Filters");

    // Low-pass filter single update
    group.bench_function("low_pass_update", |b| {
        let mut f = LowPassFilter::new(0.1);
        f.update(0.0); // initialize
        let mut i = 0.0_f64;
        b.iter(|| {
            i += 0.01;
            black_box(f.update(i.sin()))
        })
    });

    // Moving average — ring buffer, various window sizes
    for window in [4, 16, 64] {
        group.bench_function(&format!("moving_avg_w{window}_update"), |b| {
            let mut f = MovingAverageFilter::<64>::new(window);
            let mut i = 0.0_f64;
            b.iter(|| {
                i += 0.01;
                black_box(f.update(i.sin()))
            })
        });
    }

    // Low-pass filter burst (simulate 500Hz control loop, 1 second)
    group.bench_function("low_pass_500_updates", |b| {
        let mut f = LowPassFilter::from_cutoff(10.0, 500.0);
        b.iter(|| {
            for i in 0..500 {
                let signal = (i as f64 * 0.01).sin() + (i as f64 * 0.1).sin() * 0.1;
                black_box(f.update(signal));
            }
            f.reset();
        })
    });

    // Moving average burst
    group.bench_function("moving_avg_w16_500_updates", |b| {
        let mut f = MovingAverageFilter::<64>::new(16);
        b.iter(|| {
            for i in 0..500 {
                let signal = (i as f64 * 0.01).sin();
                black_box(f.update(signal));
            }
            f.reset();
        })
    });

    group.finish();
}

// ── Channel Benchmarks ──────────────────────────────────────────────────────

fn bench_channel(c: &mut Criterion) {
    let mut group = c.benchmark_group("Channel");

    // Bounded send+recv
    group.bench_function("bounded_send_recv", |b| {
        let (tx, rx) = rfx_core::comm::bounded_channel::<u64>(64);
        b.iter(|| {
            tx.send(42).unwrap();
            black_box(rx.recv().unwrap());
        })
    });

    // Unbounded send+recv
    group.bench_function("unbounded_send_recv", |b| {
        let (tx, rx) = rfx_core::comm::unbounded_channel::<u64>();
        b.iter(|| {
            tx.send(42).unwrap();
            black_box(rx.recv().unwrap());
        })
    });

    // Latest (drain to last) — simulates the teleop pattern
    group.bench_function("latest_after_burst_10", |b| {
        let (tx, rx) = rfx_core::comm::bounded_channel::<u64>(64);
        b.iter(|| {
            for i in 0..10 {
                tx.try_send(i).ok();
            }
            black_box(rx.latest())
        })
    });

    group.finish();
}

// ── Control Loop Stats Benchmark ────────────────────────────────────────────

fn bench_control_loop_overhead(c: &mut Criterion) {
    use std::time::Duration;

    let mut group = c.benchmark_group("Control Loop Overhead");

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

// ── Derivative Filter Benchmarks ────────────────────────────────────────────

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

// ── Full Simulated Control Tick ─────────────────────────────────────────────

fn bench_full_control_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full Control Tick");

    // Simulate a single 500Hz control tick:
    //   read IMU quaternion → rotate gravity → PID on 12 joints → write commands
    group.bench_function("12_joint_pid_tick", |b| {
        let orientation = Quaternion::from_euler(0.05, 0.02, 0.01);
        let gravity_world = [0.0, 0.0, -9.81];
        let mut pids = [Pid::pid(20.0, 0.1, 0.5); 12];
        let setpoints = [0.0_f64; 12];
        let measurements = [
            0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.02, -0.03,
        ];
        let dt = 0.002;

        b.iter(|| {
            // 1. Rotate gravity to body frame
            let _gravity_body = orientation.rotate_vector(gravity_world);

            // 2. Run PID on all 12 joints
            let torques: [f64; 12] =
                std::array::from_fn(|i| pids[i].update(setpoints[i], measurements[i], dt));
            black_box(torques)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pid_update,
    bench_pid_sequence,
    bench_quaternion,
    bench_transform,
    bench_filters,
    bench_channel,
    bench_control_loop_overhead,
    bench_derivative_filter,
    bench_full_control_tick,
);
criterion_main!(benches);
