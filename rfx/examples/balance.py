#!/usr/bin/env python3
"""
Low-level balance control example for the Unitree Go2

This example demonstrates:
- Using EDU mode for low-level motor control
- Reading IMU data
- Simple PID-based balance control
- Running a control loop at 500Hz

NOTE: This requires the EDU version of the Go2 firmware.
"""

import time

import rfx


def main():
    # Connect in EDU mode for low-level control
    print("Connecting to Go2 in EDU mode...")
    config = rfx.Go2Config("192.168.123.161").with_edu_mode()
    go2 = rfx.Go2.connect(config)

    if not go2.is_connected():
        print("Failed to connect!")
        return

    print(f"Connected in EDU mode: {go2}")

    # PID controllers for roll and pitch
    roll_pid = rfx.Pid.pid(kp=50.0, ki=0.5, kd=2.0)
    pitch_pid = rfx.Pid.pid(kp=50.0, ki=0.5, kd=2.0)

    # Motor gains
    KP = 20.0
    KD = 0.5

    # Control loop state
    iteration_count = 0
    max_iterations = 5000  # Run for ~10 seconds at 500Hz

    def balance_callback(iteration: int, dt: float) -> bool:
        nonlocal iteration_count
        iteration_count = iteration

        # Get current state
        state = go2.state()
        roll = state.imu.roll  # radians
        pitch = state.imu.pitch

        # Compute corrections
        # Target is level (roll=0, pitch=0)
        roll_correction = roll_pid.update(setpoint=0.0, measurement=roll, dt=dt)
        pitch_correction = pitch_pid.update(setpoint=0.0, measurement=pitch, dt=dt)

        # Apply corrections to hip joints
        # Front legs adjust for pitch, all legs adjust for roll
        try:
            # Roll correction: tilt left/right by adjusting hip abduction
            # FR and RR hips move one way, FL and RL move the other
            go2.set_motor_position(
                rfx.motor_idx.FR_HIP,
                position=float(-roll_correction * 0.1),
                kp=KP,
                kd=KD,
            )
            go2.set_motor_position(
                rfx.motor_idx.FL_HIP,
                position=float(roll_correction * 0.1),
                kp=KP,
                kd=KD,
            )
            go2.set_motor_position(
                rfx.motor_idx.RR_HIP,
                position=float(-roll_correction * 0.1),
                kp=KP,
                kd=KD,
            )
            go2.set_motor_position(
                rfx.motor_idx.RL_HIP,
                position=float(roll_correction * 0.1),
                kp=KP,
                kd=KD,
            )
        except Exception as e:
            print(f"Motor command error: {e}")
            return False

        # Print status every 100 iterations
        if iteration % 100 == 0:
            print(
                f"[{iteration:5d}] roll={roll * 57.3:6.2f}° pitch={pitch * 57.3:6.2f}° "
                f"dt={dt * 1000:.2f}ms corr=({roll_correction:.3f}, {pitch_correction:.3f})"
            )

        return iteration < max_iterations

    print("Starting balance control loop at 500Hz...")
    print("Press Ctrl+C to stop")
    print()

    try:
        # Run the control loop
        stats = rfx.run_control_loop(
            rate_hz=500.0,
            callback=balance_callback,
            name="balance",
        )

        print()
        print(f"Control loop completed!")
        print(f"  Iterations: {stats.iterations}")
        print(f"  Overruns: {stats.overruns}")
        print(f"  Avg iteration time: {stats.avg_iteration_time_ms:.3f}ms")
        print(f"  Max iteration time: {stats.max_iteration_time_ms:.3f}ms")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        print("Disconnecting...")
        go2.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
