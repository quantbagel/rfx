#!/usr/bin/env python3
"""
Basic locomotion example for the Unitree Go2

This example demonstrates:
- Connecting to the Go2 robot
- Using sport mode commands (walk, stand)
- Reading robot state
"""

import time

import rfx


def main():
    # Connect to the robot (use default IP or specify)
    print("Connecting to Go2...")
    go2 = rfx.Go2.connect()  # Uses default IP: 192.168.123.161

    if not go2.is_connected():
        print("Failed to connect!")
        return

    print(f"Connected! Robot: {go2}")

    try:
        # Stand up first
        print("Standing...")
        go2.stand()
        time.sleep(1.0)

        # Get initial state
        state = go2.state()
        print(
            f"Initial state: IMU roll={state.imu.roll_deg:.1f}°, pitch={state.imu.pitch_deg:.1f}°"
        )

        # Walk forward
        print("Walking forward...")
        go2.walk(vx=0.3, vy=0.0, vyaw=0.0)  # 0.3 m/s forward

        # Walk for 2 seconds
        for i in range(20):
            time.sleep(0.1)
            state = go2.state()
            print(f"  Position: {state.position}, IMU: {state.imu.rpy}")

        # Stop walking
        print("Stopping...")
        go2.stand()
        time.sleep(0.5)

        # Turn in place
        print("Turning...")
        go2.walk(vx=0.0, vy=0.0, vyaw=0.5)  # 0.5 rad/s rotation
        time.sleep(2.0)

        # Stop
        print("Standing...")
        go2.stand()
        time.sleep(0.5)

        # Final state
        state = go2.state()
        print(f"Final state: position={state.position}")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        # Always stand before disconnecting
        print("Cleaning up...")
        go2.stand()
        go2.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
