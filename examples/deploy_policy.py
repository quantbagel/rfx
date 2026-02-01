#!/usr/bin/env python3
"""
Deploy a trained policy to the Go2 robot

This example shows how to:
1. Load a trained policy
2. Connect to the robot (sim or real)
3. Run inference in a control loop

Usage:
    # Simulation mode (default)
    python examples/deploy_policy.py

    # Real robot
    python examples/deploy_policy.py --real --ip 192.168.123.161

Requirements:
    pip install tinygrad
    # For real robot: maturin develop (to build Rust extension)
"""

import argparse
import time
import numpy as np

import rfx
from rfx.nn import MLP, JitPolicy
from rfx.envs import Go2Env


def main():
    parser = argparse.ArgumentParser(description="Deploy policy to Go2")
    parser.add_argument(
        "--model",
        type=str,
        default="walking_policy.safetensors",
        help="Path to trained policy",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real robot instead of simulation",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="192.168.123.161",
        help="Robot IP address (for real robot)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration to run in seconds",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Control loop rate in Hz",
    )
    args = parser.parse_args()

    print("Pi Policy Deployment")
    print("=" * 50)

    # Load trained policy
    print(f"Loading policy from: {args.model}")
    try:
        policy = MLP.load(args.model)
        print(f"  Loaded: {policy}")
    except FileNotFoundError:
        print(f"  Model not found, using random policy for demo")
        policy = MLP(obs_dim=48, act_dim=12, hidden=[256, 256])

    # Wrap with JIT for faster inference
    policy = JitPolicy(policy)
    print(f"  JIT enabled: {policy}")

    # Create environment
    if args.real:
        print(f"\nConnecting to real robot at {args.ip}...")
        env = Go2Env(sim=False, robot_ip=args.ip)
    else:
        print("\nUsing simulation backend...")
        env = Go2Env(sim=True)

    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # Set velocity commands (can be changed dynamically)
    env.set_commands(vx=0.5, vy=0.0, yaw_rate=0.0)

    # Control loop
    print(f"\nRunning control loop at {args.rate} Hz for {args.duration}s...")
    print("-" * 50)

    dt = 1.0 / args.rate
    num_steps = int(args.duration * args.rate)
    total_reward = 0.0
    start_time = time.time()

    try:
        from tinygrad import Tensor

        for step in range(num_steps):
            loop_start = time.time()

            # Convert observation to tensor
            obs_tensor = Tensor(obs.reshape(1, -1).astype(np.float32))

            # Run policy inference
            action_tensor = policy(obs_tensor)
            action = action_tensor.numpy().flatten()

            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Print progress every second
            if step % int(args.rate) == 0:
                elapsed = time.time() - start_time
                print(
                    f"t={elapsed:5.1f}s | "
                    f"step={step:5d} | "
                    f"reward={reward:6.3f} | "
                    f"total={total_reward:8.2f}"
                )

            if done:
                print("Episode terminated, resetting...")
                obs = env.reset()

            # Sleep to maintain loop rate
            loop_time = time.time() - loop_start
            sleep_time = dt - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()

    print("-" * 50)
    elapsed = time.time() - start_time
    print(f"Deployment complete!")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward/step: {total_reward / num_steps:.4f}")


if __name__ == "__main__":
    main()
