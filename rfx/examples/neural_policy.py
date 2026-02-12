#!/usr/bin/env python3
"""
Neural Policy Example (v2)

Demonstrates how to use neural network policies with rfx.
This example uses the mock simulation backend for testing.

For real ONNX model inference, see examples/onnx_policy.py (requires onnxruntime).
"""

import rfx
import time
import numpy as np


def main():
    print("pi Neural Policy Example (v2)")
    print("=" * 50)

    # Create a mock simulation backend
    config = rfx.SimConfig.mock()
    sim = rfx.MockSimBackend(config)

    print(f"Simulation backend: {sim.name()}")
    print(f"Initial time: {sim.sim_time():.3f}s")

    # Reset simulation
    state = sim.reset()
    print(f"\nAfter reset: {state}")

    # Simple policy: move joints towards a target position
    target_positions = [0.5] * 12  # Target position for all joints

    print("\nRunning simulation with simple policy...")
    print("Target joint position: 0.5 rad")
    print()

    # Run for 500 steps
    for step in range(500):
        # Get current state
        current_positions = state.joint_positions()

        # Simple proportional controller
        kp = 0.5
        actions = [
            kp * (target - current) for target, current in zip(target_positions, current_positions)
        ]

        # Step simulation
        state, done = sim.step(actions)

        # Print every 100 steps
        if step % 100 == 0:
            print(
                f"Step {step:4d} | Time: {state.sim_time():.3f}s | "
                f"Joint[0]: {current_positions[0]:.4f} rad"
            )

        if done:
            print("Episode terminated!")
            break

    # Final state
    final_positions = state.joint_positions()
    print(f"\nFinal joint positions:")
    for i, pos in enumerate(final_positions):
        print(f"  Joint {i:2d}: {pos:.4f} rad (target: 0.5)")

    print("\n" + "=" * 50)
    print("Example complete!")


def demo_physics_config():
    """Demonstrate different physics configurations."""
    print("\nPhysics Configuration Examples:")
    print("-" * 40)

    # Default config
    default = rfx.PhysicsConfig()
    print(f"Default: {default}")

    # Fast config (for rapid prototyping)
    fast = rfx.PhysicsConfig.fast()
    print(f"Fast:    dt={fast.dt}, substeps={fast.substeps}")

    # Accurate config (for precise simulation)
    accurate = rfx.PhysicsConfig.accurate()
    print(f"Accurate: dt={accurate.dt}, substeps={accurate.substeps}")


def demo_sim_backends():
    """Demonstrate different simulation backend configurations."""
    print("\nSimulation Backend Configurations:")
    print("-" * 40)

    # Mock (for testing)
    mock_cfg = rfx.SimConfig.mock()
    print(f"Mock:      backend='{mock_cfg.backend}'")

    # Isaac Sim (placeholder)
    isaac_cfg = rfx.SimConfig.isaac_sim()
    print(f"Isaac Sim: backend='{isaac_cfg.backend}'")

    # Genesis (placeholder)
    genesis_cfg = rfx.SimConfig.genesis()
    print(f"Genesis:   backend='{genesis_cfg.backend}'")

    # MuJoCo (placeholder)
    mujoco_cfg = rfx.SimConfig.mujoco()
    print(f"MuJoCo:    backend='{mujoco_cfg.backend}'")

    # Parallel environments
    parallel_cfg = rfx.SimConfig.mock().with_num_envs(4096)
    print(f"Parallel:  num_envs={parallel_cfg.num_envs}")


if __name__ == "__main__":
    demo_physics_config()
    demo_sim_backends()
    print()
    main()
