#!/usr/bin/env python3
"""
Train a walking policy using PPO and tinygrad

This example demonstrates the complete training loop for a Go2 locomotion policy:
1. Create environment and policy
2. Collect rollouts
3. Update policy with PPO
4. Save trained model

Usage:
    python examples/train_walking.py

Requirements:
    pip install tinygrad
"""

import rfx
from rfx.nn import go2_actor_critic
from rfx.rl import PPOTrainer, collect_rollout
from rfx.envs import Go2Env


def main():
    print("Pi Walking Policy Training")
    print("=" * 50)

    # Create environment (simulation mode)
    env = Go2Env(sim=True)
    print(f"Environment: Go2Env (sim=True)")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Create policy (ActorCritic for PPO)
    policy = go2_actor_critic(hidden=[256, 256])
    print(f"\nPolicy: {policy}")

    # Create trainer
    trainer = PPOTrainer(
        policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        update_epochs=10,
        minibatch_size=64,
    )
    print(f"Trainer: PPO (lr=3e-4)")

    # Training loop
    num_epochs = 100
    steps_per_epoch = 2048

    print(f"\nTraining for {num_epochs} epochs, {steps_per_epoch} steps each...")
    print("-" * 50)

    best_reward = float("-inf")

    for epoch in range(num_epochs):
        # Collect rollout
        rollout = collect_rollout(env, policy, steps=steps_per_epoch)

        # Update policy
        metrics = trainer.update(rollout)

        # Print progress
        mean_reward = metrics["mean_reward"]
        total_reward = metrics["total_reward"]
        policy_loss = metrics["policy_loss"]
        value_loss = metrics["value_loss"]

        print(
            f"Epoch {epoch:3d} | "
            f"Reward: {mean_reward:7.3f} (total: {total_reward:8.1f}) | "
            f"Policy Loss: {policy_loss:7.4f} | "
            f"Value Loss: {value_loss:7.4f}"
        )

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            policy.save("walking_policy_best.safetensors")

    # Save final model
    policy.save("walking_policy.safetensors")
    print("-" * 50)
    print(f"\nTraining complete!")
    print(f"  Best reward: {best_reward:.2f}")
    print(f"  Model saved to: walking_policy.safetensors")
    print(f"  Best model saved to: walking_policy_best.safetensors")


if __name__ == "__main__":
    main()
