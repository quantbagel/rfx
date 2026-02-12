"""
Reinforcement learning utilities for training locomotion policies

Provides a minimal PPO implementation and rollout collection for training
neural network policies with tinygrad.

Example:
    >>> from rfx.nn import go2_actor_critic
    >>> from rfx.rl import PPOTrainer, collect_rollout
    >>> from rfx.envs import Go2Env
    >>>
    >>> env = Go2Env(sim=True)
    >>> policy = go2_actor_critic()
    >>> trainer = PPOTrainer(policy)
    >>>
    >>> for epoch in range(100):
    ...     rollout = collect_rollout(env, policy, steps=2048)
    ...     metrics = trainer.update(rollout)
    ...     print(f"Epoch {epoch}: reward={metrics['mean_reward']:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

try:
    from tinygrad import Tensor
    from tinygrad.nn.optim import Adam
    from tinygrad.nn.state import get_parameters

    TINYGRAD_AVAILABLE = True
except ImportError:
    TINYGRAD_AVAILABLE = False
    Tensor = object

if TYPE_CHECKING:
    from rfx.nn import ActorCritic, Policy
    from rfx.envs import BaseEnv


def _check_tinygrad():
    if not TINYGRAD_AVAILABLE:
        raise ImportError(
            "tinygrad is required for RL training. Install with: pip install tinygrad"
        )


@dataclass
class Rollout:
    """
    Stores trajectory data from environment rollouts.

    All arrays have shape (num_steps, ...) where ... depends on the data type.

    Attributes:
        observations: Observations at each step (num_steps, obs_dim)
        actions: Actions taken at each step (num_steps, act_dim)
        rewards: Rewards received at each step (num_steps,)
        dones: Episode termination flags (num_steps,)
        values: Value estimates at each step (num_steps,)
        log_probs: Log probabilities of actions (num_steps,)
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    log_probs: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def num_steps(self) -> int:
        """Number of steps in the rollout."""
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        """Total reward accumulated."""
        return float(np.sum(self.rewards))

    @property
    def mean_reward(self) -> float:
        """Mean reward per step."""
        return float(np.mean(self.rewards))

    def compute_returns(self, gamma: float = 0.99) -> np.ndarray:
        """
        Compute discounted returns.

        Args:
            gamma: Discount factor

        Returns:
            Array of discounted returns (num_steps,)
        """
        returns = np.zeros_like(self.rewards)
        running_return = 0.0

        for t in reversed(range(self.num_steps)):
            if self.dones[t]:
                running_return = 0.0
            running_return = self.rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns

    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Array of advantages (num_steps,)
        """
        if len(self.values) == 0:
            raise ValueError("Values required for GAE computation")

        advantages = np.zeros_like(self.rewards)
        last_gae = 0.0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]

            if self.dones[t]:
                next_value = 0.0
                last_gae = 0.0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae

        return advantages


def collect_rollout(
    env: "BaseEnv",
    policy: "Policy",
    steps: int,
    deterministic: bool = False,
) -> Rollout:
    """
    Collect experience by running the policy in the environment.

    Args:
        env: Environment to collect from
        policy: Policy to use for action selection
        steps: Number of steps to collect
        deterministic: If True, use deterministic actions (no exploration)

    Returns:
        Rollout containing the collected trajectory

    Example:
        >>> env = Go2Env(sim=True)
        >>> policy = go2_mlp()
        >>> rollout = collect_rollout(env, policy, steps=2048)
        >>> print(f"Collected {rollout.num_steps} steps, total reward: {rollout.total_reward:.2f}")
    """
    _check_tinygrad()

    observations = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []

    obs = env.reset()
    has_actor_critic = hasattr(policy, "get_action_and_value")

    for _ in range(steps):
        obs_tensor = Tensor(obs.reshape(1, -1).astype(np.float32))

        if has_actor_critic:
            # ActorCritic: get action, log_prob, entropy, value
            action_tensor, log_prob, _, value = policy.get_action_and_value(obs_tensor)
            values.append(float(value.numpy()))
            log_probs.append(float(log_prob.numpy()))
        else:
            # Simple policy: just get action
            action_tensor = policy(obs_tensor)

        action = action_tensor.numpy().flatten()

        # Step environment
        next_obs, reward, done, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        obs = next_obs
        if done:
            obs = env.reset()

    return Rollout(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        dones=np.array(dones),
        values=np.array(values) if values else np.array([]),
        log_probs=np.array(log_probs) if log_probs else np.array([]),
    )


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer.

    A minimal but complete PPO implementation for training locomotion policies.

    Args:
        policy: ActorCritic network to train
        lr: Learning rate (default: 3e-4)
        gamma: Discount factor (default: 0.99)
        gae_lambda: GAE lambda (default: 0.95)
        clip_coef: PPO clip coefficient (default: 0.2)
        vf_coef: Value function loss coefficient (default: 0.5)
        ent_coef: Entropy bonus coefficient (default: 0.01)
        max_grad_norm: Maximum gradient norm for clipping (default: 0.5)
        update_epochs: Number of epochs per update (default: 10)
        minibatch_size: Minibatch size for updates (default: 64)

    Example:
        >>> policy = go2_actor_critic()
        >>> trainer = PPOTrainer(policy, lr=3e-4)
        >>> rollout = collect_rollout(env, policy, steps=2048)
        >>> metrics = trainer.update(rollout)
    """

    def __init__(
        self,
        policy: "ActorCritic",
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 64,
    ):
        _check_tinygrad()

        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.optimizer = Adam(get_parameters(policy), lr=lr)
        self.total_updates = 0

    def update(self, rollout: Rollout) -> dict:
        """
        Perform PPO update on collected rollout.

        Args:
            rollout: Collected trajectory data

        Returns:
            Dictionary of training metrics:
                - mean_reward: Mean reward per step
                - total_reward: Total reward in rollout
                - policy_loss: Policy loss
                - value_loss: Value function loss
                - entropy: Policy entropy
                - approx_kl: Approximate KL divergence
        """
        # Compute advantages and returns
        advantages = rollout.compute_advantages(self.gamma, self.gae_lambda)
        returns = rollout.compute_returns(self.gamma)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs = Tensor(rollout.observations.astype(np.float32))
        actions = Tensor(rollout.actions.astype(np.float32))
        old_log_probs = Tensor(rollout.log_probs.astype(np.float32))
        advantages_t = Tensor(advantages.astype(np.float32))
        returns_t = Tensor(returns.astype(np.float32))

        num_samples = rollout.num_steps
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        num_updates = 0

        for _ in range(self.update_epochs):
            # Random permutation for minibatches
            indices = np.random.permutation(num_samples)

            for start in range(0, num_samples, self.minibatch_size):
                end = min(start + self.minibatch_size, num_samples)
                mb_indices = indices[start:end]

                # Get minibatch
                mb_obs = Tensor(rollout.observations[mb_indices].astype(np.float32))
                mb_actions = Tensor(rollout.actions[mb_indices].astype(np.float32))
                mb_old_log_probs = Tensor(rollout.log_probs[mb_indices].astype(np.float32))
                mb_advantages = Tensor(advantages[mb_indices].astype(np.float32))
                mb_returns = Tensor(returns[mb_indices].astype(np.float32))

                # Forward pass
                _, new_log_probs, entropy, values = self.policy.get_action_and_value(
                    mb_obs, mb_actions
                )

                # Policy loss (clipped surrogate objective)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()
                clip_ratio = ratio.clip(1 - self.clip_coef, 1 + self.clip_coef)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * clip_ratio
                policy_loss = pg_loss1.maximum(pg_loss2).mean()

                # Value loss
                value_loss = ((values - mb_returns) ** 2).mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track metrics
                metrics["policy_loss"] += float(policy_loss.numpy())
                metrics["value_loss"] += float(value_loss.numpy())
                metrics["entropy"] += float(entropy.numpy())

                # Approximate KL divergence
                with Tensor.inference_mode():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    metrics["approx_kl"] += float(approx_kl.numpy())

                num_updates += 1

        self.total_updates += num_updates

        # Average metrics
        for key in metrics:
            metrics[key] /= max(num_updates, 1)

        # Add rollout metrics
        metrics["mean_reward"] = rollout.mean_reward
        metrics["total_reward"] = rollout.total_reward
        metrics["num_steps"] = rollout.num_steps

        return metrics


class ReplayBuffer:
    """
    Simple replay buffer for off-policy algorithms.

    Stores transitions and allows random sampling for training.

    Args:
        capacity: Maximum number of transitions to store
        obs_dim: Observation dimension
        act_dim: Action dimension
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """
        Sample a random batch of transitions.

        Returns:
            Dictionary with keys: observations, actions, rewards, next_observations, dones
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size


__all__ = [
    "Rollout",
    "collect_rollout",
    "PPOTrainer",
    "ReplayBuffer",
]
