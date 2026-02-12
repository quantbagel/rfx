"""
Neural network primitives using tinygrad

A simple, transparent implementation of policies using tinygrad tensors.
Follows the tinygrad philosophy: simple to start, powerful enough for production.

Example:
    >>> from rfx.nn import MLP, go2_mlp
    >>> policy = go2_mlp()
    >>> obs = Tensor.randn(1, 48)
    >>> actions = policy(obs)  # JIT compiled on second call
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from tinygrad import Tensor, dtypes
    from tinygrad.nn import Linear
    from tinygrad.nn.state import (
        get_parameters,
        get_state_dict,
        load_state_dict,
        safe_save,
        safe_load,
    )
    from tinygrad.engine.jit import TinyJit

    TINYGRAD_AVAILABLE = True
except ImportError:
    TINYGRAD_AVAILABLE = False
    Tensor = Any
    TinyJit = lambda x: x  # no-op decorator


def _check_tinygrad():
    if not TINYGRAD_AVAILABLE:
        raise ImportError(
            "tinygrad is required for neural network support. Install with: pip install tinygrad"
        )


class Policy:
    """
    Base policy class for neural network policies.

    Users can subclass this to create custom architectures.
    The forward method should take observations and return actions.

    Example:
        >>> class CustomPolicy(Policy):
        ...     def __init__(self):
        ...         self.l1 = Linear(48, 256)
        ...         self.l2 = Linear(256, 12)
        ...
        ...     def forward(self, obs: Tensor) -> Tensor:
        ...         x = self.l1(obs).tanh()
        ...         return self.l2(x).tanh()
    """

    def forward(self, obs: Tensor) -> Tensor:
        """
        Forward pass: observations -> actions.

        Args:
            obs: Observation tensor of shape (batch, obs_dim)

        Returns:
            Action tensor of shape (batch, act_dim)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, obs: Tensor) -> Tensor:
        """Run inference (JIT compiled after first call)."""
        return self.forward(obs)

    def parameters(self) -> list:
        """Get all trainable parameters."""
        _check_tinygrad()
        return get_parameters(self)

    def save(self, path: str | Path) -> None:
        """
        Save policy weights to a safetensors file.

        Args:
            path: Path to save the weights (should end in .safetensors)
        """
        _check_tinygrad()
        state = get_state_dict(self)
        safe_save(state, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        """
        Load policy weights from a safetensors file.

        Note: This creates a new instance and loads weights into it.
        The subclass must have a no-argument constructor.

        Args:
            path: Path to the saved weights

        Returns:
            Policy instance with loaded weights
        """
        _check_tinygrad()
        policy = cls()
        state = safe_load(str(path))
        load_state_dict(policy, state)
        return policy

    def to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert a tinygrad tensor to numpy array."""
        return tensor.numpy()


class MLP(Policy):
    """
    Multi-layer perceptron policy.

    A simple feedforward network with tanh activations.
    Suitable for most locomotion tasks.

    Args:
        obs_dim: Observation dimension
        act_dim: Action dimension
        hidden: List of hidden layer sizes (default: [256, 256])

    Example:
        >>> policy = MLP(obs_dim=48, act_dim=12, hidden=[256, 256])
        >>> obs = Tensor.randn(1, 48)
        >>> actions = policy(obs)
        >>> print(actions.shape)  # (1, 12)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: list[int] | None = None,
    ):
        _check_tinygrad()

        if hidden is None:
            hidden = [256, 256]

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        # Build layers
        dims = [obs_dim] + hidden + [act_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))

    def forward(self, obs: Tensor) -> Tensor:
        """Forward pass with tanh activations."""
        x = obs
        for layer in self.layers[:-1]:
            x = layer(x).tanh()
        # Final layer also uses tanh to bound actions to [-1, 1]
        return self.layers[-1](x).tanh()

    def __repr__(self) -> str:
        return f"MLP(obs_dim={self.obs_dim}, act_dim={self.act_dim}, hidden={self.hidden})"


class JitPolicy(Policy):
    """
    A policy wrapper that enables TinyJit compilation.

    Wraps any policy and JIT compiles its forward pass for faster inference.
    The first call traces the computation graph, subsequent calls are fast.

    Args:
        policy: The policy to wrap

    Example:
        >>> mlp = MLP(48, 12)
        >>> jit_policy = JitPolicy(mlp)
        >>> # First call: traces graph
        >>> actions = jit_policy(obs)
        >>> # Second call: runs compiled kernel
        >>> actions = jit_policy(obs)
    """

    def __init__(self, policy: Policy):
        _check_tinygrad()
        self._policy = policy
        self._jit_forward = TinyJit(policy.forward)

    def forward(self, obs: Tensor) -> Tensor:
        return self._jit_forward(obs)

    def __repr__(self) -> str:
        return f"JitPolicy({self._policy!r})"


class ActorCritic(Policy):
    """
    Actor-critic network for PPO training.

    Shares a backbone between actor (policy) and critic (value function).

    Args:
        obs_dim: Observation dimension
        act_dim: Action dimension
        hidden: Hidden layer sizes for shared backbone

    Example:
        >>> ac = ActorCritic(48, 12)
        >>> obs = Tensor.randn(32, 48)
        >>> actions, values = ac.forward_actor_critic(obs)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: list[int] | None = None,
    ):
        _check_tinygrad()

        if hidden is None:
            hidden = [256, 256]

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Shared backbone
        self.backbone = []
        dims = [obs_dim] + hidden
        for i in range(len(dims) - 1):
            self.backbone.append(Linear(dims[i], dims[i + 1]))

        # Actor head (outputs action mean)
        self.actor_head = Linear(hidden[-1], act_dim)

        # Critic head (outputs value)
        self.critic_head = Linear(hidden[-1], 1)

        # Learnable log std for action distribution
        self.log_std = Tensor.zeros(act_dim)

    def _backbone_forward(self, obs: Tensor) -> Tensor:
        """Forward through shared backbone."""
        x = obs
        for layer in self.backbone:
            x = layer(x).tanh()
        return x

    def forward(self, obs: Tensor) -> Tensor:
        """Get action mean (for inference)."""
        features = self._backbone_forward(obs)
        return self.actor_head(features).tanh()

    def forward_actor_critic(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Get both actions and values (for training).

        Returns:
            Tuple of (action_mean, value)
        """
        features = self._backbone_forward(obs)
        action_mean = self.actor_head(features).tanh()
        value = self.critic_head(features)
        return action_mean, value

    def get_action_and_value(
        self, obs: Tensor, action: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample action and compute log prob + entropy (for PPO update).

        Args:
            obs: Observations
            action: Optional pre-sampled action (for computing log prob)

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        features = self._backbone_forward(obs)
        action_mean = self.actor_head(features).tanh()
        value = self.critic_head(features)

        # Gaussian action distribution
        std = self.log_std.exp()

        if action is None:
            # Sample from Gaussian
            noise = Tensor.randn(*action_mean.shape)
            action = (action_mean + noise * std).tanh()

        # Log probability (simplified, ignoring tanh correction)
        log_prob = (
            -0.5 * ((action - action_mean) / std).pow(2) - self.log_std - 0.5 * np.log(2 * np.pi)
        ).sum(axis=-1)

        # Entropy
        entropy = (0.5 + 0.5 * np.log(2 * np.pi) + self.log_std).sum()

        return action, log_prob, entropy, value.squeeze(-1)


# Convenience constructors for Go2 robot
def go2_mlp(hidden: list[int] | None = None) -> MLP:
    """
    Create an MLP policy sized for the Go2 robot.

    Go2 observation space: 48 dimensions
    Go2 action space: 12 dimensions (joint positions)

    Args:
        hidden: Hidden layer sizes (default: [256, 256])

    Returns:
        MLP policy for Go2
    """
    if hidden is None:
        hidden = [256, 256]
    return MLP(obs_dim=48, act_dim=12, hidden=hidden)


def go2_actor_critic(hidden: list[int] | None = None) -> ActorCritic:
    """
    Create an ActorCritic network sized for the Go2 robot.

    Args:
        hidden: Hidden layer sizes (default: [256, 256])

    Returns:
        ActorCritic network for Go2
    """
    if hidden is None:
        hidden = [256, 256]
    return ActorCritic(obs_dim=48, act_dim=12, hidden=hidden)


__all__ = [
    "Policy",
    "MLP",
    "JitPolicy",
    "ActorCritic",
    "go2_mlp",
    "go2_actor_critic",
    "TINYGRAD_AVAILABLE",
]
