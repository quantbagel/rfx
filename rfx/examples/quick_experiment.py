#!/usr/bin/env python3
"""
Quick experimentation examples with pi + tinygrad

This file shows how easy it is to try new ideas:
- Custom network architectures
- Different activation functions
- Attention mechanisms
- Recurrent policies

The tinygrad-style approach makes experimentation fast and transparent.

Usage:
    python examples/quick_experiment.py
"""

import numpy as np

import rfx
from rfx.nn import Policy, MLP, go2_mlp

# Check if tinygrad is available
try:
    from tinygrad import Tensor
    from tinygrad.nn import Linear

    TINYGRAD_AVAILABLE = True
except ImportError:
    TINYGRAD_AVAILABLE = False
    print("tinygrad not installed. Install with: pip install tinygrad")
    print("Showing code examples only.\n")


def experiment_1_bigger_network():
    """Experiment: What if we use a bigger network?"""
    print("Experiment 1: Bigger Network")
    print("-" * 40)

    # Standard policy
    standard = go2_mlp(hidden=[256, 256])
    print(f"Standard: {standard}")

    # Bigger policy
    bigger = MLP(48, 12, hidden=[512, 512, 512])
    print(f"Bigger:   {bigger}")

    if TINYGRAD_AVAILABLE:
        # Compare parameter counts
        standard_params = sum(p.numel() for p in standard.parameters())
        bigger_params = sum(p.numel() for p in bigger.parameters())
        print(f"\nParameter count:")
        print(f"  Standard: {standard_params:,}")
        print(f"  Bigger:   {bigger_params:,}")
        print(f"  Ratio:    {bigger_params / standard_params:.1f}x")

    print()


def experiment_2_custom_activations():
    """Experiment: What if we use different activations?"""
    print("Experiment 2: Custom Activations")
    print("-" * 40)

    if not TINYGRAD_AVAILABLE:
        print("(Requires tinygrad)")
        print()
        return

    class GELUPolicy(Policy):
        """Policy with GELU activations instead of tanh."""

        def __init__(self, obs_dim: int = 48, act_dim: int = 12):
            self.l1 = Linear(obs_dim, 256)
            self.l2 = Linear(256, 256)
            self.l3 = Linear(256, act_dim)

        def forward(self, obs: Tensor) -> Tensor:
            x = self.l1(obs).gelu()
            x = self.l2(x).gelu()
            return self.l3(x).tanh()  # Still bound output

    policy = GELUPolicy()
    print(f"GELU Policy: {policy}")

    # Test inference
    obs = Tensor.randn(1, 48)
    action = policy(obs)
    print(f"Input shape:  {obs.shape}")
    print(f"Output shape: {action.shape}")
    print(f"Output range: [{action.min().numpy():.3f}, {action.max().numpy():.3f}]")
    print()


def experiment_3_residual_connections():
    """Experiment: What if we add residual connections?"""
    print("Experiment 3: Residual Connections")
    print("-" * 40)

    if not TINYGRAD_AVAILABLE:
        print("(Requires tinygrad)")
        print()
        return

    class ResidualPolicy(Policy):
        """Policy with residual connections for gradient flow."""

        def __init__(self, obs_dim: int = 48, act_dim: int = 12, hidden: int = 256):
            # Project obs to hidden dim
            self.proj = Linear(obs_dim, hidden)

            # Residual blocks
            self.block1_l1 = Linear(hidden, hidden)
            self.block1_l2 = Linear(hidden, hidden)

            self.block2_l1 = Linear(hidden, hidden)
            self.block2_l2 = Linear(hidden, hidden)

            # Output projection
            self.out = Linear(hidden, act_dim)

        def _residual_block(self, x: Tensor, l1: Linear, l2: Linear) -> Tensor:
            """Residual block: x + f(x)"""
            residual = x
            x = l1(x).relu()
            x = l2(x)
            return (x + residual).relu()

        def forward(self, obs: Tensor) -> Tensor:
            x = self.proj(obs).relu()
            x = self._residual_block(x, self.block1_l1, self.block1_l2)
            x = self._residual_block(x, self.block2_l1, self.block2_l2)
            return self.out(x).tanh()

    policy = ResidualPolicy()
    print(f"Residual Policy with 2 residual blocks")

    # Test inference
    obs = Tensor.randn(1, 48)
    action = policy(obs)
    print(f"Output shape: {action.shape}")
    print()


def experiment_4_attention():
    """Experiment: What if we add self-attention?"""
    print("Experiment 4: Self-Attention")
    print("-" * 40)

    if not TINYGRAD_AVAILABLE:
        print("(Requires tinygrad)")
        print()
        return

    class AttentionPolicy(Policy):
        """
        Policy with self-attention over observation groups.

        Treats the 48-dim observation as 12 groups of 4 features each,
        then applies self-attention between groups.
        """

        def __init__(self):
            self.num_groups = 12
            self.group_dim = 4
            self.hidden_dim = 64

            # Project each group to hidden dim
            self.group_proj = Linear(self.group_dim, self.hidden_dim)

            # Attention: Q, K, V projections
            self.q_proj = Linear(self.hidden_dim, self.hidden_dim)
            self.k_proj = Linear(self.hidden_dim, self.hidden_dim)
            self.v_proj = Linear(self.hidden_dim, self.hidden_dim)

            # Output MLP
            self.out_l1 = Linear(self.hidden_dim * self.num_groups, 256)
            self.out_l2 = Linear(256, 12)

        def forward(self, obs: Tensor) -> Tensor:
            batch_size = obs.shape[0]

            # Reshape to groups: (batch, 12, 4)
            x = obs.reshape(batch_size, self.num_groups, self.group_dim)

            # Project groups: (batch, 12, hidden)
            x = self.group_proj(x)

            # Self-attention
            q = self.q_proj(x)  # (batch, 12, hidden)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Attention scores: (batch, 12, 12)
            scale = self.hidden_dim**0.5
            scores = (q @ k.transpose(-2, -1)) / scale
            attn = scores.softmax(axis=-1)

            # Apply attention
            x = attn @ v  # (batch, 12, hidden)

            # Flatten and output
            x = x.reshape(batch_size, -1)  # (batch, 12*hidden)
            x = self.out_l1(x).relu()
            return self.out_l2(x).tanh()

    policy = AttentionPolicy()
    print(f"Attention Policy (12 groups x 4 features)")

    # Test inference
    obs = Tensor.randn(1, 48)
    action = policy(obs)
    print(f"Output shape: {action.shape}")
    print()


def experiment_5_jit_comparison():
    """Experiment: JIT compilation speedup"""
    print("Experiment 5: JIT Compilation Speedup")
    print("-" * 40)

    if not TINYGRAD_AVAILABLE:
        print("(Requires tinygrad)")
        print()
        return

    import time
    from rfx.nn import JitPolicy

    policy = go2_mlp()
    jit_policy = JitPolicy(policy)

    # Warmup
    obs = Tensor.randn(1, 48)
    _ = policy(obs)
    _ = jit_policy(obs)

    # Benchmark non-JIT
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = policy(obs)
    non_jit_time = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark JIT (first call compiles)
    _ = jit_policy(obs)

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = jit_policy(obs)
    jit_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"Non-JIT: {non_jit_time:.3f} ms/inference")
    print(f"JIT:     {jit_time:.3f} ms/inference")
    print(f"Speedup: {non_jit_time / jit_time:.1f}x")
    print()


def main():
    print("Pi Quick Experiments")
    print("=" * 50)
    print("These examples show how easy it is to experiment")
    print("with different architectures using tinygrad.\n")

    experiment_1_bigger_network()
    experiment_2_custom_activations()
    experiment_3_residual_connections()
    experiment_4_attention()
    experiment_5_jit_comparison()

    print("=" * 50)
    print("All experiments complete!")
    print("\nThe tinygrad approach makes it trivial to:")
    print("  - Try new architectures (just write Python)")
    print("  - Debug with print statements")
    print("  - Deploy with @TinyJit for speed")


if __name__ == "__main__":
    main()
