"""
rfx.utils.padding - DOF padding for multi-embodiment training
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class PaddingConfig:
    """Configuration for state/action padding."""

    state_dim: int
    action_dim: int
    max_state_dim: int = 64
    max_action_dim: int = 64

    def __post_init__(self):
        if self.max_state_dim < self.state_dim:
            raise ValueError(
                f"max_state_dim ({self.max_state_dim}) must be >= state_dim ({self.state_dim})"
            )
        if self.max_action_dim < self.action_dim:
            raise ValueError(
                f"max_action_dim ({self.max_action_dim}) must be >= action_dim ({self.action_dim})"
            )


def pad_state(state: "torch.Tensor", state_dim: int, max_state_dim: int) -> "torch.Tensor":
    """Pad state tensor to max_state_dim."""
    import torch

    if state.shape[-1] >= max_state_dim:
        return state[..., :max_state_dim]

    pad_size = max_state_dim - state.shape[-1]

    if state.dim() == 2:
        padding = torch.zeros(state.shape[0], pad_size, device=state.device, dtype=state.dtype)
        return torch.cat([state, padding], dim=-1)
    elif state.dim() == 3:
        padding = torch.zeros(
            state.shape[0], state.shape[1], pad_size, device=state.device, dtype=state.dtype
        )
        return torch.cat([state, padding], dim=-1)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {state.dim()}D")


def pad_action(action: "torch.Tensor", action_dim: int, max_action_dim: int) -> "torch.Tensor":
    """Pad action tensor to max_action_dim."""
    import torch

    if action.shape[-1] >= max_action_dim:
        return action[..., :max_action_dim]

    pad_size = max_action_dim - action.shape[-1]

    if action.dim() == 2:
        padding = torch.zeros(action.shape[0], pad_size, device=action.device, dtype=action.dtype)
        return torch.cat([action, padding], dim=-1)
    elif action.dim() == 3:
        padding = torch.zeros(
            action.shape[0], action.shape[1], pad_size, device=action.device, dtype=action.dtype
        )
        return torch.cat([action, padding], dim=-1)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {action.dim()}D")


def unpad_action(action: "torch.Tensor", action_dim: int) -> "torch.Tensor":
    """Extract actual action from padded tensor."""
    return action[..., :action_dim]
