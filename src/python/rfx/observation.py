"""
rfx.observation - Multi-modal observation handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class ObservationSpec:
    """Specification for observation structure."""

    state_dim: int
    max_state_dim: int = 64
    image_shape: Optional[tuple] = None
    num_cameras: int = 0
    language_dim: Optional[int] = None

    @property
    def has_images(self) -> bool:
        return self.num_cameras > 0 and self.image_shape is not None

    @property
    def has_language(self) -> bool:
        return self.language_dim is not None


def make_observation(
    state: "torch.Tensor",
    state_dim: int,
    max_state_dim: int,
    images: Optional["torch.Tensor"] = None,
    language: Optional["torch.Tensor"] = None,
    device: str = "cpu",
) -> Dict[str, "torch.Tensor"]:
    """Create a properly formatted observation dictionary."""
    import torch

    batch_size = state.shape[0]

    if state.shape[-1] < max_state_dim:
        padding = torch.zeros(
            batch_size, max_state_dim - state.shape[-1], device=device, dtype=state.dtype
        )
        state_padded = torch.cat([state, padding], dim=-1)
    else:
        state_padded = state[:, :max_state_dim]

    obs: Dict[str, torch.Tensor] = {"state": state_padded.to(device)}

    if images is not None:
        obs["images"] = images.to(device)
    if language is not None:
        obs["language"] = language.to(device)

    return obs


def unpad_action(action: "torch.Tensor", action_dim: int) -> "torch.Tensor":
    """Extract actual action from padded tensor."""
    if action.dim() == 2:
        return action[:, :action_dim]
    elif action.dim() == 3:
        return action[:, :, :action_dim]
    else:
        raise ValueError(f"Expected 2D or 3D action tensor, got {action.dim()}D")


@dataclass
class ObservationBuffer:
    """Buffer for storing observation history (for frame stacking)."""

    capacity: int
    _buffer: List[Dict[str, "torch.Tensor"]] = field(default_factory=list)

    def push(self, obs: Dict[str, "torch.Tensor"]) -> None:
        import torch

        obs_copy = {k: v.clone() for k, v in obs.items()}
        self._buffer.append(obs_copy)
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def get_stacked(self) -> Dict[str, "torch.Tensor"]:
        import torch

        if not self._buffer:
            raise ValueError("Buffer is empty")
        result = {}
        for key in self._buffer[0].keys():
            tensors = [obs[key] for obs in self._buffer]
            result[key] = torch.stack(tensors, dim=1)
        return result

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
