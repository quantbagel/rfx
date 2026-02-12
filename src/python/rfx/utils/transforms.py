"""
rfx.utils.transforms - Observation and action transforms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    import torch


@dataclass
class ObservationNormalizer:
    """Running mean/std normalizer for observations."""

    state_dim: int
    clip: float = 10.0
    eps: float = 1e-8
    _mean: "torch.Tensor" = field(init=False)
    _var: "torch.Tensor" = field(init=False)
    _count: int = field(default=0, init=False)

    def __post_init__(self):
        import torch

        self._mean = torch.zeros(self.state_dim)
        self._var = torch.ones(self.state_dim)

    def update(self, state: "torch.Tensor") -> None:
        batch_mean = state.mean(dim=0)
        batch_var = state.var(dim=0)
        batch_count = state.shape[0]

        delta = batch_mean - self._mean
        total_count = self._count + batch_count

        self._mean = self._mean + delta * batch_count / total_count
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self._count * batch_count / total_count
        self._var = M2 / total_count
        self._count = total_count

    def normalize(self, obs: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        import torch

        result = {}
        if "state" in obs:
            state = obs["state"]
            normalized = (state[..., : self.state_dim] - self._mean) / (
                torch.sqrt(self._var) + self.eps
            )
            normalized = torch.clamp(normalized, -self.clip, self.clip)
            if state.shape[-1] > self.state_dim:
                result["state"] = torch.cat([normalized, state[..., self.state_dim :]], dim=-1)
            else:
                result["state"] = normalized

        if "images" in obs:
            result["images"] = obs["images"].float() / 255.0
        if "language" in obs:
            result["language"] = obs["language"]

        return result


@dataclass
class ActionChunker:
    """Action chunking for temporal abstraction."""

    horizon: int
    action_dim: int
    ensemble_mode: str = "exponential"
    temperature: float = 0.5
    _chunks: List["torch.Tensor"] = field(default_factory=list, init=False)

    def add_chunk(self, chunk: "torch.Tensor") -> None:
        self._chunks.append(chunk)
        if len(self._chunks) > self.horizon:
            self._chunks.pop(0)

    def get_action(self) -> "torch.Tensor":
        import torch

        if not self._chunks:
            raise ValueError("No chunks available")

        if self.ensemble_mode == "first":
            return self._chunks[-1][:, 0, :]

        elif self.ensemble_mode == "average":
            predictions = []
            for i, chunk in enumerate(self._chunks):
                step_idx = len(self._chunks) - 1 - i
                if step_idx < chunk.shape[1]:
                    predictions.append(chunk[:, step_idx, :])
            return torch.stack(predictions).mean(dim=0)

        elif self.ensemble_mode == "exponential":
            predictions, weights = [], []
            for i, chunk in enumerate(self._chunks):
                step_idx = len(self._chunks) - 1 - i
                if step_idx < chunk.shape[1]:
                    predictions.append(chunk[:, step_idx, :])
                    weights.append(torch.exp(torch.tensor(-step_idx * self.temperature)))

            predictions = torch.stack(predictions)
            weights = torch.stack(weights)
            weights = weights / weights.sum()
            return (predictions * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown ensemble mode: {self.ensemble_mode}")

    def reset(self) -> None:
        self._chunks.clear()
