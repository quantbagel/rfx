"""
rfx.sim.mock - Mock simulation backend for testing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from ..config import RobotConfig
from ..observation import make_observation
from ..robot import RobotBase

if TYPE_CHECKING:
    pass


class MockBackend:
    """Mock physics backend - pure PyTorch, no dependencies."""

    def __init__(
        self,
        config: RobotConfig,
        num_envs: int = 1,
        device: str = "cpu",
        **kwargs,
    ):
        self.config = config
        self.num_envs = num_envs
        self.device = device

        self._positions = torch.zeros(num_envs, config.action_dim, device=device)
        self._velocities = torch.zeros(num_envs, config.action_dim, device=device)
        self._dt = 1.0 / config.control_freq_hz
        self._step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._max_steps = kwargs.get("max_steps", 1000)
        self._done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reward = torch.zeros(num_envs, device=device)

    def observe(self) -> dict[str, torch.Tensor]:
        state = torch.cat([self._positions, self._velocities], dim=-1)
        return cast(
            dict[str, torch.Tensor],
            make_observation(
                state=state,
                state_dim=self.config.state_dim,
                max_state_dim=self.config.max_state_dim,
                device=self.device,
            ),
        )

    def act(self, action: torch.Tensor) -> None:
        action = action[:, : self.config.action_dim]
        k, d = 100.0, 10.0
        error = action - self._positions
        acceleration = k * error - d * self._velocities
        self._velocities = self._velocities + acceleration * self._dt
        self._positions = self._positions + self._velocities * self._dt
        self._positions = torch.clamp(self._positions, -3.14159, 3.14159)
        self._velocities = torch.clamp(self._velocities, -10.0, 10.0)
        self._step_count += 1
        self._done = self._step_count >= self._max_steps
        self._reward = -torch.norm(error, dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._positions[env_ids] = 0.0
        self._velocities[env_ids] = 0.0
        self._step_count[env_ids] = 0
        self._done[env_ids] = False
        self._reward[env_ids] = 0.0
        return self.observe()

    def get_reward(self) -> torch.Tensor:
        return self._reward

    def get_done(self) -> torch.Tensor:
        return self._done


class MockRobot(RobotBase):
    """Mock robot for testing."""

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 6,
        num_envs: int = 1,
        max_state_dim: int = 64,
        max_action_dim: int = 64,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(state_dim, action_dim, num_envs, max_state_dim, max_action_dim, device)
        config = RobotConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
        )
        self._backend = MockBackend(config, num_envs=num_envs, device=device, **kwargs)

    def observe(self) -> dict[str, torch.Tensor]:
        return self._backend.observe()

    def act(self, action: torch.Tensor) -> None:
        self._backend.act(action)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self._backend.reset(env_ids)

    def get_reward(self) -> torch.Tensor:
        return self._backend.get_reward()

    def get_done(self) -> torch.Tensor:
        return self._backend.get_done()
