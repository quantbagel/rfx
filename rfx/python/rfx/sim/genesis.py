"""
rfx.sim.genesis - Genesis physics backend (GPU-accelerated)

Requires: pip install genesis-world
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from ..config import RobotConfig
from ..observation import make_observation

if TYPE_CHECKING:
    pass


class GenesisBackend:
    """Genesis physics backend for GPU-accelerated simulation."""

    def __init__(
        self,
        config: RobotConfig,
        num_envs: int = 1,
        device: str = "cuda",
        **kwargs,
    ):
        self.config = config
        self.num_envs = num_envs
        self.device = device

        try:
            import genesis as gs

            self._gs = gs
        except ImportError:
            raise ImportError(
                "Genesis not installed. Install with: pip install genesis-world\n"
                "For testing without Genesis, use backend='mock'"
            )

        gs.init(backend=gs.cuda if device == "cuda" else gs.cpu)

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1.0 / config.control_freq_hz,
                substeps=kwargs.get("substeps", 4),
            ),
        )

        self._scene.add_entity(gs.morphs.Plane())

        if config.urdf_path:
            self._robot = self._scene.add_entity(
                gs.morphs.URDF(file=config.urdf_path, pos=(0, 0, 0.5)),
            )
        else:
            self._robot = self._scene.add_entity(
                self._gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)),
            )

        self._scene.build(n_envs=num_envs)

        self._step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._max_steps = kwargs.get("max_steps", 1000)
        self._done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reward = torch.zeros(num_envs, device=device)

    def observe(self) -> Dict[str, torch.Tensor]:
        positions = self._robot.get_dofs_position()
        velocities = self._robot.get_dofs_velocity()
        state = torch.cat([positions, velocities], dim=-1)
        return make_observation(
            state=state,
            state_dim=self.config.state_dim,
            max_state_dim=self.config.max_state_dim,
            device=self.device,
        )

    def act(self, action: torch.Tensor) -> None:
        action = action[:, : self.config.action_dim]
        self._robot.control_dofs_position(action)
        self._scene.step()
        self._step_count += 1
        self._done = self._step_count >= self._max_steps
        self._reward = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._robot.set_dofs_position(
            torch.zeros(len(env_ids), self.config.action_dim, device=self.device), env_ids
        )
        self._robot.set_dofs_velocity(
            torch.zeros(len(env_ids), self.config.action_dim, device=self.device), env_ids
        )
        self._step_count[env_ids] = 0
        self._done[env_ids] = False
        self._reward[env_ids] = 0.0
        return self.observe()

    def get_reward(self) -> torch.Tensor:
        return self._reward

    def get_done(self) -> torch.Tensor:
        return self._done
