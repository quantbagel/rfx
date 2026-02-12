"""
rfx.sim.mjx - MuJoCo MJX physics backend (JAX-based)

Requires: pip install mujoco mjx jax[cuda]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from ..config import RobotConfig
from ..observation import make_observation

if TYPE_CHECKING:
    pass


class MjxBackend:
    """MuJoCo MJX backend for JAX-accelerated simulation."""

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
            import mujoco
            from mujoco import mjx
            import jax
            import jax.numpy as jnp

            self._mujoco = mujoco
            self._mjx = mjx
            self._jax = jax
            self._jnp = jnp
        except ImportError:
            raise ImportError(
                "MuJoCo MJX not installed. Install with: pip install mujoco mjx jax[cuda]\n"
                "For testing without MJX, use backend='mock'"
            )

        if config.urdf_path:
            xml_path = config.urdf_path.replace(".urdf", ".xml")
            self._model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            self._model = self._create_default_model()

        self._mjx_model = mjx.put_model(self._model)
        self._data = mujoco.MjData(self._model)
        self._mjx_data = mjx.put_data(self._model, self._data)
        self._batched_data = jax.vmap(lambda _: self._mjx_data)(jnp.arange(num_envs))
        self._step_fn = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

        self._step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._max_steps = kwargs.get("max_steps", 1000)
        self._done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reward = torch.zeros(num_envs, device=device)

    def _create_default_model(self):
        xml = """
        <mujoco>
            <worldbody>
                <light pos="0 0 3"/>
                <geom type="plane" size="10 10 0.1"/>
                <body pos="0 0 0.5">
                    <joint type="slide" axis="1 0 0"/>
                    <joint type="slide" axis="0 1 0"/>
                    <joint type="slide" axis="0 0 1"/>
                    <joint type="hinge" axis="1 0 0"/>
                    <joint type="hinge" axis="0 1 0"/>
                    <joint type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        return self._mujoco.MjModel.from_xml_string(xml)

    def _jax_to_torch(self, arr) -> torch.Tensor:
        import numpy as np

        return torch.from_numpy(np.asarray(arr)).to(self.device)

    def _torch_to_jax(self, tensor: torch.Tensor):
        return self._jnp.array(tensor.cpu().numpy())

    def observe(self) -> Dict[str, torch.Tensor]:
        qpos = self._jax_to_torch(self._batched_data.qpos)
        qvel = self._jax_to_torch(self._batched_data.qvel)
        state = torch.cat([qpos, qvel], dim=-1)
        return make_observation(
            state=state,
            state_dim=self.config.state_dim,
            max_state_dim=self.config.max_state_dim,
            device=self.device,
        )

    def act(self, action: torch.Tensor) -> None:
        action = action[:, : self.config.action_dim]
        ctrl = self._torch_to_jax(action)
        self._batched_data = self._batched_data.replace(ctrl=ctrl)
        self._batched_data = self._step_fn(self._mjx_model, self._batched_data)
        self._step_count += 1
        self._done = self._step_count >= self._max_steps
        self._reward = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        fresh_data = self._mjx.put_data(self._model, self._data)
        for idx in env_ids.tolist():
            self._batched_data = self._jax.tree_map(
                lambda batched, fresh: batched.at[idx].set(fresh),
                self._batched_data,
                fresh_data,
            )
        self._step_count[env_ids] = 0
        self._done[env_ids] = False
        self._reward[env_ids] = 0.0
        return self.observe()

    def get_reward(self) -> torch.Tensor:
        return self._reward

    def get_done(self) -> torch.Tensor:
        return self._done
