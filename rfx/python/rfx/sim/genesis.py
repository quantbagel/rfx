"""
rfx.sim.genesis - Genesis physics backend (GPU-accelerated)

Requires: pip install genesis-world
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from ..config import RobotConfig
from ..observation import make_observation, unpad_action

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
        self._closed = False

        auto_install = bool(
            kwargs.get(
                "auto_install",
                kwargs.get(
                    "install_if_missing",
                    os.getenv("RFX_AUTO_INSTALL_GENESIS", "0") == "1",
                ),
            )
        )
        self._gs = self._import_genesis(auto_install=auto_install)

        self._max_steps = int(kwargs.get("max_steps", 1000))
        self._step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reward = torch.zeros(num_envs, device=device)

        # Viewer and physics knobs for end-to-end Genesis workflows.
        self._viewer_enabled = bool(
            kwargs.get(
                "viewer",
                kwargs.get(
                    "render",
                    kwargs.get("show_viewer", kwargs.get("visualize", False)),
                ),
            )
        )
        self._dt = float(kwargs.get("dt", 1.0 / config.control_freq_hz))
        self._substeps = int(kwargs.get("substeps", 4))
        self._gravity = tuple(kwargs.get("gravity", (0.0, 0.0, -9.81)))
        self._robot_pos = tuple(kwargs.get("robot_pos", (0.0, 0.0, 0.5)))

        self._init_genesis_runtime(kwargs)
        self._build_scene(kwargs)

    def _import_genesis(self, auto_install: bool):
        try:
            import genesis as gs

            return gs
        except ImportError as exc:
            if auto_install:
                if sys.version_info >= (3, 14):
                    raise RuntimeError(
                        "Genesis auto-install is currently unavailable on CPython 3.14. "
                        "Upstream Genesis dependencies do not publish cp314 wheels yet. "
                        "Use a Python <= 3.13 environment for Genesis until upstream support lands."
                    ) from exc
                self._install_genesis()
                import genesis as gs

                return gs
            raise ImportError(
                "Genesis not installed. Install with: uv pip install genesis-world "
                "(or pass auto_install=True).\n"
                "For testing without Genesis, use backend='mock'"
            ) from exc

    def _install_genesis(self) -> None:
        uv_bin = shutil.which("uv")
        commands = []
        if uv_bin:
            commands.append([uv_bin, "pip", "install", "--python", sys.executable, "genesis-world"])
        if importlib.util.find_spec("pip") is not None:
            commands.append([sys.executable, "-m", "pip", "install", "genesis-world"])

        errors = []
        for cmd in commands:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                return
            tool = "uv pip" if cmd[0] == uv_bin else "pip"
            stderr = (result.stderr or "").strip()
            errors.append(f"{tool}: {stderr}")

        if not commands:
            raise RuntimeError(
                "Automatic Genesis installation failed: no installer available in runtime "
                "(neither `uv` nor `pip` was found). Try manually in your shell:\n"
                "uv pip install genesis-world"
            )

        joined_errors = "\n".join(errors)
        if "cp314" in joined_errors:
            raise RuntimeError(
                "Automatic Genesis installation failed: Genesis dependencies do not currently "
                "provide CPython 3.14 wheels.\n"
                "Use a Python <= 3.13 environment for Genesis until upstream support lands."
            )

        raise RuntimeError(
            "Automatic Genesis installation failed.\n"
            "Try manually: uv pip install genesis-world\n" + joined_errors
        )

    def _init_genesis_runtime(self, kwargs: dict[str, Any]) -> None:
        backend_override = kwargs.get("genesis_backend")
        if backend_override is not None:
            backend = backend_override
        else:
            backend = getattr(self._gs, "cuda", None) if self.device == "cuda" else None
            if backend is None:
                backend = getattr(self._gs, "gpu", None) if self.device == "cuda" else None
            if backend is None:
                backend = getattr(self._gs, "cpu", None)

        init_kwargs: dict[str, Any] = {}
        if backend is not None:
            init_kwargs["backend"] = backend

        try:
            self._gs.init(**init_kwargs)
        except Exception as exc:
            # Some Genesis versions only allow a single global init.
            if "already" not in str(exc).lower():
                raise

    def _build_scene(self, kwargs: dict[str, Any]) -> None:
        sim_options = self._build_sim_options()

        scene_kwargs: dict[str, Any] = {"sim_options": sim_options}
        scene_sig = self._safe_signature(self._gs.Scene)
        if scene_sig is not None and self._viewer_enabled:
            if "show_viewer" in scene_sig.parameters:
                scene_kwargs["show_viewer"] = True
            elif "viewer" in scene_sig.parameters:
                scene_kwargs["viewer"] = True
            elif "headless" in scene_sig.parameters:
                scene_kwargs["headless"] = False

        self._scene = self._gs.Scene(**scene_kwargs)
        self._scene.add_entity(self._gs.morphs.Plane())
        self._robot = self._scene.add_entity(self._build_robot_morph(kwargs))
        self._scene.build(n_envs=self.num_envs)
        self._control_dofs_idx = self._infer_control_dofs_idx()
        self._dofs_idx_kwarg = self._infer_dofs_idx_kwarg()

    def _build_sim_options(self) -> Any:
        options_ctor = self._gs.options.SimOptions
        sig = self._safe_signature(options_ctor)
        kwargs: dict[str, Any] = {}
        if sig is not None:
            if "dt" in sig.parameters:
                kwargs["dt"] = self._dt
            if "substeps" in sig.parameters:
                kwargs["substeps"] = self._substeps
            if "gravity" in sig.parameters:
                kwargs["gravity"] = self._gravity
        try:
            return options_ctor(**kwargs)
        except TypeError:
            return options_ctor(dt=self._dt, substeps=self._substeps)

    def _build_robot_morph(self, kwargs: dict[str, Any]) -> Any:
        urdf_path = kwargs.get("urdf_path", self.config.urdf_path)
        if urdf_path:
            resolved = self._resolve_asset_path(str(urdf_path))
            return self._gs.morphs.URDF(file=resolved, pos=self._robot_pos)
        return self._gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=self._robot_pos)

    def _resolve_asset_path(self, path: str) -> str:
        candidate = Path(path).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)

        search_roots = [Path.cwd(), Path(__file__).resolve().parents[3]]
        for root in search_roots:
            full = root / candidate
            if full.exists():
                return str(full)
        return str(candidate)

    def _safe_signature(self, fn: Any) -> inspect.Signature | None:
        try:
            return inspect.signature(fn)
        except (TypeError, ValueError):
            return None

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.as_tensor(value, device=self.device)

    def _get_robot_dof_tensor(self, method_names: tuple[str, ...]) -> torch.Tensor:
        for name in method_names:
            method = getattr(self._robot, name, None)
            if callable(method):
                return self._to_tensor(method())
        raise AttributeError(f"Genesis robot entity missing methods: {method_names}")

    def _normalize_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        return tensor

    def _infer_control_dofs_idx(self) -> list[int] | None:
        n_dofs = int(
            self._normalize_batch(self._get_robot_dof_tensor(("get_dofs_position",))).shape[-1]
        )
        if n_dofs <= self.config.action_dim:
            return None
        # Common floating-base convention: actuated joints are the last action_dim DOFs.
        start = max(0, n_dofs - self.config.action_dim)
        return list(range(start, n_dofs))

    def _infer_dofs_idx_kwarg(self) -> str:
        for method_name in ("control_dofs_position", "set_dofs_position"):
            method = getattr(self._robot, method_name, None)
            sig = self._safe_signature(method) if callable(method) else None
            if sig is None:
                continue
            if "dofs_idx_local" in sig.parameters:
                return "dofs_idx_local"
            if "dofs_idx" in sig.parameters:
                return "dofs_idx"
        return "dofs_idx_local"

    def observe(self) -> dict[str, torch.Tensor]:
        positions = self._normalize_batch(
            self._get_robot_dof_tensor(("get_dofs_position", "get_dof_positions"))
        )
        velocities = self._normalize_batch(
            self._get_robot_dof_tensor(("get_dofs_velocity", "get_dof_velocities"))
        )
        state = torch.cat([positions, velocities], dim=-1)
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
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim == 3:
            action = action[:, -1, :]
        action = unpad_action(action, self.config.action_dim)
        if self._control_dofs_idx is not None:
            kwargs = {self._dofs_idx_kwarg: self._control_dofs_idx}
            self._robot.control_dofs_position(action, **kwargs)
        else:
            self._robot.control_dofs_position(action)
        self._scene.step()

        if self._viewer_enabled:
            self.render()

        self._step_count += 1
        self._done = self._step_count >= self._max_steps
        self._reward.zero_()

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif env_ids.dtype != torch.long:
            env_ids = env_ids.to(dtype=torch.long)
        ids_list = [int(x) for x in env_ids.tolist()]

        dof_dim = self.config.action_dim
        if self._control_dofs_idx is not None:
            dof_dim = len(self._control_dofs_idx)
        zero_pos = torch.zeros(len(ids_list), dof_dim, device=self.device)
        zero_vel = torch.zeros(len(ids_list), dof_dim, device=self.device)

        if self._control_dofs_idx is not None:
            dof_kwargs = {self._dofs_idx_kwarg: self._control_dofs_idx}
            self._robot.set_dofs_position(
                zero_pos,
                **dof_kwargs,
                envs_idx=ids_list,
            )
            self._robot.set_dofs_velocity(
                zero_vel,
                **dof_kwargs,
                envs_idx=ids_list,
            )
        else:
            self._robot.set_dofs_position(zero_pos, envs_idx=ids_list)
            self._robot.set_dofs_velocity(zero_vel, envs_idx=ids_list)

        self._step_count[env_ids] = 0
        self._done[env_ids] = False
        self._reward[env_ids] = 0.0
        return self.observe()

    def render(self) -> None:
        render_fn = getattr(self._scene, "render", None)
        if callable(render_fn):
            render_fn()

    def close(self) -> None:
        if self._closed:
            return
        close_fn = getattr(self._scene, "close", None)
        if callable(close_fn):
            close_fn()
        self._closed = True

    def get_reward(self) -> torch.Tensor:
        return self._reward

    def get_done(self) -> torch.Tensor:
        return self._done
