"""
rfx.real.go2 - Unitree Go2 hardware backend
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Dict, Optional

import torch

from ..config import RobotConfig, GO2_CONFIG
from ..observation import make_observation

if TYPE_CHECKING:
    pass


class Go2Backend:
    """Unitree Go2 hardware backend using Rust DDS driver."""

    _channel_initialized = False
    _system_python = "/usr/bin/python3"

    def __init__(
        self,
        config: RobotConfig,
        ip_address: str = "192.168.123.161",
        edu_mode: bool = False,
        **kwargs,
    ):
        self.config = config
        self.ip_address = ip_address
        self.edu_mode = edu_mode
        self._backend_mode = "rust"
        self._robot = None
        self._sport_client = None
        self._state_sub = None
        self._latest_lowstate = None
        self._state_lock = threading.Lock()

        backend_pref = os.getenv("RFX_GO2_BACKEND", "auto").strip().lower()
        if backend_pref not in {"auto", "rust", "unitree", "unitree_sdk2py"}:
            backend_pref = "auto"

        if not self.edu_mode and backend_pref in {"auto", "unitree", "unitree_sdk2py"}:
            if self._init_unitree_sdk_backend():
                self._backend_mode = "unitree_sdk2py"
                return
            if self._init_unitree_subprocess_backend():
                self._backend_mode = "unitree_subprocess"
                return

        try:
            from rfx._rfx import Go2, Go2Config

            self._Go2 = Go2
            self._Go2Config = Go2Config
        except ImportError:
            raise ImportError("rfx Rust bindings not available. Build with: maturin develop")

        rust_config = Go2Config(ip_address)
        if edu_mode:
            rust_config = rust_config.with_edu_mode()

        self._robot = Go2.connect(rust_config)

    def _init_unitree_sdk_backend(self) -> bool:
        pet_go_path = "/unitree/module/pet_go"
        if pet_go_path not in sys.path:
            sys.path.insert(0, pet_go_path)

        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        except Exception:
            return False

        try:
            if not Go2Backend._channel_initialized:
                ChannelFactoryInitialize(0)
                Go2Backend._channel_initialized = True

            client = SportClient()
            client.SetTimeout(5.0)
            client.Init()

            def _on_state(msg):
                with self._state_lock:
                    self._latest_lowstate = msg

            sub = ChannelSubscriber("rt/lowstate", LowState_)
            sub.Init(_on_state, 10)

            self._sport_client = client
            self._state_sub = sub
            return True
        except Exception:
            return False

    def _check_rc(self, rc: int, command: str) -> None:
        if rc != 0:
            raise RuntimeError(f"Go2 command '{command}' failed with code {rc}")

    def _init_unitree_subprocess_backend(self) -> bool:
        check = (
            "import sys; "
            'sys.path.insert(0, "/unitree/module/pet_go"); '
            "from unitree_sdk2py.go2.sport.sport_client import SportClient"
        )
        try:
            p = subprocess.run(
                [self._system_python, "-c", check],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3.0,
                check=False,
            )
            return p.returncode == 0
        except Exception:
            return False

    def _run_unitree_cmd(self, command: str, *args: float) -> int:
        if command == "Move":
            cmd_expr = f"c.Move({args[0]}, {args[1]}, {args[2]})"
        elif command in {"Sit", "RiseSit", "RecoveryStand", "StopMove", "GetServerApiVersion"}:
            cmd_expr = f"c.{command}()"
        else:
            raise RuntimeError(f"Unsupported command for unitree subprocess backend: {command}")

        if command == "GetServerApiVersion":
            py = (
                "import sys; "
                'sys.path.insert(0, "/unitree/module/pet_go"); '
                "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; "
                "from unitree_sdk2py.go2.sport.sport_client import SportClient; "
                "ChannelFactoryInitialize(0); "
                "c=SportClient(); c.SetTimeout(5.0); c.Init(); "
                "rc,_=c.GetServerApiVersion(); "
                "print(rc)"
            )
        else:
            py = (
                "import sys; "
                'sys.path.insert(0, "/unitree/module/pet_go"); '
                "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; "
                "from unitree_sdk2py.go2.sport.sport_client import SportClient; "
                "ChannelFactoryInitialize(0); "
                "c=SportClient(); c.SetTimeout(5.0); c.Init(); "
                f"rc={cmd_expr}; "
                "print(rc)"
            )

        p = subprocess.run(
            [self._system_python, "-c", py],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=6.0,
            check=False,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"unitree subprocess command failed (rc={p.returncode}): {p.stderr.strip()}"
            )
        out = p.stdout.strip().splitlines()
        if not out:
            return -1
        return int(out[-1])

    def is_connected(self) -> bool:
        if self._backend_mode == "unitree_sdk2py":
            if self._sport_client is None:
                return False
            code, _ = self._sport_client.GetServerApiVersion()
            return code == 0
        if self._backend_mode == "unitree_subprocess":
            return self._run_unitree_cmd("GetServerApiVersion") == 0
        return self._robot.is_connected()

    def observe(self) -> Dict[str, torch.Tensor]:
        if self._backend_mode in {"unitree_sdk2py", "unitree_subprocess"}:
            with self._state_lock:
                low_state = self._latest_lowstate

            if low_state is None:
                positions = torch.zeros(12, dtype=torch.float32)
                velocities = torch.zeros(12, dtype=torch.float32)
                orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                angular_vel = torch.zeros(3, dtype=torch.float32)
                linear_acc = torch.zeros(3, dtype=torch.float32)
            else:
                positions = torch.tensor(
                    [m.q for m in low_state.motor_state[:12]], dtype=torch.float32
                )
                velocities = torch.tensor(
                    [m.dq for m in low_state.motor_state[:12]], dtype=torch.float32
                )
                imu = low_state.imu_state
                orientation = torch.tensor(list(imu.quaternion), dtype=torch.float32)
                angular_vel = torch.tensor(list(imu.gyroscope), dtype=torch.float32)
                linear_acc = torch.tensor(list(imu.accelerometer), dtype=torch.float32)

            raw_state = torch.cat(
                [positions, velocities, orientation, angular_vel, linear_acc]
            ).unsqueeze(0)

            return make_observation(
                state=raw_state,
                state_dim=self.config.state_dim,
                max_state_dim=self.config.max_state_dim,
                device="cpu",
            )

        state = self._robot.state()
        positions = torch.tensor(state.joint_positions(), dtype=torch.float32)
        velocities = torch.tensor(state.joint_velocities(), dtype=torch.float32)
        imu = state.imu
        orientation = torch.tensor(imu.quaternion, dtype=torch.float32)
        angular_vel = torch.tensor(imu.gyroscope, dtype=torch.float32)
        linear_acc = torch.tensor(imu.accelerometer, dtype=torch.float32)

        raw_state = torch.cat(
            [positions, velocities, orientation, angular_vel, linear_acc]
        ).unsqueeze(0)

        return make_observation(
            state=raw_state,
            state_dim=self.config.state_dim,
            max_state_dim=self.config.max_state_dim,
            device="cpu",
        )

    def act(self, action: torch.Tensor) -> None:
        if self._backend_mode == "unitree_sdk2py":
            if self.edu_mode:
                raise RuntimeError("EDU mode is not supported with unitree_sdk2py backend")
            vx = action[0, 0].item()
            vy = action[0, 1].item()
            vyaw = action[0, 2].item()
            self._check_rc(self._sport_client.Move(vx, vy, vyaw), "Move")
            return
        if self._backend_mode == "unitree_subprocess":
            if self.edu_mode:
                raise RuntimeError("EDU mode is not supported with unitree subprocess backend")
            vx = action[0, 0].item()
            vy = action[0, 1].item()
            vyaw = action[0, 2].item()
            self._check_rc(self._run_unitree_cmd("Move", vx, vy, vyaw), "Move")
            return

        if not self.edu_mode:
            vx = action[0, 0].item()
            vy = action[0, 1].item()
            vyaw = action[0, 2].item()
            self._robot.walk(vx, vy, vyaw)
        else:
            action_12dof = action[0, :12].cpu().numpy()
            kp = action[0, 12].item() if action.shape[1] > 12 else 20.0
            kd = action[0, 13].item() if action.shape[1] > 13 else 0.5
            self._robot.set_motor_positions(list(action_12dof), kp, kd)

    def reset(self) -> Dict[str, torch.Tensor]:
        if self._backend_mode == "unitree_sdk2py":
            self._check_rc(self._sport_client.RecoveryStand(), "RecoveryStand")
            return self.observe()
        if self._backend_mode == "unitree_subprocess":
            self._check_rc(self._run_unitree_cmd("RecoveryStand"), "RecoveryStand")
            return self.observe()
        self._robot.stand()
        return self.observe()

    def go_home(self) -> None:
        if self._backend_mode == "unitree_sdk2py":
            self._check_rc(self._sport_client.RecoveryStand(), "RecoveryStand")
            return
        if self._backend_mode == "unitree_subprocess":
            self._check_rc(self._run_unitree_cmd("RecoveryStand"), "RecoveryStand")
            return
        self._robot.stand()

    def disconnect(self) -> None:
        if self._backend_mode == "unitree_sdk2py":
            if self._sport_client is not None:
                try:
                    self._sport_client.StopMove()
                except Exception:
                    pass
            if self._state_sub is not None:
                try:
                    self._state_sub.Close()
                except Exception:
                    pass
            return
        if self._backend_mode == "unitree_subprocess":
            try:
                self._run_unitree_cmd("StopMove")
            except Exception:
                pass
            return
        self._robot.disconnect()

    def stand(self) -> None:
        if self._backend_mode == "unitree_sdk2py":
            self._check_rc(self._sport_client.RecoveryStand(), "RecoveryStand")
            return
        if self._backend_mode == "unitree_subprocess":
            self._check_rc(self._run_unitree_cmd("RecoveryStand"), "RecoveryStand")
            return
        self._robot.stand()

    def sit(self) -> None:
        if self._backend_mode == "unitree_sdk2py":
            self._check_rc(self._sport_client.Sit(), "Sit")
            return
        if self._backend_mode == "unitree_subprocess":
            self._check_rc(self._run_unitree_cmd("Sit"), "Sit")
            return
        self._robot.sit()

    def walk(self, vx: float, vy: float, vyaw: float) -> None:
        if self._backend_mode == "unitree_sdk2py":
            self._check_rc(self._sport_client.Move(vx, vy, vyaw), "Move")
            return
        if self._backend_mode == "unitree_subprocess":
            self._check_rc(self._run_unitree_cmd("Move", vx, vy, vyaw), "Move")
            return
        self._robot.walk(vx, vy, vyaw)


class Go2Robot:
    """Convenience class for Go2 robot."""

    def __new__(cls, ip_address: str = "192.168.123.161", **kwargs):
        from .base import RealRobot

        return RealRobot(config=GO2_CONFIG, robot_type="go2", ip_address=ip_address, **kwargs)
