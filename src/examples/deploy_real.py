#!/usr/bin/env python3
"""
Deploy VLA to Real Robot

Usage:
    python deploy_real.py --port /dev/ttyACM0 --checkpoint checkpoints/vla.pt
    python deploy_real.py --mock  # Test without hardware
"""

import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path

import rfx
from rfx.real import RealRobot
from rfx.sim import MockRobot


class SimpleVLA(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, obs: dict) -> torch.Tensor:
        return self.policy(self.encoder(obs["state"]))


def deploy(args):
    if args.mock:
        robot = MockRobot(state_dim=12, action_dim=6, num_envs=1)
    else:
        robot = RealRobot.from_config(Path(__file__).parent.parent / "configs" / "so101.yaml", port=args.port)

    policy = SimpleVLA(robot.max_state_dim, robot.max_action_dim)
    if Path(args.checkpoint).exists():
        policy.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model_state_dict"])
    policy.eval()

    robot.reset()
    time.sleep(1.0)

    print("Running... Ctrl+C to stop")
    try:
        while True:
            obs = robot.observe()
            with torch.no_grad():
                action = policy(obs)
            robot.act(action)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    robot.reset()
    if hasattr(robot, "disconnect"):
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--checkpoint", default="checkpoints/vla.pt")
    parser.add_argument("--mock", action="store_true")
    deploy(parser.parse_args())
