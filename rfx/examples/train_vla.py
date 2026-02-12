#!/usr/bin/env python3
"""
VLA Training Example

Usage:
    python train_vla.py --num_envs 16 --steps 1000 --backend mock
    python train_vla.py --num_envs 4096 --steps 1000000 --backend genesis
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path

import rfx
from rfx.sim import SimRobot


class SimpleVLA(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        return self.policy(self.encoder(obs["state"]))


def train(args):
    print(f"Training with {args.num_envs} envs, backend={args.backend}")

    robot = SimRobot.from_config(
        Path(__file__).parent.parent / "configs" / "so101.yaml",
        num_envs=args.num_envs,
        backend=args.backend,
        device=args.device,
    )
    print(f"Robot: {robot}")

    policy = SimpleVLA(robot.max_state_dim, robot.max_action_dim).to(args.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    target = torch.randn(args.num_envs, 6, device=args.device) * 0.5
    obs = robot.reset()

    for step in range(args.steps):
        action = policy(obs)
        robot.act(action.detach())
        new_obs = robot.observe()

        positions = new_obs["state"][:, :6]
        reward = -torch.norm(positions - target, dim=-1)
        loss = -(reward.unsqueeze(-1) * action).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        done = robot.get_done()
        if done.any():
            robot.reset(done.nonzero().squeeze(-1))
            target[done] = torch.randn(done.sum(), 6, device=args.device) * 0.5

        obs = new_obs

        if step % args.log_interval == 0:
            print(
                f"Step {step:6d} | Loss: {loss.item():8.4f} | Reward: {reward.mean().item():8.4f}"
            )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    train(parser.parse_args())
