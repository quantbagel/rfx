"""Tests for rfx v2 Robot API"""

import pytest
import torch
from pathlib import Path

from rfx import Robot, RobotBase, SimRobot, MockRobot, RobotConfig
from rfx.sim.mock import MockBackend
from rfx.utils import pad_state, pad_action, unpad_action


class TestMockRobot:
    def test_create(self):
        robot = MockRobot(state_dim=12, action_dim=6, num_envs=16)
        assert robot.num_envs == 16
        assert robot.state_dim == 12
        assert robot.action_dim == 6

    def test_observe(self):
        robot = MockRobot(state_dim=12, action_dim=6, num_envs=16)
        obs = robot.observe()
        assert obs["state"].shape == (16, 64)

    def test_act(self):
        robot = MockRobot(state_dim=12, action_dim=6, num_envs=16)
        robot.act(torch.randn(16, 64))

    def test_reset(self):
        robot = MockRobot(state_dim=12, action_dim=6, num_envs=16)
        obs = robot.reset()
        assert obs["state"].shape == (16, 64)


class TestSimRobot:
    def test_from_config(self):
        config_path = Path(__file__).parent.parent / "configs" / "so101.yaml"
        robot = SimRobot.from_config(config_path, num_envs=32, backend="mock")
        assert robot.num_envs == 32
        assert robot.state_dim == 12
        assert robot.backend == "mock"


class TestRobotConfig:
    def test_load_so101(self):
        config_path = Path(__file__).parent.parent / "configs" / "so101.yaml"
        config = RobotConfig.from_yaml(config_path)
        assert config.name == "SO-101"
        assert config.state_dim == 12
        assert config.action_dim == 6


class TestPadding:
    def test_pad_state(self):
        state = torch.randn(16, 12)
        padded = pad_state(state, 12, 64)
        assert padded.shape == (16, 64)

    def test_unpad_action(self):
        padded = torch.randn(16, 64)
        action = unpad_action(padded, 6)
        assert action.shape == (16, 6)


class TestProtocol:
    def test_mock_robot_is_robot(self):
        robot = MockRobot(state_dim=12, action_dim=6)
        assert isinstance(robot, Robot)


class TestMultiEmbodiment:
    def test_same_padding(self):
        so101 = MockRobot(state_dim=12, action_dim=6, num_envs=8)
        go2 = MockRobot(state_dim=24, action_dim=12, num_envs=8)
        assert so101.observe()["state"].shape == go2.observe()["state"].shape
