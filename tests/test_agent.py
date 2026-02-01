"""Tests for rfx.agent module."""

import pytest
from typing import Any, Callable
from unittest.mock import MagicMock, patch

from rfx.agent import Agent, AgentConfig, MockAgent
from rfx.skills import skill, Skill


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AgentConfig()

        assert config.model == "claude-sonnet-4-20250514"
        assert config.api_key is None
        assert config.max_tool_calls == 10
        assert config.verbose is False
        assert "robot control" in config.system_prompt.lower()

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = AgentConfig(
            model="gpt-4",
            api_key="test-key",
            max_tool_calls=5,
            verbose=True,
        )

        assert config.model == "gpt-4"
        assert config.api_key == "test-key"
        assert config.max_tool_calls == 5
        assert config.verbose is True


class TestAgent:
    """Tests for the Agent class."""

    def test_agent_init_default(self) -> None:
        """Test agent initialization with defaults."""
        agent = Agent()

        assert agent.config.model == "claude-sonnet-4-20250514"
        assert len(agent.registry) == 0

    def test_agent_init_with_model(self) -> None:
        """Test agent initialization with custom model."""
        agent = Agent(model="gpt-4o")

        assert agent.config.model == "gpt-4o"

    def test_agent_init_with_skills(self, sample_skill_func: Callable[..., str]) -> None:
        """Test agent initialization with skills."""
        agent = Agent(skills=[sample_skill_func])

        assert len(agent.registry) == 1
        assert "walk_forward" in agent.registry

    def test_agent_init_with_skill_objects(self) -> None:
        """Test agent initialization with Skill objects."""
        @skill
        def test_skill() -> None:
            """Test skill."""
            pass

        agent = Agent(skills=[test_skill])

        assert len(agent.registry) == 1
        assert "test_skill" in agent.registry

    def test_agent_add_skill(self, sample_skill_func: Callable[..., str]) -> None:
        """Test adding skills after initialization."""
        agent = Agent()

        s = agent.add_skill(sample_skill_func)

        assert isinstance(s, Skill)
        assert "walk_forward" in agent.registry

    def test_agent_describe_skills(self, sample_skill_func: Callable[..., str]) -> None:
        """Test skill description."""
        agent = Agent(skills=[sample_skill_func])

        description = agent.describe_skills()

        assert "walk_forward" in description

    def test_agent_repr(self) -> None:
        """Test agent string representation."""
        agent = Agent(model="gpt-4", skills=[])

        repr_str = repr(agent)

        assert "Agent" in repr_str
        assert "gpt-4" in repr_str


class TestAgentApiDetection:
    """Tests for API type detection."""

    def test_detect_anthropic_claude(self) -> None:
        """Test detection of Anthropic Claude models."""
        agent = Agent(model="claude-sonnet-4-20250514")
        assert agent._api_type == "anthropic"

        agent = Agent(model="claude-3-opus-20240229")
        assert agent._api_type == "anthropic"

        agent = Agent(model="Claude-3-Haiku")
        assert agent._api_type == "anthropic"

    def test_detect_openai_gpt(self) -> None:
        """Test detection of OpenAI GPT models."""
        agent = Agent(model="gpt-4")
        assert agent._api_type == "openai"

        agent = Agent(model="gpt-4o")
        assert agent._api_type == "openai"

        agent = Agent(model="GPT-4-turbo")
        assert agent._api_type == "openai"

    def test_detect_openai_o1(self) -> None:
        """Test detection of OpenAI o1 models."""
        agent = Agent(model="o1-preview")
        assert agent._api_type == "openai"

        agent = Agent(model="o1-mini")
        assert agent._api_type == "openai"

    def test_detect_default_to_anthropic(self) -> None:
        """Test unknown models default to Anthropic."""
        agent = Agent(model="some-unknown-model")
        assert agent._api_type == "anthropic"


class TestAgentGetClient:
    """Tests for client initialization."""

    def test_get_client_anthropic_missing_package(self) -> None:
        """Test error when anthropic package is missing."""
        agent = Agent(model="claude-sonnet-4-20250514")

        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                agent._get_client()

    def test_get_client_openai_missing_package(self) -> None:
        """Test error when openai package is missing."""
        agent = Agent(model="gpt-4")

        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                agent._get_client()

    def test_get_client_cached(self) -> None:
        """Test that client is cached after first creation."""
        agent = Agent(model="claude-sonnet-4-20250514")
        mock_client = MagicMock()
        agent._client = mock_client

        client = agent._get_client()

        assert client is mock_client


class TestMockAgent:
    """Tests for the MockAgent class."""

    def test_mock_agent_init_empty(self) -> None:
        """Test MockAgent initialization without skills."""
        agent = MockAgent()

        assert len(agent.registry) == 0

    def test_mock_agent_init_with_skills(self, sample_skill_func: Callable[..., str]) -> None:
        """Test MockAgent initialization with skills."""
        agent = MockAgent(skills=[sample_skill_func])

        assert len(agent.registry) == 1
        assert "walk_forward" in agent.registry

    def test_mock_agent_add_skill(self, sample_skill_func: Callable[..., str]) -> None:
        """Test adding skills to MockAgent."""
        agent = MockAgent()

        s = agent.add_skill(sample_skill_func)

        assert isinstance(s, Skill)
        assert "walk_forward" in agent.registry

    def test_mock_agent_execute_skill(self, sample_skill_func: Callable[..., str]) -> None:
        """Test executing skills on MockAgent."""
        agent = MockAgent(skills=[sample_skill_func])

        result = agent.execute_skill("walk_forward", distance=5.0)

        assert result == "Walked 5.0m"

    def test_mock_agent_execute_skill_default_args(self, sample_skill_func: Callable[..., str]) -> None:
        """Test executing skills with default arguments."""
        agent = MockAgent(skills=[sample_skill_func])

        result = agent.execute_skill("walk_forward")

        assert result == "Walked 1.0m"

    def test_mock_agent_execute_unknown_skill(self) -> None:
        """Test executing unknown skill raises error."""
        agent = MockAgent()

        with pytest.raises(KeyError, match="Unknown skill"):
            agent.execute_skill("nonexistent")

    def test_mock_agent_describe_skills(self, sample_skill_func: Callable[..., str]) -> None:
        """Test skill description for MockAgent."""
        agent = MockAgent(skills=[sample_skill_func])

        description = agent.describe_skills()

        assert "walk_forward" in description


class TestAgentExecuteIntegration:
    """Integration tests for agent execution (requires API mocking)."""

    def test_execute_anthropic_no_tools(self) -> None:
        """Test Anthropic execution when no tool calls are made."""
        agent = Agent(model="claude-sonnet-4-20250514")

        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "I understand. Hello!"
        mock_response.content = [mock_text_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        agent._client = mock_client

        result = agent.execute("Hello")

        assert result == "I understand. Hello!"

    def test_execute_anthropic_with_tool_call(self, sample_skill_func: Callable[..., str]) -> None:
        """Test Anthropic execution with tool calls."""
        agent = Agent(model="claude-sonnet-4-20250514", skills=[sample_skill_func])

        # First response: tool use
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "walk_forward"
        mock_tool_block.input = {"distance": 2.0}
        mock_tool_block.id = "tool_123"

        mock_response1 = MagicMock()
        mock_response1.content = [mock_tool_block]

        # Second response: text
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Done walking!"

        mock_response2 = MagicMock()
        mock_response2.content = [mock_text_block]

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [mock_response1, mock_response2]
        agent._client = mock_client

        result = agent.execute("Walk forward 2 meters")

        assert "Done walking!" in result

    def test_execute_openai_no_tools(self) -> None:
        """Test OpenAI execution when no tool calls are made."""
        agent = Agent(model="gpt-4")

        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "Hello there!"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        agent._client = mock_client

        result = agent.execute("Hello")

        assert result == "Hello there!"

    def test_execute_openai_with_tool_call(self, sample_skill_func: Callable[..., str]) -> None:
        """Test OpenAI execution with tool calls."""
        import json

        agent = Agent(model="gpt-4", skills=[sample_skill_func])

        # First response: tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "walk_forward"
        mock_tool_call.function.arguments = json.dumps({"distance": 3.0})

        mock_message1 = MagicMock()
        mock_message1.tool_calls = [mock_tool_call]

        mock_choice1 = MagicMock()
        mock_choice1.message = mock_message1

        mock_response1 = MagicMock()
        mock_response1.choices = [mock_choice1]

        # Second response: done
        mock_message2 = MagicMock()
        mock_message2.tool_calls = None
        mock_message2.content = "Walked successfully!"

        mock_choice2 = MagicMock()
        mock_choice2.message = mock_message2

        mock_response2 = MagicMock()
        mock_response2.choices = [mock_choice2]

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
        agent._client = mock_client

        result = agent.execute("Walk forward 3 meters")

        assert "successfully" in result.lower() or "Walked" in result
