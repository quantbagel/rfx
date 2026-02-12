"""
LLM Agent integration for natural language robot control

Provides an Agent class that can execute skills based on natural language commands.
Supports both OpenAI and Anthropic APIs through a unified LLMClient interface.

Example:
    >>> import rfx
    >>> from rfx.agent import Agent
    >>>
    >>> @rfx.skill
    >>> def walk_forward(distance: float = 1.0):
    ...     '''Walk forward by the specified distance in meters'''
    ...     go2.walk(0.3, 0, 0)
    ...     time.sleep(distance / 0.3)
    ...     go2.stand()
    >>>
    >>> agent = Agent(
    ...     model="claude-sonnet-4-20250514",
    ...     skills=[walk_forward],
    ...     robot=go2,
    ... )
    >>>
    >>> agent.execute("walk forward 2 meters")
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from .skills import Skill, SkillRegistry, skill


# =============================================================================
# Completion Result Types
# =============================================================================


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class CompletionResult:
    """Result from an LLM completion."""

    text: Optional[str]
    tool_calls: List[ToolCall]
    raw_response: Any = None


# =============================================================================
# LLM Client Protocol
# =============================================================================


class LLMClient(ABC):
    """
    Abstract base class for LLM API clients.

    Provides a unified interface for interacting with different LLM providers
    (Anthropic, OpenAI, etc.) while handling the differences in their APIs.
    """

    @abstractmethod
    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict[str, Any]],
        model: str,
    ) -> CompletionResult:
        """
        Create a completion from the LLM.

        Args:
            messages: Conversation history
            system_prompt: System instructions
            tools: Available tools/functions
            model: Model identifier

        Returns:
            CompletionResult with text and/or tool calls
        """
        ...

    @abstractmethod
    def format_tool_result(
        self,
        tool_call: ToolCall,
        result: str,
        is_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Format a tool result for the next API call.

        Args:
            tool_call: The original tool call
            result: The result string
            is_error: Whether the result is an error

        Returns:
            Formatted tool result for the API
        """
        ...

    @abstractmethod
    def format_assistant_message(self, response: Any) -> Dict[str, Any]:
        """Format the assistant's response as a message for conversation history."""
        ...


class AnthropicClient(LLMClient):
    """LLM client for Anthropic's Claude API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic models. "
                "Install with: pip install anthropic"
            )

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict[str, Any]],
        model: str,
    ) -> CompletionResult:
        response = self._client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            tools=tools if tools else None,
            messages=messages,
        )

        # Extract text blocks
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return CompletionResult(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            raw_response=response,
        )

    def format_tool_result(
        self,
        tool_call: ToolCall,
        result: str,
        is_error: bool = False,
    ) -> Dict[str, Any]:
        tool_result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": result,
        }
        if is_error:
            tool_result["is_error"] = True
        return tool_result

    def format_assistant_message(self, response: Any) -> Dict[str, Any]:
        return {"role": "assistant", "content": response.raw_response.content}


class OpenAIClient(LLMClient):
    """LLM client for OpenAI's API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        try:
            import openai

            self._client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI models. Install with: pip install openai"
            )

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict[str, Any]],
        model: str,
    ) -> CompletionResult:
        # OpenAI uses system message in messages array
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = self._client.chat.completions.create(
            model=model,
            messages=full_messages,
            tools=tools if tools else None,
        )

        message = response.choices[0].message
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return CompletionResult(
            text=message.content,
            tool_calls=tool_calls,
            raw_response=response,
        )

    def format_tool_result(
        self,
        tool_call: ToolCall,
        result: str,
        is_error: bool = False,
    ) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        }

    def format_assistant_message(self, response: Any) -> Dict[str, Any]:
        return response.raw_response.choices[0].message


# =============================================================================
# Agent Configuration
# =============================================================================


@dataclass
class AgentConfig:
    """Configuration for the LLM agent."""

    # Model settings
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None

    # System prompt
    system_prompt: str = """You are a robot control assistant. You help users control a quadruped robot using available skills.

When the user asks you to do something:
1. Determine which skill(s) to use
2. Call the appropriate skill function(s) with the right parameters
3. Report what you did

Be concise and action-oriented. Focus on executing the user's commands efficiently."""

    # Behavior settings
    max_tool_calls: int = 10
    verbose: bool = False


# =============================================================================
# Agent
# =============================================================================


class Agent:
    """
    LLM agent for natural language robot control.

    Uses function calling / tool use to execute robot skills based on
    natural language commands.

    Args:
        model: The LLM model to use (e.g., "claude-sonnet-4-20250514", "gpt-4")
        skills: List of skills or functions to make available
        robot: Optional robot instance for context
        config: Additional configuration options

    Example:
        >>> agent = Agent(
        ...     model="claude-sonnet-4-20250514",
        ...     skills=[walk_forward, look_around, sit_down],
        ... )
        >>> result = agent.execute("walk forward 2 meters, then look around")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        skills: Optional[List[Union[Skill, Callable[..., Any]]]] = None,
        robot: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.config = config or AgentConfig(model=model, api_key=api_key)
        self.robot = robot

        # Set up skill registry
        self.registry = SkillRegistry()
        if skills:
            for s in skills:
                self.registry.register(s)

        # Detect API type from model name
        self._api_type = self._detect_api_type(self.config.model)

        # LLM client will be initialized lazily
        self._llm_client: Optional[LLMClient] = None

    def _detect_api_type(self, model: str) -> str:
        """Detect which API to use based on model name."""
        model_lower = model.lower()
        if "claude" in model_lower:
            return "anthropic"
        elif "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        else:
            # Default to Anthropic
            return "anthropic"

    def _get_llm_client(self) -> LLMClient:
        """Get or create the typed LLM client wrapper."""
        if self._llm_client is not None:
            return self._llm_client

        if self._api_type == "anthropic":
            self._llm_client = AnthropicClient(api_key=self.config.api_key)
        else:
            self._llm_client = OpenAIClient(api_key=self.config.api_key)

        return self._llm_client

    def _get_client(self) -> Any:
        """Legacy accessor returning the raw provider client."""
        llm_client = self._get_llm_client()
        return getattr(llm_client, "_client", None)

    # Legacy property for backward compatibility
    @property
    def _client(self) -> Optional[Any]:
        """Legacy accessor for raw client. Use _get_client() instead."""
        if self._llm_client is None:
            return None
        return getattr(self._llm_client, "_client", None)

    @_client.setter
    def _client(self, value: Any) -> None:
        """Legacy setter for testing. Creates appropriate wrapper."""
        if value is None:
            self._llm_client = None
        elif self._api_type == "anthropic":
            client = AnthropicClient.__new__(AnthropicClient)
            client._client = value
            self._llm_client = client
        else:
            client = OpenAIClient.__new__(OpenAIClient)
            client._client = value
            self._llm_client = client

    def add_skill(self, skill_or_func: Union[Skill, Callable[..., Any]], **kwargs: Any) -> Skill:
        """Add a skill to the agent."""
        return self.registry.register(skill_or_func, **kwargs)

    def execute(self, command: str) -> str:
        """
        Execute a natural language command.

        Args:
            command: The command to execute (e.g., "walk forward 2 meters")

        Returns:
            A string describing what was done
        """
        client = self._get_llm_client()

        # Get tools in the appropriate format
        if self._api_type == "anthropic":
            tools = self.registry.to_anthropic_tools()
        else:
            tools = self.registry.to_tools()

        messages: List[Dict[str, Any]] = [{"role": "user", "content": command}]
        tool_calls_made = 0
        results: List[str] = []

        while tool_calls_made < self.config.max_tool_calls:
            completion = client.create_completion(
                messages=messages,
                system_prompt=self.config.system_prompt,
                tools=tools,
                model=self.config.model,
            )

            # No tool calls - return text response
            if not completion.tool_calls:
                return completion.text or "Done."

            # Execute tool calls
            tool_results: List[Dict[str, Any]] = []
            for tool_call in completion.tool_calls:
                tool_calls_made += 1

                try:
                    result = self.registry.execute(tool_call.name, **tool_call.arguments)
                    result_str = str(result) if result is not None else "Success"
                    results.append(f"Executed {tool_call.name}: {result_str}")

                    tool_results.append(
                        client.format_tool_result(tool_call, result_str, is_error=False)
                    )
                except Exception as e:
                    error_str = f"Error: {e}"
                    results.append(f"Failed {tool_call.name}: {error_str}")

                    tool_results.append(
                        client.format_tool_result(tool_call, error_str, is_error=True)
                    )

            # Add messages to conversation
            messages.append(client.format_assistant_message(completion))

            # Add tool results (format differs by API)
            if self._api_type == "anthropic":
                messages.append({"role": "user", "content": tool_results})
            else:
                messages.extend(tool_results)

        return "\n".join(results) if results else "No actions taken."

    def describe_skills(self) -> str:
        """Get a description of available skills."""
        return self.registry.describe()

    def __repr__(self) -> str:
        return f"Agent(model={self.config.model!r}, skills={len(self.registry)})"


class MockAgent:
    """
    A mock agent for testing without API calls.

    Executes skills directly without LLM interpretation.
    Useful for testing skill implementations.
    """

    def __init__(
        self,
        skills: Optional[List[Union[Skill, Callable[..., Any]]]] = None,
    ) -> None:
        self.registry = SkillRegistry()
        if skills:
            for s in skills:
                self.registry.register(s)

    def add_skill(self, skill_or_func: Union[Skill, Callable[..., Any]], **kwargs: Any) -> Skill:
        """Add a skill to the agent."""
        return self.registry.register(skill_or_func, **kwargs)

    def execute_skill(self, name: str, **kwargs: Any) -> Any:
        """Execute a skill by name."""
        return self.registry.execute(name, **kwargs)

    def describe_skills(self) -> str:
        """Get a description of available skills."""
        return self.registry.describe()
