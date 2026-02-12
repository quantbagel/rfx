"""
Skill system for LLM agent integration

The @skill decorator exposes Python functions to LLM agents for natural language control.
Inspired by DimensionalOS's skill-based architecture.

Example:
    >>> import rfx
    >>>
    >>> @rfx.skill
    >>> def wave_hello():
    ...     '''Make the robot wave its front leg'''
    ...     go2.set_motor("FL_hip", position=0.5)
    ...     time.sleep(0.5)
    ...     go2.set_motor("FL_hip", position=0.0)
    >>>
    >>> # Skills can be discovered by LLM agents
    >>> registry = rfx.SkillRegistry()
    >>> registry.register(wave_hello)
    >>> tools = registry.to_tools()  # OpenAI/Anthropic tool format
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, get_type_hints

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Docstring Parsing
# =============================================================================


@dataclass
class ParsedDocstring:
    """Parsed docstring components."""

    description: str
    params: Dict[str, str]  # param_name -> description
    returns: Optional[str] = None


def _parse_docstring(docstring: Optional[str]) -> ParsedDocstring:
    """
    Parse a docstring to extract description and parameter documentation.

    Supports Google-style and NumPy-style docstrings.

    Google-style example:
        '''Brief description.

        Longer description if needed.

        Args:
            param1: Description of param1.
            param2: Description of param2.

        Returns:
            Description of return value.
        '''

    NumPy-style example:
        '''Brief description.

        Longer description if needed.

        Parameters
        ----------
        param1 : type
            Description of param1.
        param2 : type
            Description of param2.

        Returns
        -------
        type
            Description of return value.
        '''
    """
    if not docstring:
        return ParsedDocstring(description="", params={})

    lines = docstring.strip().split("\n")
    description_lines: List[str] = []
    params: Dict[str, str] = {}
    returns: Optional[str] = None

    # State machine for parsing
    state = "description"
    current_param: Optional[str] = None
    current_param_desc: List[str] = []

    # Patterns for different docstring styles
    google_args_pattern = re.compile(r"^\s*Args:\s*$", re.IGNORECASE)
    google_returns_pattern = re.compile(r"^\s*Returns:\s*$", re.IGNORECASE)
    google_param_pattern = re.compile(r"^\s{4,}(\w+):\s*(.*)$")
    google_continuation_pattern = re.compile(r"^\s{8,}(.+)$")

    numpy_params_pattern = re.compile(r"^\s*Parameters\s*$", re.IGNORECASE)
    numpy_returns_pattern = re.compile(r"^\s*Returns\s*$", re.IGNORECASE)
    numpy_separator_pattern = re.compile(r"^\s*-{3,}\s*$")
    numpy_param_pattern = re.compile(r"^\s*(\w+)\s*:\s*.*$")
    numpy_param_desc_pattern = re.compile(r"^\s{4,}(.+)$")

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for section headers
        if google_args_pattern.match(line):
            state = "google_args"
            i += 1
            continue
        elif google_returns_pattern.match(line):
            # Save current param if any
            if current_param and current_param_desc:
                params[current_param] = " ".join(current_param_desc)
            state = "google_returns"
            current_param = None
            current_param_desc = []
            i += 1
            continue
        elif numpy_params_pattern.match(line):
            # Check for separator line
            if i + 1 < len(lines) and numpy_separator_pattern.match(lines[i + 1]):
                state = "numpy_params"
                i += 2
                continue
        elif numpy_returns_pattern.match(line):
            # Save current param if any
            if current_param and current_param_desc:
                params[current_param] = " ".join(current_param_desc)
            if i + 1 < len(lines) and numpy_separator_pattern.match(lines[i + 1]):
                state = "numpy_returns"
                current_param = None
                current_param_desc = []
                i += 2
                continue

        # Handle content based on state
        if state == "description":
            # Skip empty lines at the start
            if line.strip() or description_lines:
                description_lines.append(line.strip())

        elif state == "google_args":
            param_match = google_param_pattern.match(line)
            continuation_match = google_continuation_pattern.match(line)

            if param_match:
                # Save previous param
                if current_param and current_param_desc:
                    params[current_param] = " ".join(current_param_desc)

                current_param = param_match.group(1)
                current_param_desc = (
                    [param_match.group(2).strip()] if param_match.group(2).strip() else []
                )
            elif continuation_match and current_param:
                current_param_desc.append(continuation_match.group(1).strip())
            elif not line.strip():
                # Empty line might end the section
                pass

        elif state == "google_returns":
            stripped = line.strip()
            if stripped and not stripped.startswith("Returns"):
                if returns is None:
                    returns = stripped
                else:
                    returns += " " + stripped

        elif state == "numpy_params":
            param_match = numpy_param_pattern.match(line)
            desc_match = numpy_param_desc_pattern.match(line)

            if param_match:
                # Save previous param
                if current_param and current_param_desc:
                    params[current_param] = " ".join(current_param_desc)

                current_param = param_match.group(1)
                current_param_desc = []
            elif desc_match and current_param:
                current_param_desc.append(desc_match.group(1).strip())

        elif state == "numpy_returns":
            desc_match = numpy_param_desc_pattern.match(line)
            if desc_match:
                if returns is None:
                    returns = desc_match.group(1).strip()
                else:
                    returns += " " + desc_match.group(1).strip()

        i += 1

    # Save final param
    if current_param and current_param_desc:
        params[current_param] = " ".join(current_param_desc)

    # Build description, stopping at the first section header or empty line run
    description = ""
    for line in description_lines:
        if not line:
            break
        if google_args_pattern.match(line) or numpy_params_pattern.match(line):
            break
        description += (" " if description else "") + line

    return ParsedDocstring(
        description=description.strip(),
        params=params,
        returns=returns,
    )


@dataclass
class Skill:
    """A skill that can be executed by an LLM agent."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the skill."""
        return self.func(*args, **kwargs)

    def to_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI/Anthropic tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


def _extract_parameters(
    func: Callable[..., Any],
    parsed_docstring: Optional[ParsedDocstring] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Extract parameter information from function signature, type hints, and docstring.

    Args:
        func: The function to extract parameters from.
        parsed_docstring: Pre-parsed docstring (if None, will parse func.__doc__).

    Returns:
        Tuple of (parameters dict, required params list).
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    # Parse docstring for parameter descriptions
    if parsed_docstring is None:
        parsed_docstring = _parse_docstring(func.__doc__)

    parameters: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        param_info: Dict[str, Any] = {}

        # Get type hint
        if name in hints:
            hint = hints[name]
            type_name = _python_type_to_json_type(hint)
            param_info["type"] = type_name
        else:
            param_info["type"] = "string"  # Default to string

        # Check if required
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            param_info["default"] = param.default

        # Add description from docstring if available
        if name in parsed_docstring.params:
            param_info["description"] = parsed_docstring.params[name]

        parameters[name] = param_info

    return parameters, required


def _python_type_to_json_type(hint: Any) -> str:
    """Convert Python type hint to JSON schema type."""
    origin = getattr(hint, "__origin__", None)

    if hint is str:
        return "string"
    elif hint is int:
        return "integer"
    elif hint is float:
        return "number"
    elif hint is bool:
        return "boolean"
    elif hint is list or origin is list:
        return "array"
    elif hint is dict or origin is dict:
        return "object"
    elif hint is None or hint is type(None):
        return "null"
    else:
        return "string"


def skill(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Union[Skill, Callable[[F], Skill]]:
    """
    Decorator to mark a function as a skill for LLM agents.

    Args:
        func: The function to decorate (when used without arguments)
        name: Custom name for the skill (defaults to function name)
        description: Custom description (defaults to docstring)
        tags: Tags for categorization

    Returns:
        A Skill object wrapping the function

    Example:
        >>> @skill
        >>> def walk_forward(distance: float = 1.0):
        ...     '''Walk forward by the specified distance in meters'''
        ...     go2.walk(0.3, 0, 0)
        ...     time.sleep(distance / 0.3)
        ...     go2.stand()

        >>> @skill(name="look_around", tags=["perception"])
        >>> def survey():
        ...     '''Rotate in place to survey surroundings'''
        ...     go2.walk(0, 0, 0.5)
        ...     time.sleep(2.0)
        ...     go2.stand()
    """

    def decorator(f: F) -> Skill:
        skill_name = name or f.__name__

        # Parse docstring once for both description and parameter docs
        parsed_doc = _parse_docstring(f.__doc__)

        # Use explicit description, or parsed description, or fallback
        if description:
            skill_description = description
        elif parsed_doc.description:
            skill_description = parsed_doc.description
        else:
            skill_description = f"Execute {skill_name}"

        parameters, required = _extract_parameters(f, parsed_doc)

        return Skill(
            name=skill_name,
            description=skill_description,
            func=f,
            parameters=parameters,
            required=required,
            returns=parsed_doc.returns,
            tags=tags or [],
        )

    # Handle @skill and @skill() syntax
    if func is not None:
        return decorator(func)
    return decorator


class SkillRegistry:
    """Registry for managing skills available to LLM agents."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill_or_func: Union[Skill, Callable[..., Any]], **kwargs: Any) -> Skill:
        """
        Register a skill or function.

        Args:
            skill_or_func: A Skill object or a function to wrap as a skill
            **kwargs: Additional arguments passed to skill() decorator if wrapping a function

        Returns:
            The registered Skill
        """
        if isinstance(skill_or_func, Skill):
            s = skill_or_func
        else:
            s = skill(skill_or_func, **kwargs)

        self._skills[s.name] = s
        return s

    def unregister(self, name: str) -> Optional[Skill]:
        """Remove a skill from the registry."""
        return self._skills.pop(name, None)

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def __getitem__(self, name: str) -> Skill:
        """Get a skill by name."""
        return self._skills[name]

    def __contains__(self, name: str) -> bool:
        """Check if a skill is registered."""
        return name in self._skills

    def __iter__(self):
        """Iterate over skills."""
        return iter(self._skills.values())

    def __len__(self) -> int:
        """Number of registered skills."""
        return len(self._skills)

    @property
    def skills(self) -> List[Skill]:
        """List all registered skills."""
        return list(self._skills.values())

    def to_tools(self) -> List[Dict[str, Any]]:
        """Convert all skills to OpenAI tool format."""
        return [s.to_tool() for s in self._skills.values()]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Convert all skills to Anthropic tool format."""
        return [s.to_anthropic_tool() for s in self._skills.values()]

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a skill by name with the given arguments."""
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"Unknown skill: {name}")
        return skill(**kwargs)

    def filter_by_tag(self, tag: str) -> List[Skill]:
        """Get skills with a specific tag."""
        return [s for s in self._skills.values() if tag in s.tags]

    def describe(self) -> str:
        """Get a human-readable description of all skills."""
        if not self._skills:
            return "No skills registered."

        lines = ["Available skills:"]
        for skill in self._skills.values():
            lines.append(f"  - {skill.name}: {skill.description}")
            if skill.parameters:
                params = ", ".join(
                    f"{k}: {v.get('type', 'any')}" for k, v in skill.parameters.items()
                )
                lines.append(f"      Parameters: {params}")
        return "\n".join(lines)


# =============================================================================
# Context-Scoped Registry
# =============================================================================

import warnings
from contextvars import ContextVar, Token
from contextlib import contextmanager
from typing import Iterator


# Context variable for async-safe registry scoping
_registry_context: ContextVar[Optional[SkillRegistry]] = ContextVar("skill_registry", default=None)

# Legacy global registry (deprecated)
_global_registry: Optional[SkillRegistry] = None


def _get_or_create_global() -> SkillRegistry:
    """Get or lazily create the global registry (for backward compatibility)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


@contextmanager
def skill_registry_context(registry: Optional[SkillRegistry] = None) -> Iterator[SkillRegistry]:
    """
    Context manager for scoped skill registry access.

    This is the recommended way to manage skill registries in async code
    or when you need isolated registries for testing.

    Args:
        registry: Optional registry to use. If None, creates a new one.

    Yields:
        The scoped SkillRegistry instance.

    Example:
        >>> async def test_skills():
        ...     with skill_registry_context() as registry:
        ...         @skill
        ...         def my_skill():
        ...             pass
        ...         registry.register(my_skill)
        ...         # registry is isolated to this context
        ...     # registry is cleaned up after context exits

        >>> # Or with an existing registry:
        >>> my_registry = SkillRegistry()
        >>> with skill_registry_context(my_registry) as registry:
        ...     assert registry is my_registry
    """
    if registry is None:
        registry = SkillRegistry()

    token = _registry_context.set(registry)
    try:
        yield registry
    finally:
        _registry_context.reset(token)


def get_current_registry() -> SkillRegistry:
    """
    Get the current skill registry.

    Returns the context-scoped registry if inside a skill_registry_context,
    otherwise returns the global registry (with deprecation warning if used
    without a context).

    Returns:
        The current SkillRegistry instance.
    """
    ctx_registry = _registry_context.get()
    if ctx_registry is not None:
        return ctx_registry

    # Fall back to global with warning
    return _get_or_create_global()


def get_global_registry() -> SkillRegistry:
    """
    Get the global skill registry.

    .. deprecated::
        Use `skill_registry_context()` for new code, or `get_current_registry()`
        if you need backward compatibility.

    Returns:
        The global SkillRegistry instance.
    """
    warnings.warn(
        "get_global_registry() is deprecated. Use skill_registry_context() "
        "for async-safe scoped registries, or get_current_registry() for "
        "backward-compatible access.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _get_or_create_global()


def register_skill(func: Callable[..., Any], **kwargs: Any) -> Skill:
    """
    Register a skill to the current registry.

    Uses the context-scoped registry if inside a skill_registry_context,
    otherwise uses the global registry.

    Args:
        func: The function to register as a skill.
        **kwargs: Additional arguments passed to the skill() decorator.

    Returns:
        The registered Skill.
    """
    return get_current_registry().register(func, **kwargs)
