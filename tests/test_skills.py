"""Tests for rfx.skills module."""

import pytest
from typing import Any, Callable

from rfx.skills import (
    Skill,
    SkillRegistry,
    skill,
    get_global_registry,
    register_skill,
    _extract_parameters,
    _python_type_to_json_type,
)


class TestSkillDecorator:
    """Tests for the @skill decorator."""

    def test_skill_decorator_basic(self, sample_skill_func: Callable[..., str]) -> None:
        """Test basic decorator usage."""
        decorated = skill(sample_skill_func)

        assert isinstance(decorated, Skill)
        assert decorated.name == "walk_forward"
        assert "Walk forward" in decorated.description
        assert callable(decorated)

    def test_skill_decorator_with_name(self, sample_skill_func: Callable[..., str]) -> None:
        """Test decorator with custom name."""
        decorated = skill(sample_skill_func, name="custom_walk")

        assert decorated.name == "custom_walk"

    def test_skill_decorator_with_description(self, sample_skill_func: Callable[..., str]) -> None:
        """Test decorator with custom description."""
        decorated = skill(sample_skill_func, description="Custom description")

        assert decorated.description == "Custom description"

    def test_skill_decorator_with_tags(self, sample_skill_func: Callable[..., str]) -> None:
        """Test decorator with tags."""
        decorated = skill(sample_skill_func, tags=["locomotion", "basic"])

        assert decorated.tags == ["locomotion", "basic"]

    def test_skill_decorator_parentheses_syntax(self) -> None:
        """Test @skill() syntax with parentheses."""
        @skill(name="test_skill", tags=["test"])
        def my_func() -> None:
            """A test function."""
            pass

        assert isinstance(my_func, Skill)
        assert my_func.name == "test_skill"
        assert my_func.tags == ["test"]

    def test_skill_decorator_no_parentheses_syntax(self) -> None:
        """Test @skill syntax without parentheses."""
        @skill
        def my_func() -> None:
            """A test function."""
            pass

        assert isinstance(my_func, Skill)
        assert my_func.name == "my_func"

    def test_skill_execution(self, sample_skill_func: Callable[..., str]) -> None:
        """Test that decorated function can still be called."""
        decorated = skill(sample_skill_func)

        result = decorated(2.5)
        assert result == "Walked 2.5m"

        result = decorated()
        assert result == "Walked 1.0m"

    def test_skill_parameters_extraction(self, sample_skill_with_types: Callable[..., dict]) -> None:
        """Test that parameters are correctly extracted."""
        decorated = skill(sample_skill_with_types)

        assert "x" in decorated.parameters
        assert "y" in decorated.parameters
        assert "speed" in decorated.parameters
        assert "blocking" in decorated.parameters

        assert decorated.parameters["x"]["type"] == "number"
        assert decorated.parameters["y"]["type"] == "number"
        assert decorated.parameters["speed"]["type"] == "number"
        assert decorated.parameters["blocking"]["type"] == "boolean"

    def test_skill_required_parameters(self, sample_skill_with_types: Callable[..., dict]) -> None:
        """Test required vs optional parameter detection."""
        decorated = skill(sample_skill_with_types)

        assert "x" in decorated.required
        assert "y" in decorated.required
        assert "speed" not in decorated.required
        assert "blocking" not in decorated.required


class TestSkill:
    """Tests for the Skill class."""

    def test_skill_to_tool(self, sample_skill_func: Callable[..., str]) -> None:
        """Test conversion to OpenAI tool format."""
        s = skill(sample_skill_func)
        tool = s.to_tool()

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "walk_forward"
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["type"] == "object"

    def test_skill_to_anthropic_tool(self, sample_skill_func: Callable[..., str]) -> None:
        """Test conversion to Anthropic tool format."""
        s = skill(sample_skill_func)
        tool = s.to_anthropic_tool()

        assert tool["name"] == "walk_forward"
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"


class TestSkillRegistry:
    """Tests for the SkillRegistry class."""

    def test_registry_register_skill(self, sample_skill_func: Callable[..., str]) -> None:
        """Test registering a skill."""
        registry = SkillRegistry()
        s = skill(sample_skill_func)

        registered = registry.register(s)

        assert registered.name == "walk_forward"
        assert "walk_forward" in registry
        assert len(registry) == 1

    def test_registry_register_function(self, sample_skill_func: Callable[..., str]) -> None:
        """Test registering a plain function."""
        registry = SkillRegistry()

        registered = registry.register(sample_skill_func)

        assert isinstance(registered, Skill)
        assert "walk_forward" in registry

    def test_registry_get(self, sample_skill_func: Callable[..., str]) -> None:
        """Test getting a skill by name."""
        registry = SkillRegistry()
        registry.register(sample_skill_func)

        s = registry.get("walk_forward")
        assert s is not None
        assert s.name == "walk_forward"

        s = registry.get("nonexistent")
        assert s is None

    def test_registry_getitem(self, sample_skill_func: Callable[..., str]) -> None:
        """Test bracket access."""
        registry = SkillRegistry()
        registry.register(sample_skill_func)

        s = registry["walk_forward"]
        assert s.name == "walk_forward"

        with pytest.raises(KeyError):
            _ = registry["nonexistent"]

    def test_registry_unregister(self, sample_skill_func: Callable[..., str]) -> None:
        """Test unregistering a skill."""
        registry = SkillRegistry()
        registry.register(sample_skill_func)

        removed = registry.unregister("walk_forward")
        assert removed is not None
        assert "walk_forward" not in registry

        removed = registry.unregister("walk_forward")
        assert removed is None

    def test_registry_iteration(self) -> None:
        """Test iterating over skills."""
        registry = SkillRegistry()

        @skill
        def skill1() -> None:
            """Skill 1."""
            pass

        @skill
        def skill2() -> None:
            """Skill 2."""
            pass

        registry.register(skill1)
        registry.register(skill2)

        names = [s.name for s in registry]
        assert "skill1" in names
        assert "skill2" in names

    def test_registry_to_tools(self) -> None:
        """Test converting all skills to OpenAI tools."""
        registry = SkillRegistry()

        @skill
        def skill1() -> None:
            """Skill 1."""
            pass

        @skill
        def skill2() -> None:
            """Skill 2."""
            pass

        registry.register(skill1)
        registry.register(skill2)

        tools = registry.to_tools()
        assert len(tools) == 2
        assert all(t["type"] == "function" for t in tools)

    def test_registry_to_anthropic_tools(self) -> None:
        """Test converting all skills to Anthropic tools."""
        registry = SkillRegistry()

        @skill
        def skill1() -> None:
            """Skill 1."""
            pass

        registry.register(skill1)

        tools = registry.to_anthropic_tools()
        assert len(tools) == 1
        assert "input_schema" in tools[0]

    def test_registry_execute(self, sample_skill_func: Callable[..., str]) -> None:
        """Test executing a skill by name."""
        registry = SkillRegistry()
        registry.register(sample_skill_func)

        result = registry.execute("walk_forward", distance=3.0)
        assert result == "Walked 3.0m"

    def test_registry_execute_unknown_skill(self) -> None:
        """Test executing unknown skill raises error."""
        registry = SkillRegistry()

        with pytest.raises(KeyError, match="Unknown skill"):
            registry.execute("nonexistent")

    def test_registry_filter_by_tag(self) -> None:
        """Test filtering skills by tag."""
        registry = SkillRegistry()

        @skill(tags=["locomotion"])
        def walk() -> None:
            """Walk."""
            pass

        @skill(tags=["locomotion", "fast"])
        def run() -> None:
            """Run."""
            pass

        @skill(tags=["perception"])
        def look() -> None:
            """Look."""
            pass

        registry.register(walk)
        registry.register(run)
        registry.register(look)

        locomotion_skills = registry.filter_by_tag("locomotion")
        assert len(locomotion_skills) == 2

        fast_skills = registry.filter_by_tag("fast")
        assert len(fast_skills) == 1
        assert fast_skills[0].name == "run"

    def test_registry_describe(self) -> None:
        """Test describing all skills."""
        registry = SkillRegistry()

        @skill
        def walk(distance: float = 1.0) -> None:
            """Walk forward."""
            pass

        registry.register(walk)

        description = registry.describe()
        assert "walk" in description
        assert "Walk forward" in description


class TestExtractParameters:
    """Tests for parameter extraction utilities."""

    def test_extract_parameters_no_params(self) -> None:
        """Test extracting from function with no parameters."""
        def func() -> None:
            pass

        params, required = _extract_parameters(func)
        assert params == {}
        assert required == []

    def test_extract_parameters_with_defaults(self) -> None:
        """Test extracting parameters with defaults."""
        def func(x: int, y: int = 10) -> None:
            pass

        params, required = _extract_parameters(func)
        assert "x" in params
        assert "y" in params
        assert "x" in required
        assert "y" not in required
        assert params["y"]["default"] == 10

    def test_extract_parameters_self_excluded(self) -> None:
        """Test that self/cls are excluded."""
        class MyClass:
            def method(self, x: int) -> None:
                pass

            @classmethod
            def classmethod(cls, x: int) -> None:
                pass

        params, _ = _extract_parameters(MyClass.method)
        assert "self" not in params

        params, _ = _extract_parameters(MyClass.classmethod.__func__)
        assert "cls" not in params


class TestPythonTypeToJsonType:
    """Tests for type conversion."""

    def test_basic_types(self) -> None:
        """Test basic Python to JSON type conversion."""
        assert _python_type_to_json_type(str) == "string"
        assert _python_type_to_json_type(int) == "integer"
        assert _python_type_to_json_type(float) == "number"
        assert _python_type_to_json_type(bool) == "boolean"
        assert _python_type_to_json_type(list) == "array"
        assert _python_type_to_json_type(dict) == "object"
        assert _python_type_to_json_type(type(None)) == "null"

    def test_generic_types(self) -> None:
        """Test generic type conversion."""
        from typing import List, Dict

        assert _python_type_to_json_type(List[int]) == "array"
        assert _python_type_to_json_type(Dict[str, int]) == "object"

    def test_unknown_type(self) -> None:
        """Test unknown types default to string."""
        class CustomType:
            pass

        assert _python_type_to_json_type(CustomType) == "string"


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_global_registry(self) -> None:
        """Test getting the global registry."""
        registry = get_global_registry()
        assert isinstance(registry, SkillRegistry)

    def test_get_global_registry_singleton(self) -> None:
        """Test that global registry is a singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        assert registry1 is registry2
