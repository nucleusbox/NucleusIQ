"""Tests for the ``@tool`` decorator.

Covers:
- All three decorator forms (bare, name-only, keyword args)
- Sync and async function wrapping
- Type inference from annotations → JSON Schema
- Optional parameters with defaults
- Docstring extraction (first-line, :param:, Google-style Args:)
- Pydantic ``args_schema`` validation
- Edge cases: no docstring, no type hints, no params
- ``BaseTool`` contract compliance (get_spec, execute, initialize)
- Integration with ``BaseTool.from_function`` parity
"""

from __future__ import annotations

import pytest
from nucleusiq.tools import tool
from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.decorators import DecoratedTool, _parse_param_docs
from pydantic import BaseModel, Field

# ====================================================================== #
# Fixtures — sample functions                                              #
# ====================================================================== #


async def _async_greet(name: str, greeting: str = "Hello") -> str:
    """Greet someone by name.

    :param name: The person's name
    :param greeting: Greeting word
    """
    return f"{greeting}, {name}!"


def _sync_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _no_docstring(x: str) -> str:
    return x.upper()


def _no_annotations(x, y="default"):
    """Do something with x and y."""
    return f"{x}-{y}"


def _no_params() -> str:
    """Return a fixed string."""
    return "fixed"


class WeatherArgs(BaseModel):
    """Validated weather parameters."""

    city: str = Field(..., description="City name")
    unit: str = Field("celsius", description="Temperature unit")


# ====================================================================== #
# Decorator forms                                                          #
# ====================================================================== #


class TestDecoratorForms:
    """All three decorator usage forms produce a valid DecoratedTool."""

    def test_bare_decorator(self):
        """@tool without parentheses."""

        @tool
        async def my_tool(x: str) -> str:
            """Do X."""
            return x

        assert isinstance(my_tool, DecoratedTool)
        assert isinstance(my_tool, BaseTool)
        assert my_tool.name == "my_tool"
        assert my_tool.description == "Do X."

    def test_name_only_decorator(self):
        """@tool("custom_name")."""

        @tool("custom_name")
        async def my_tool(x: str) -> str:
            """Do X."""
            return x

        assert my_tool.name == "custom_name"
        assert my_tool.description == "Do X."

    def test_keyword_args_decorator(self):
        """@tool(name="...", description="...")."""

        @tool(name="custom", description="Custom description")
        async def my_tool(x: str) -> str:
            """Original docstring."""
            return x

        assert my_tool.name == "custom"
        assert my_tool.description == "Custom description"

    def test_name_kwarg_without_positional(self):
        """@tool(name="...")."""

        @tool(name="named_tool")
        async def my_tool(x: str) -> str:
            """Doc."""
            return x

        assert my_tool.name == "named_tool"

    def test_empty_parentheses(self):
        """@tool() — defaults to function name and docstring."""

        @tool()
        async def fetch_data(url: str) -> str:
            """Fetch data from URL."""
            return url

        assert fetch_data.name == "fetch_data"
        assert fetch_data.description == "Fetch data from URL."


# ====================================================================== #
# Async / sync execution                                                   #
# ====================================================================== #


class TestExecution:
    """Both async and sync functions execute correctly."""

    @pytest.mark.asyncio
    async def test_async_execution(self):
        decorated = tool("greet")(_async_greet)
        result = await decorated.execute(name="World")
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_async_with_override(self):
        decorated = tool("greet")(_async_greet)
        result = await decorated.execute(name="World", greeting="Hi")
        assert result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_sync_wrapped_as_async(self):
        decorated = tool("add")(_sync_add)
        result = await decorated.execute(a=3, b=4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_no_params_function(self):
        decorated = tool("fixed")(_no_params)
        result = await decorated.execute()
        assert result == "fixed"

    @pytest.mark.asyncio
    async def test_initialize_is_noop(self):
        decorated = tool("greet")(_async_greet)
        await decorated.initialize()  # should not raise


# ====================================================================== #
# Spec generation from type hints                                          #
# ====================================================================== #


class TestSpecGeneration:
    """get_spec() produces correct JSON Schema from type hints."""

    def test_basic_spec_structure(self):
        decorated = tool("greet")(_async_greet)
        spec = decorated.get_spec()

        assert spec["name"] == "greet"
        assert spec["description"] != ""
        assert "parameters" in spec
        assert spec["parameters"]["type"] == "object"
        assert "properties" in spec["parameters"]
        assert "required" in spec["parameters"]

    def test_required_and_optional_params(self):
        decorated = tool("greet")(_async_greet)
        spec = decorated.get_spec()
        params = spec["parameters"]

        assert "name" in params["required"]
        assert "greeting" not in params["required"]

    def test_param_types(self):
        decorated = tool("add")(_sync_add)
        spec = decorated.get_spec()
        props = spec["parameters"]["properties"]

        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "integer"

    def test_default_values_in_spec(self):
        decorated = tool("greet")(_async_greet)
        spec = decorated.get_spec()
        props = spec["parameters"]["properties"]

        assert props["greeting"]["default"] == "Hello"

    def test_no_annotations_defaults_to_string(self):
        decorated = tool("something")(_no_annotations)
        spec = decorated.get_spec()
        props = spec["parameters"]["properties"]

        assert props["x"]["type"] == "string"
        assert props["y"]["type"] == "string"

    def test_no_params_empty_properties(self):
        decorated = tool("fixed")(_no_params)
        spec = decorated.get_spec()

        assert spec["parameters"]["properties"] == {}
        assert spec["parameters"]["required"] == []

    def test_all_supported_types(self):

        @tool("typed")
        def fn(s: str, i: int, f: float, b: bool, items: list, d: dict) -> str:
            """All types."""
            return ""

        spec = fn.get_spec()
        props = spec["parameters"]["properties"]

        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["items"]["type"] == "array"
        assert props["d"]["type"] == "object"


# ====================================================================== #
# Docstring parsing                                                        #
# ====================================================================== #


class TestDocstringParsing:
    """Docstring → description + per-param docs."""

    def test_first_line_as_description(self):
        desc, _ = _parse_param_docs("Get weather for a city.\n\nMore details.")
        assert "Get weather for a city." in desc
        assert "More details." in desc

    def test_param_style_docs(self):
        _, params = _parse_param_docs(
            "Do something.\n\n:param city: The city name\n:param unit: Temp unit"
        )
        assert params["city"] == "The city name"
        assert params["unit"] == "Temp unit"

    def test_google_style_args(self):
        docstring = """Do something.

        Args:
            city: The city name
            unit: Temperature unit
        """
        _, params = _parse_param_docs(docstring)
        assert params["city"] == "The city name"
        assert params["unit"] == "Temperature unit"

    def test_no_docstring(self):
        desc, params = _parse_param_docs(None)
        assert desc == ""
        assert params == {}

    def test_empty_docstring(self):
        desc, params = _parse_param_docs("")
        assert desc == ""
        assert params == {}

    def test_param_descriptions_in_spec(self):
        decorated = tool("greet")(_async_greet)
        spec = decorated.get_spec()
        props = spec["parameters"]["properties"]

        assert props["name"]["description"] == "The person's name"
        assert props["greeting"]["description"] == "Greeting word"

    def test_no_docstring_empty_description(self):
        decorated = tool("upper")(_no_docstring)
        assert decorated.description == ""


# ====================================================================== #
# Pydantic args_schema                                                     #
# ====================================================================== #


class TestPydanticSchema:
    """args_schema validates inputs and generates schema."""

    def test_schema_from_pydantic(self):

        @tool(name="weather", args_schema=WeatherArgs)
        async def get_weather(city: str, unit: str = "celsius") -> str:
            """Get weather."""
            return f"{city}: 22°"

        spec = get_weather.get_spec()
        props = spec["parameters"]["properties"]

        assert "city" in props
        assert "unit" in props
        assert "city" in spec["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_schema_validates_input(self):

        @tool(name="weather", args_schema=WeatherArgs)
        async def get_weather(city: str, unit: str = "celsius") -> str:
            """Get weather."""
            return f"{city}: {unit}"

        result = await get_weather.execute(city="London")
        assert "London" in result
        assert "celsius" in result

    @pytest.mark.asyncio
    async def test_schema_rejects_invalid_input(self):
        from pydantic import ValidationError

        class StrictArgs(BaseModel):
            count: int = Field(..., ge=1)

        @tool(name="strict", args_schema=StrictArgs)
        async def strict_fn(count: int) -> int:
            """Strict."""
            return count

        with pytest.raises(ValidationError):
            await strict_fn.execute(count=-1)


# ====================================================================== #
# BaseTool contract compliance                                             #
# ====================================================================== #


class TestBaseToolContract:
    """DecoratedTool is a valid BaseTool for all framework interactions."""

    def test_is_base_tool_instance(self):
        decorated = tool("greet")(_async_greet)
        assert isinstance(decorated, BaseTool)

    def test_has_name_and_description(self):
        decorated = tool("greet")(_async_greet)
        assert hasattr(decorated, "name")
        assert hasattr(decorated, "description")
        assert isinstance(decorated.name, str)
        assert isinstance(decorated.description, str)

    def test_to_dict(self):
        decorated = tool("greet")(_async_greet)
        d = decorated.to_dict()
        assert d["name"] == "greet"
        assert "description" in d

    def test_shutdown_is_noop(self):
        decorated = tool("greet")(_async_greet)
        decorated.shutdown()  # should not raise

    def test_spec_matches_base_tool_from_function(self):
        """@tool spec should be structurally compatible with BaseTool.from_function."""
        legacy = BaseTool.from_function(_sync_add, name="add")
        decorated = tool("add")(_sync_add)

        legacy_spec = legacy.get_spec()
        decorated_spec = decorated.get_spec()

        assert legacy_spec["name"] == decorated_spec["name"]
        assert set(legacy_spec["parameters"]["required"]) == set(
            decorated_spec["parameters"]["required"]
        )
        for key in legacy_spec["parameters"]["properties"]:
            assert key in decorated_spec["parameters"]["properties"]


# ====================================================================== #
# Edge cases                                                               #
# ====================================================================== #


class TestEdgeCases:
    """Unusual inputs handled gracefully."""

    def test_non_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="callable"):
            tool("bad")(42)

    def test_lambda_function(self):
        decorated = tool("double")(lambda x: x * 2)
        assert decorated.name == "double"
        spec = decorated.get_spec()
        assert "x" in spec["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_lambda_execution(self):
        decorated = tool("double")(lambda x: x * 2)
        result = await decorated.execute(x=5)
        assert result == 10

    def test_class_method_like_function(self):
        """Functions with 'self' param should skip it in spec."""

        def method(self, x: str) -> str:
            return x

        decorated = tool("method")(method)
        spec = decorated.get_spec()
        assert "self" not in spec["parameters"]["properties"]
        assert "x" in spec["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_function_raising_exception(self):

        @tool("failing")
        async def failing(x: str) -> str:
            """Fail."""
            raise ValueError("intentional")

        with pytest.raises(ValueError, match="intentional"):
            await failing.execute(x="test")

    def test_version_is_none(self):
        decorated = tool("greet")(_async_greet)
        assert decorated.version is None
