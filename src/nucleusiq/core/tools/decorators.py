"""
``@tool`` decorator — create a ``BaseTool`` from a plain function.

Converts an async (or sync) function into a fully-featured ``BaseTool``
instance with auto-generated JSON Schema spec derived from type hints
and docstring.

Usage::

    from nucleusiq.tools import tool

    @tool("get_weather")
    async def get_weather(city: str, unit: str = "celsius") -> str:
        \"\"\"Get current weather for a city.\"\"\"
        return f"Weather in {city}: 22°{unit[0].upper()}"

    agent = Agent(name="helper", llm=llm, tools=[get_weather])

The decorator supports:

- ``str``, ``int``, ``float``, ``bool``, ``list``, ``dict`` parameter types
- Optional parameters with defaults
- Docstring extraction for tool description
- Per-parameter descriptions from ``:param:`` docstring entries
- Both sync and async functions (sync is wrapped transparently)
- Pydantic ``BaseModel`` as ``args_schema`` for advanced validation
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, Type, get_origin, overload

from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.errors import ToolValidationError

__all__ = ["tool"]


# ====================================================================== #
# Type annotation → JSON Schema mapping                                    #
# ====================================================================== #

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_STR_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _annotation_to_json_type(annotation: Any) -> str:
    """Map a Python type annotation to a JSON Schema type string.

    Handles both real types and string annotations (from
    ``from __future__ import annotations``).
    """
    if annotation is inspect.Parameter.empty:
        return "string"
    if isinstance(annotation, str):
        return _STR_TYPE_MAP.get(annotation, "string")
    origin = get_origin(annotation) or annotation
    return _TYPE_MAP.get(origin, "string")


# ====================================================================== #
# Docstring → per-parameter description extraction                         #
# ====================================================================== #


def _parse_param_docs(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Extract tool description and per-parameter docs from a docstring.

    Supports ``:param name: description`` and ``Args:`` (Google) styles.

    Returns:
        ``(tool_description, {param_name: param_description})``
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().splitlines()
    description_lines: list[str] = []
    param_docs: dict[str, str] = {}
    in_args_section = False

    for line in lines:
        stripped = line.strip()

        # :param name: description
        if stripped.startswith(":param "):
            rest = stripped[7:]
            colon_idx = rest.find(":")
            if colon_idx > 0:
                pname = rest[:colon_idx].strip()
                pdesc = rest[colon_idx + 1 :].strip()
                param_docs[pname] = pdesc
            continue

        # Google-style "Args:" section
        if stripped == "Args:":
            in_args_section = True
            continue

        if in_args_section:
            if stripped and not stripped.startswith(
                ("Returns", "Raises", "Example", "Note")
            ):
                colon_idx = stripped.find(":")
                if colon_idx > 0 and not stripped[0].isspace():
                    pname = stripped[:colon_idx].strip().split("(")[0].strip()
                    pdesc = stripped[colon_idx + 1 :].strip()
                    param_docs[pname] = pdesc
                    continue
            if not stripped or stripped.startswith(
                ("Returns", "Raises", "Example", "Note")
            ):
                in_args_section = False

        if not in_args_section and not param_docs:
            description_lines.append(stripped)

    description = " ".join(description_lines).strip()
    return description, param_docs


# ====================================================================== #
# Spec builder                                                             #
# ====================================================================== #


def _build_spec_from_signature(
    fn: Callable[..., Any],
    tool_name: str,
    tool_description: str,
    param_docs: dict[str, str],
) -> dict[str, Any]:
    """Build a JSON Schema tool spec from a function's signature."""
    sig = inspect.signature(fn)

    try:
        from typing import get_type_hints

        resolved_hints = get_type_hints(fn)
    except Exception:
        resolved_hints = {}

    properties: Dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue

        annotation = resolved_hints.get(pname, param.annotation)
        json_type = _annotation_to_json_type(annotation)
        prop: Dict[str, Any] = {
            "type": json_type,
            "description": param_docs.get(pname, ""),
        }

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(pname)

        properties[pname] = prop

    return {
        "name": tool_name,
        "description": tool_description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ====================================================================== #
# DecoratedTool — the concrete BaseTool produced by @tool                  #
# ====================================================================== #


class DecoratedTool(BaseTool):
    """``BaseTool`` produced by the ``@tool`` decorator.

    Wraps a plain function with auto-generated spec and async execution.
    Instances are callable (delegates to ``execute``).
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        tool_name: str,
        tool_description: str,
        args_schema: Type[Any] | None = None,
    ) -> None:
        super().__init__(name=tool_name, description=tool_description)
        self._fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)
        self._args_schema = args_schema

        _, param_docs = _parse_param_docs(fn.__doc__)
        if args_schema is not None:
            self._spec = self._build_pydantic_spec(tool_name, tool_description)
        else:
            self._spec = _build_spec_from_signature(
                fn, tool_name, tool_description, param_docs
            )

    # ---- BaseTool contract ------------------------------------------ #

    async def initialize(self) -> None:
        pass

    async def execute(self, **kwargs: Any) -> Any:
        if self._args_schema is not None:
            validated = self._args_schema(**kwargs)
            kwargs = validated.model_dump()

        if self._is_async:
            return await self._fn(**kwargs)
        return self._fn(**kwargs)

    def get_spec(self) -> Dict[str, Any]:
        return self._spec

    # ---- Pydantic schema builder ------------------------------------ #

    def _build_pydantic_spec(
        self, tool_name: str, tool_description: str
    ) -> dict[str, Any]:
        from nucleusiq.tools.base_tool import _pydantic_model_to_json_schema

        assert self._args_schema is not None, (
            "_args_schema must be set for Pydantic spec"
        )
        parameters = _pydantic_model_to_json_schema(self._args_schema)
        return {
            "name": tool_name,
            "description": tool_description,
            "parameters": parameters,
        }


# ====================================================================== #
# @tool decorator                                                          #
# ====================================================================== #


@overload
def tool(fn: Callable[..., Any]) -> DecoratedTool: ...


@overload
def tool(  # pyrefly: ignore[inconsistent-overload]
    name: str | None = None,
    *,
    description: str | None = None,
    args_schema: Type[Any] | None = None,
) -> Callable[[Callable[..., Any]], DecoratedTool]: ...


def tool(
    fn: Callable[..., Any] | str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: Type[Any] | None = None,
) -> DecoratedTool | Callable[[Callable[..., Any]], DecoratedTool]:
    """Create a ``BaseTool`` from a function via decorator.

    Can be used in three forms::

        # 1. Bare decorator (no arguments)
        @tool
        async def my_tool(x: int) -> str: ...


        # 2. With name only
        @tool("custom_name")
        async def my_tool(x: int) -> str: ...


        # 3. With keyword arguments
        @tool(name="custom_name", description="Does X")
        async def my_tool(x: int) -> str: ...

    Parameters
    ----------
    fn : callable or str or None
        The function to wrap (bare decorator) or the tool name (string).
    name : str | None
        Tool name override.  Defaults to ``fn.__name__``.
    description : str | None
        Description override.  Defaults to first line of docstring.
    args_schema : type[BaseModel] | None
        Pydantic model for parameter validation and schema generation.

    Returns
    -------
    DecoratedTool or decorator
        If called with a function directly, returns a ``DecoratedTool``.
        Otherwise returns a decorator that produces a ``DecoratedTool``.

    Raises
    ------
    ToolValidationError
        If the decorated object is not callable.
    """
    # Case 1: @tool  (bare, no parentheses)
    if callable(fn) and not isinstance(fn, str):
        _validate_function(fn)
        doc_desc, _ = _parse_param_docs(fn.__doc__)
        return DecoratedTool(
            fn,
            tool_name=fn.__name__,
            tool_description=doc_desc,
            args_schema=args_schema,
        )

    # Case 2: @tool("name") or @tool(name="name", ...)
    resolved_name = fn if isinstance(fn, str) else name

    def _decorator(func: Callable[..., Any]) -> DecoratedTool:
        _validate_function(func)
        doc_desc, _ = _parse_param_docs(func.__doc__)
        final_name = resolved_name or func.__name__
        final_desc = description or doc_desc
        return DecoratedTool(
            func,
            tool_name=final_name,
            tool_description=final_desc,
            args_schema=args_schema,
        )

    return _decorator


def _validate_function(fn: Any) -> None:
    """Fail fast if the decorated object is not a valid function."""
    if not callable(fn):
        raise ToolValidationError(
            f"@tool can only decorate callable objects, got {type(fn).__name__}"
        )
