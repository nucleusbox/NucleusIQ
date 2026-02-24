# File: src/nucleusiq/core/tools/base_tool.py
import inspect
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Type,
    get_origin,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

try:
    from pydantic import BaseModel as _BaseModel

    PYDANTIC_AVAILABLE = True
    BaseModel = _BaseModel
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None


def _parse_annotation(annotation: Any) -> str:
    """
    Map Python type annotations to JSON Schema basic types.
    Defaults to 'string' if unrecognized.
    """
    origin = get_origin(annotation) or annotation
    mapping = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        dict: "object",
        list: "array",
    }
    return mapping.get(origin, "string")


def _pydantic_model_to_json_schema(model: Type[Any]) -> Dict[str, Any]:
    """
    Convert a Pydantic BaseModel to generic JSON Schema.

    This returns standard JSON Schema (not LLM-specific).
    The LLM provider will convert this to its own format.

    Args:
        model: Pydantic BaseModel class

    Returns:
        Generic JSON Schema dict
    """
    if not PYDANTIC_AVAILABLE or BaseModel is None:
        raise ImportError(
            "Pydantic is required for schema-based tools. Install with: pip install pydantic"
        )

    if not issubclass(model, BaseModel):
        raise TypeError(f"Expected Pydantic BaseModel, got {type(model)}")

    # Get JSON schema from Pydantic model
    json_schema = model.model_json_schema()

    # Extract properties and required fields
    properties = json_schema.get("properties", {})
    required = json_schema.get("required", [])

    # Return generic JSON Schema (LLM-agnostic)
    generic_schema = {
        "type": "object",
        "properties": {},
        "required": required,
    }

    # Process each property
    for prop_name, prop_schema in properties.items():
        # Handle different schema formats
        if "$ref" in prop_schema:
            # Reference to a definition - resolve it
            ref_path = prop_schema["$ref"].split("/")[-1]
            definitions = json_schema.get("$defs", {}) or json_schema.get(
                "definitions", {}
            )
            if ref_path in definitions:
                prop_schema = definitions[ref_path]

        # Extract type information
        prop_type = prop_schema.get("type", "string")

        # Handle Literal types (enum)
        if "enum" in prop_schema:
            generic_schema["properties"][prop_name] = {
                "type": prop_type,
                "enum": prop_schema["enum"],
                "description": prop_schema.get("description", ""),
            }
        else:
            # Standard property
            prop = {
                "type": prop_type,
                "description": prop_schema.get("description", ""),
            }

            # Add default if present
            if "default" in prop_schema:
                prop["default"] = prop_schema["default"]

            # Add constraints
            for constraint in ["minimum", "maximum", "minLength", "maxLength"]:
                if constraint in prop_schema:
                    prop[constraint] = prop_schema[constraint]

            generic_schema["properties"][prop_name] = prop

    return generic_schema


class BaseTool(ABC):
    """
    Core abstraction for generic function-calling tools in NucleusIQ.

    BaseTool is for tools that work with ANY LLM via function calling.
    These tools:
    - Generate generic tool specifications (LLM-agnostic JSON Schema)
    - Work with any LLM that supports function calling (OpenAI, Anthropic, Gemini, etc.)
    - Use function calling protocol (not LLM-native features)
    - LLM providers convert BaseTool specs to their own format

    For LLM-specific built-in tools (like OpenAI's search, code interpreter),
    use provider-specific tool factories (e.g., OpenAITool).

    Every tool must have:
      • name        – unique identifier
      • description – human-readable description
      • version     – optional metadata

    And implement:
      • async initialize()       – optional one-time setup
      • async execute(**kwargs) – perform the call
      • get_spec()               – emit a generic tool spec (LLM-agnostic)
    """

    name: str
    description: str
    version: str | None

    def __init__(
        self,
        *,
        name: str,
        description: str,
        version: str | None = None,
    ):
        self.name = name
        self.description = description
        self.version = version

    @abstractmethod
    async def initialize(self) -> None:
        """
        Optional setup before execute().
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """
        Run the tool’s logic with provided keyword arguments.
        """
        pass

    @abstractmethod
    def get_spec(self) -> Dict[str, Any]:
        """
        Return a generic tool specification (LLM-agnostic).

        This should return a dict with:
          {
            "name": name,
            "description": description,
            "parameters": {...}  # Generic JSON Schema
          }

        The LLM provider will convert this to its own format.
        """
        pass

    def shutdown(self) -> None:
        """
        Optional cleanup (e.g. close connections).
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        data = {"name": self.name, "description": self.description}
        if self.version:
            data["version"] = self.version
        return data

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        args_schema: Type[Any] | None = None,
    ) -> "BaseTool":
        """
        Wrap any Python function as a tool with auto-generated spec.

        Args:
            fn: Python function to wrap
            name: Optional custom tool name (defaults to function name)
            description: Optional tool description (defaults to function docstring)
            args_schema: Optional Pydantic BaseModel for parameter schema.
                        If provided, uses Pydantic schema instead of function signature.

        Returns:
            BaseTool instance wrapping the function
        """
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or ""

        class FunctionTool(BaseTool):
            def __init__(self):
                super().__init__(name=tool_name, description=tool_desc)
                self.fn = fn
                self.args_schema = args_schema

            async def initialize(self) -> None:
                # No setup required
                return

            async def execute(self, **kwargs: Any) -> Any:
                # If args_schema is provided, validate and instantiate the model
                if self.args_schema:
                    # Validate and create Pydantic model instance
                    model_instance = self.args_schema(**kwargs)
                    # Convert to dict and pass to function
                    return self.fn(**model_instance.model_dump())
                else:
                    # Direct function call
                    return self.fn(**kwargs)

            def get_spec(self) -> Dict[str, Any]:
                # If Pydantic schema is provided, use it
                if self.args_schema:
                    parameters = _pydantic_model_to_json_schema(self.args_schema)
                else:
                    # Generate from function signature
                    sig = inspect.signature(self.fn)
                    props: Dict[str, Any] = {}
                    required: list[str] = []
                    for pname, param in sig.parameters.items():
                        ann = (
                            param.annotation
                            if param.annotation is not inspect._empty
                            else str
                        )
                        typ = _parse_annotation(ann)
                        props[pname] = {"type": typ, "description": ""}
                        if param.default is inspect._empty:
                            required.append(pname)
                    parameters = {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    }

                # Return generic spec (LLM-agnostic)
                # LLM provider will convert to its own format
                return {
                    "name": self.name,
                    "description": self.description,
                    "parameters": parameters,
                }

        return FunctionTool()
