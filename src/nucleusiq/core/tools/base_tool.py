# File: src/nucleusiq/core/tools/base_tool.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import inspect
from typing import get_origin, get_args


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


class BaseTool(ABC):
    """
    Core abstraction for all "function"-type NucleusIQ tools.

    Every tool must have:
      • name        – unique identifier
      • description – human-readable description
      • version     – optional metadata

    And implement:
      • async initialize()       – optional one-time setup
      • async execute(**kwargs) – perform the call
      • get_spec()               – emit an OpenAI-style function spec
    """

    name: str
    description: str
    version: Optional[str]

    def __init__(
        self,
        *,
        name: str,
        description: str,
        version: Optional[str] = None,
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
        Return a dict matching OpenAI's function spec:
          {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": {...}
          }
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
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "BaseTool":
        """
        Wrap any Python function as a tool with auto-generated spec.
        """
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or ""

        class FunctionTool(BaseTool):
            def __init__(self):
                super().__init__(name=tool_name, description=tool_desc)
                self.fn = fn

            async def initialize(self) -> None:
                # No setup required
                return

            async def execute(self, **kwargs: Any) -> Any:
                return self.fn(**kwargs)

            def get_spec(self) -> Dict[str, Any]:
                sig = inspect.signature(self.fn)
                props: Dict[str, Any] = {}
                required: list[str] = []
                for pname, param in sig.parameters.items():
                    ann = param.annotation if param.annotation is not inspect._empty else str
                    typ = _parse_annotation(ann)
                    props[pname] = {"type": typ, "description": ""}
                    if param.default is inspect._empty:
                        required.append(pname)
                return {
                    "type": "function",
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                        "additionalProperties": False,
                    },
                }

        return FunctionTool()
