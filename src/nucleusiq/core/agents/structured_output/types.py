# src/nucleusiq/agents/structured_output/types.py
"""
Type definitions for NucleusIQ Structured Output.

NucleusIQ Design Philosophy:
- AUTO mode is the default - framework intelligently selects best strategy
- Users CAN choose specific modes, but we validate they're implemented
- Clear error messages guide users when they try unsupported features
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Set, Type, TypeVar, Union

from pydantic import BaseModel

# Type variable for schema types
T = TypeVar("T")

# Supported schema types
SchemaType = Union[
    Type[BaseModel],  # Pydantic models
    Type,  # Dataclasses, TypedDicts, or any class with annotations
    Dict[str, Any],  # Raw JSON Schema
]


class OutputMode(str, Enum):
    """
    How NucleusIQ obtains structured output from the LLM.

    Similar to execution modes (DIRECT, STANDARD, AUTONOMOUS),
    output modes control how structured data is extracted.

    Implementation Status:
    ----------------------
    ✅ AUTO   - Implemented (auto-selects NATIVE for OpenAI)
    ✅ NATIVE - Implemented (OpenAI json_schema, response_format)
    🚧 TOOL   - Coming Soon (use function calling as extraction method)
    🚧 PROMPT - Coming Soon (prompt-based JSON extraction)

    Usage:
    ------
    # Recommended: Let NucleusIQ choose
    agent = Agent(response_format=MyModel)  # Uses AUTO → NATIVE

    # Explicit (only use if you need specific behavior)
    agent = Agent(response_format=OutputSchema(
        schema=MyModel,
        mode=OutputMode.NATIVE
    ))
    """

    AUTO = "auto"
    """Let NucleusIQ choose the best method based on model capabilities."""

    NATIVE = "native"
    """Use provider's native structured output (e.g., OpenAI response_format)."""

    TOOL = "tool"
    """Use tool/function calling to obtain structured output. (Coming Soon)"""

    PROMPT = "prompt"
    """Use prompt engineering with JSON instructions. (Coming Soon)"""

    @classmethod
    def implemented_modes(cls) -> Set[OutputMode]:
        """Return set of currently implemented modes."""
        return {cls.AUTO, cls.NATIVE}

    @classmethod
    def is_implemented(cls, mode: OutputMode) -> bool:
        """Check if a mode is implemented."""
        return mode in cls.implemented_modes()

    @classmethod
    def validate_mode(cls, mode: OutputMode) -> None:
        """
        Validate that a mode is implemented.

        Raises:
            NotImplementedError: If mode is not yet implemented with helpful message
        """
        if not cls.is_implemented(mode):
            implemented = ", ".join(m.value for m in cls.implemented_modes())
            raise NotImplementedError(
                f"OutputMode.{mode.value.upper()} is not yet implemented.\n"
                f"\n"
                f"Currently implemented modes: {implemented}\n"
                f"\n"
                f"Options:\n"
                f"  1. Use AUTO (recommended): response_format=MyModel\n"
                f"  2. Use NATIVE: OutputSchema(schema=MyModel, mode=OutputMode.NATIVE)\n"
                f"\n"
                f"TOOL and PROMPT modes are coming in a future release."
            )


class ErrorHandling(str, Enum):
    """
    How to handle validation/parsing errors.
    """

    RETRY = "retry"
    """Send error feedback to LLM and retry (default)."""

    RAISE = "raise"
    """Raise exception immediately."""

    FALLBACK = "fallback"
    """Return raw text if structured parsing fails."""
