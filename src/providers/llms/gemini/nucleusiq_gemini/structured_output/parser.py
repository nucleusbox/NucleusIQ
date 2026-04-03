"""Parse structured Gemini responses into typed Python objects.

Supports:
- Pydantic ``BaseModel`` subclasses → ``model_validate``
- Dataclasses → keyword instantiation
- TypedDict / raw dict → plain dict
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from pydantic import BaseModel


def parse_gemini_response(
    message: dict[str, Any],
    schema_type: type[BaseModel] | type | dict[str, Any],
) -> Any:
    """Parse a Gemini response message into the requested structured type.

    Args:
        message: Message dict with a ``"content"`` key containing JSON.
        schema_type: The target type to parse into.

    Returns:
        An instance of *schema_type* (Pydantic model, dataclass, or dict).

    Raises:
        StructuredOutputError: If the content is empty.
        SchemaParseError: If the content is not valid JSON.
    """
    content = message.get("content", "")
    from nucleusiq.agents.structured_output.errors import (
        SchemaParseError,
        StructuredOutputError,
    )

    if not content:
        raise StructuredOutputError("Gemini returned empty content for structured output")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise SchemaParseError(f"Gemini response is not valid JSON: {e}") from e

    if isinstance(schema_type, type) and issubclass(schema_type, BaseModel):
        return schema_type.model_validate(data)

    if dataclasses.is_dataclass(schema_type) and isinstance(schema_type, type):
        return schema_type(**data)

    if isinstance(schema_type, dict) or hasattr(schema_type, "__annotations__"):
        return data

    return data
