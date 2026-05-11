"""Parse structured Ollama chat responses."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from pydantic import BaseModel


def parse_response(
    message: dict[str, Any],
    schema_type: type[BaseModel] | type | dict[str, Any],
) -> Any:
    """Parse assistant *message* JSON ``content`` into *schema_type*."""
    from nucleusiq.agents.structured_output.errors import (
        SchemaParseError,
        StructuredOutputError,
    )

    content = message.get("content", "")
    if not content:
        raise StructuredOutputError("LLM returned empty content for structured output")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise SchemaParseError(f"LLM response is not valid JSON: {e}") from e

    if isinstance(schema_type, type) and issubclass(schema_type, BaseModel):
        return schema_type.model_validate(data)

    if dataclasses.is_dataclass(schema_type) and isinstance(schema_type, type):
        return schema_type(**data)

    if isinstance(schema_type, dict) or hasattr(schema_type, "__annotations__"):
        return data

    return data
