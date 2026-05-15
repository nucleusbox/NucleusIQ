"""Parse Claude Messages assistant text → typed structured objects."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from pydantic import BaseModel


def parse_anthropic_response(
    message: dict[str, Any],
    schema_type: type[BaseModel] | type | dict[str, Any],
) -> Any:
    """Parse assistant ``content`` (JSON string) into *schema_type*.

    Compatible with normalized :class:`~nucleusiq_anthropic._shared.response_models.AssistantMessage`
    dictionaries from :func:`~nucleusiq_anthropic.nb_anthropic.messages.normalize_message_response`.
    """

    from nucleusiq.agents.structured_output.errors import (
        SchemaParseError,
        StructuredOutputError,
    )

    content = message.get("content", "")
    if not content:
        raise StructuredOutputError(
            "Claude returned empty content for structured output",
        )

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise SchemaParseError(
            f"Claude structured output is not valid JSON: {e}",
        ) from e

    if isinstance(schema_type, type) and issubclass(schema_type, BaseModel):
        return schema_type.model_validate(data)

    if dataclasses.is_dataclass(schema_type) and isinstance(schema_type, type):
        return schema_type(**data)

    if isinstance(schema_type, dict) or hasattr(schema_type, "__annotations__"):
        return data

    return data
