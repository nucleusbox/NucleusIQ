"""Structured output support for OpenAI APIs.

Converts Python types (Pydantic models, dataclasses, TypedDict) into
OpenAI's ``response_format`` JSON schema and parses responses back into
typed instances.

Usage::

    from nucleusiq_openai.structured_output import build_response_format, parse_response

    fmt = build_response_format(MyPydanticModel)
    # → {"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}

    instance = parse_response(message_dict, MyPydanticModel)
    # → MyPydanticModel(...)
"""

from nucleusiq_openai.structured_output.builder import build_response_format
from nucleusiq_openai.structured_output.parser import parse_response

__all__ = ["build_response_format", "parse_response"]
