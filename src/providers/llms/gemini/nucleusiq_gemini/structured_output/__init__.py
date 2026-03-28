"""Structured output support for Gemini API.

Converts Python types (Pydantic models, dataclasses, TypedDict) into
Gemini's ``response_mime_type`` + ``response_json_schema`` config and
parses responses back into typed instances.

Usage::

    from nucleusiq_gemini.structured_output import (
        build_gemini_response_format,
        parse_gemini_response,
    )

    fmt = build_gemini_response_format(MyPydanticModel)
    # → {"response_mime_type": "application/json", "response_json_schema": {...}}

    instance = parse_gemini_response(message_dict, MyPydanticModel)
    # → MyPydanticModel(...)
"""

from nucleusiq_gemini.structured_output.builder import build_gemini_response_format
from nucleusiq_gemini.structured_output.parser import parse_gemini_response

__all__ = ["build_gemini_response_format", "parse_gemini_response"]
