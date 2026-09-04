"""Structured-output handling for OpenAI-compatible servers."""

from .inbound import InboundFormat, normalize_response_format
from .policy import (
    DropPolicy,
    ErrorPolicy,
    PolicyDecision,
    PromptPolicy,
    StructuredOutputPolicy,
    build_policy,
    render_schema_instruction,
)

__all__ = [
    "DropPolicy",
    "ErrorPolicy",
    "InboundFormat",
    "PolicyDecision",
    "PromptPolicy",
    "StructuredOutputPolicy",
    "build_policy",
    "normalize_response_format",
    "render_schema_instruction",
]
