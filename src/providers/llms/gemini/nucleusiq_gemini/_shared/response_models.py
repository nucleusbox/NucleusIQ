"""Lightweight response wrappers that match the BaseLLM contract.

Normalises Gemini API responses so that callers always receive the
same ``GeminiLLMResponse`` shape regardless of which Gemini model or
feature was used.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCallFunction(BaseModel):
    """Function metadata inside a tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """A single tool call returned by the model."""

    id: str
    type: str = "function"
    function: ToolCallFunction


class AssistantMessage(BaseModel):
    """Typed replacement for the raw ``Dict[str, Any]`` message."""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    native_outputs: list[dict[str, Any]] | None = Field(
        default=None, alias="_native_outputs"
    )

    model_config = ConfigDict(populate_by_name=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (the BaseLLM contract format)."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        if self.native_outputs:
            d["_native_outputs"] = self.native_outputs
        return d


class UsageInfo(BaseModel):
    """Token usage statistics from a Gemini API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    thoughts_tokens: int = 0
    cached_tokens: int = 0


class _Choice(BaseModel):
    """Minimal wrapper so we match BaseLLM expectation."""

    message: AssistantMessage


class GeminiLLMResponse(BaseModel):
    """Normalised response from the Gemini API."""

    choices: list[_Choice]
    usage: UsageInfo | None = None
    response_id: str | None = None
    model: str | None = None
