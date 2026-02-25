"""Lightweight response wrappers that match the BaseLLM contract.

Used by both the Chat Completions and Responses API backends so that
callers always receive the same ``_LLMResponse`` shape regardless of
which OpenAI API was called.
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


class _Choice(BaseModel):
    """Minimal wrapper so we match BaseLLM expectation."""

    message: AssistantMessage


class _LLMResponse(BaseModel):
    choices: list[_Choice]
