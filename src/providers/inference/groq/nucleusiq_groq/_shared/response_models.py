"""Normalised chat completion responses for :class:`~nucleusiq_groq.nb_groq.base.BaseGroq`."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    model_config = ConfigDict(populate_by_name=True)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        return d


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


class _Choice(BaseModel):
    message: AssistantMessage


class GroqLLMResponse(BaseModel):
    choices: list[_Choice]
    usage: UsageInfo | None = None
    model: str | None = None
    response_id: str | None = None
