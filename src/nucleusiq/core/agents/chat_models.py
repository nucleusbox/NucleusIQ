"""
Typed models for LLM chat messages and tool calls.

These replace raw ``Dict[str, Any]`` throughout the agent subsystem,
providing compile-time type safety and runtime validation via Pydantic.

    ChatMessage      — replaces message dicts in build_messages / LLM calls
    ToolCallRequest  — replaces executor fn_call dicts
    LLMCallKwargs    — typed return for build_call_kwargs
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


class ToolCallRequest(BaseModel):
    """Normalized tool call — used for Executor input and typed messages.

    Provides a flat ``(id, name, arguments)`` view regardless of the
    provider-specific wire format.  Providers convert from this canonical
    representation to their own format in their serialization layer.
    """

    id: str | None = None
    name: str
    arguments: str = "{}"

    @classmethod
    def from_raw(cls, tc: Any) -> ToolCallRequest:
        """Parse from a tool-call dict or SDK object.

        Accepts both the flat canonical format ``{"id", "name", "arguments"}``
        and the legacy nested format ``{"function": {"name", "arguments"}}``.
        """
        if isinstance(tc, cls):
            return tc
        if isinstance(tc, dict):
            tc_id = tc.get("id")
            fn_info = tc.get("function")
            if isinstance(fn_info, dict):
                fn_name = fn_info.get("name", "")
                fn_args = fn_info.get("arguments", "{}")
            else:
                fn_name = tc.get("name", "")
                fn_args = tc.get("arguments", "{}")
        else:
            tc_id = getattr(tc, "id", None)
            fn_info = getattr(tc, "function", None)
            if fn_info is not None:
                fn_name = getattr(fn_info, "name", "") or ""
                fn_args = getattr(fn_info, "arguments", "{}") or "{}"
            else:
                fn_name = getattr(tc, "name", "") or ""
                fn_args = getattr(tc, "arguments", "{}") or "{}"
        return cls(id=tc_id, name=fn_name, arguments=fn_args)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the flat canonical format used across the framework."""
        d: dict[str, Any] = {"name": self.name, "arguments": self.arguments}
        if self.id is not None:
            d["id"] = self.id
        return d


class ChatMessage(BaseModel):
    """Type-safe representation of an LLM chat message.

    Replaces ``Dict[str, Any]`` in internal message lists.  Converts
    cleanly to / from the dict format that LLM providers expect.

    ``content`` may be a plain string **or** a list of content parts
    (multimodal format) when the message carries attachments such as
    images alongside text.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ToolCallRequest] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the dict format expected by LLM providers."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatMessage:
        """Create from a plain dict (e.g. memory context)."""
        raw_tc = d.get("tool_calls")
        tool_calls = [ToolCallRequest.from_raw(tc) for tc in raw_tc] if raw_tc else None
        return cls(
            role=d.get("role", "user"),
            content=d.get("content"),
            tool_calls=tool_calls,
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )


class LLMCallKwargs(TypedDict, total=False):
    """Typed dict for ``agent.llm.call()`` keyword arguments.

    Returned by ``build_call_kwargs`` / ``_build_llm_kwargs`` so callers
    get autocomplete and type checking instead of bare ``Dict[str, Any]``.
    """

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    max_output_tokens: int | None
    response_format: Any


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Serialize a list of ``ChatMessage`` to dicts for LLM providers."""
    return [m.to_dict() for m in messages]
