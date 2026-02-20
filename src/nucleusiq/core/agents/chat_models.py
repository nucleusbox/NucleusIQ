"""
Typed models for LLM chat messages and tool calls.

These replace raw ``Dict[str, Any]`` throughout the agent subsystem,
providing compile-time type safety and runtime validation via Pydantic.

    ChatMessage      — replaces message dicts in build_messages / LLM calls
    ToolCallRequest  — replaces executor fn_call dicts
    LLMCallKwargs    — typed return for build_call_kwargs
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


class ToolCallRequest(BaseModel):
    """Normalized tool call — used for Executor input and typed messages.

    Provides a flat ``(id, name, arguments)`` view regardless of the
    provider-specific wire format (OpenAI nests under ``function``).
    """

    id: Optional[str] = None
    name: str
    arguments: str = "{}"

    @classmethod
    def from_raw(cls, tc: Any) -> ToolCallRequest:
        """Parse from OpenAI-style tool_call (dict or SDK object)."""
        if isinstance(tc, cls):
            return tc
        if isinstance(tc, dict):
            tc_id = tc.get("id")
            fn_info = tc.get("function", {})
            if isinstance(fn_info, dict):
                fn_name = fn_info.get("name", "")
                fn_args = fn_info.get("arguments", "{}")
            else:
                fn_name = ""
                fn_args = "{}"
        else:
            tc_id = getattr(tc, "id", None)
            fn_info = getattr(tc, "function", None)
            fn_name = getattr(fn_info, "name", "") if fn_info else ""
            fn_args = getattr(fn_info, "arguments", "{}") if fn_info else "{}"
        return cls(id=tc_id, name=fn_name, arguments=fn_args)

    def to_openai_dict(self) -> Dict[str, Any]:
        """Serialize to OpenAI ``tool_calls[]`` wire format."""
        d: Dict[str, Any] = {
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }
        if self.id is not None:
            d["id"] = self.id
        return d


class ChatMessage(BaseModel):
    """Type-safe representation of an LLM chat message.

    Replaces ``Dict[str, Any]`` in internal message lists.  Converts
    cleanly to / from the dict format that LLM providers expect.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallRequest]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the dict format expected by LLM providers."""
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_openai_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ChatMessage:
        """Create from a plain dict (e.g. memory context)."""
        raw_tc = d.get("tool_calls")
        tool_calls = (
            [ToolCallRequest.from_raw(tc) for tc in raw_tc]
            if raw_tc
            else None
        )
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
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]]
    max_tokens: Optional[int]
    response_format: Any


def messages_to_dicts(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Serialize a list of ``ChatMessage`` to dicts for LLM providers."""
    return [m.to_dict() for m in messages]
