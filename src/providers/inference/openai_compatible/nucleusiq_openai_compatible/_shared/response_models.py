"""Response normalization.

OpenAI-compatible servers agree on the wire schema but differ in what they
actually populate: llama.cpp may omit ``usage``, some gateways drop
``system_fingerprint``, vLLM returns ``id`` but no ``x-request-id`` header.
Everything downstream reads a :class:`NormalizedResponse` so those gaps are
handled once, here, rather than with ``getattr`` chains at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..capabilities import REASONING_FIELDS

__all__ = [
    "NormalizedChoice",
    "NormalizedMessage",
    "NormalizedResponse",
    "NormalizedToolCall",
    "extract_reasoning",
    "normalize_response",
]


@dataclass(frozen=True, slots=True)
class NormalizedToolCall:
    """One tool call requested by the model.

    Always locally executed: a self-hosted inference server has no
    server-side tool runtime.
    """

    id: str
    name: str
    arguments: str

    def to_wire(self) -> dict[str, Any]:
        """Return the OpenAI-shaped dict for echoing back in message history."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

    @property
    def type(self) -> str:
        """Always ``"function"``; there is no server-side tool execution."""
        return "function"

    @property
    def function(self) -> _FunctionView:
        """OpenAI-shaped ``.function.name`` / ``.function.arguments`` access."""
        return _FunctionView(self.name, self.arguments)


@dataclass(frozen=True, slots=True)
class _FunctionView:
    """The ``.function`` attribute the framework reads off a tool call."""

    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    """The ``.message`` the framework reads off a choice.

    Mirrors ``ChatCompletionMessage``: the agent modes reach for
    ``msg.content``, ``msg.tool_calls`` and ``msg.refusal``, and require
    ``tool_calls`` to be a genuine ``list``.
    """

    content: str | None
    tool_calls: list[NormalizedToolCall] | None
    refusal: str | None = None
    reasoning: str | None = None
    role: str = "assistant"


@dataclass(frozen=True, slots=True)
class NormalizedChoice:
    """The ``.choices[0]`` the framework reads off a response."""

    message: NormalizedMessage
    finish_reason: str | None = None
    index: int = 0


@dataclass(frozen=True, slots=True)
class NormalizedResponse:
    """Provider-neutral view of a Chat Completions response.

    Two audiences, one object.  Application code reads the flat fields
    (:attr:`content`, :attr:`tool_calls`, :attr:`reasoning`).  The framework
    reads ``.choices[0].message``, because every agent mode, the critic, the
    decomposer and the observability parser all assume that shape — so this
    also presents it, rather than forcing the provider to return a raw SDK
    object and throw the normalization away.
    """

    content: str | None
    tool_calls: tuple[NormalizedToolCall, ...] = ()
    finish_reason: str | None = None
    request_id: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    usage_reported: bool = False
    reasoning: str | None = None
    raw: Any = field(default=None, compare=False, repr=False)

    @property
    def choices(self) -> list[NormalizedChoice]:
        """OpenAI-shaped view, as the framework's agent modes expect.

        ``tool_calls`` is a ``list`` rather than a tuple because
        ``_get_tool_calls`` gates on ``isinstance(calls, list)`` and would
        otherwise silently see no tool calls at all.
        """
        return [
            NormalizedChoice(
                message=NormalizedMessage(
                    content=self.content,
                    tool_calls=list(self.tool_calls) or None,
                    reasoning=self.reasoning,
                ),
                finish_reason=self.finish_reason,
            )
        ]

    @property
    def usage(self) -> dict[str, int] | None:
        """Token counts in the shape the framework's accounting layer reads.

        ``UsageTracker.record_from_response`` and ``build_llm_call_record``
        both start at ``response.usage`` and return early when it is absent.
        Without this the flat fields below are invisible to them, so every
        run reports zero tokens and every cost estimate is $0.00 — wrong in
        the direction nobody notices.

        ``None`` when the server reported no usage at all, which is a real
        and common case on streams: reporting zeros would be indistinguishable
        from a genuinely free call.
        """
        if not self.usage_reported:
            return None
        return {
            "prompt_tokens": self.prompt_tokens or 0,
            "completion_tokens": self.completion_tokens or 0,
            "total_tokens": self.total_tokens
            or ((self.prompt_tokens or 0) + (self.completion_tokens or 0)),
            "reasoning_tokens": self.reasoning_tokens or 0,
        }

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def has_reasoning(self) -> bool:
        """Whether the server returned separated thinking output."""
        return bool(self.reasoning)

    @property
    def reasoning_only(self) -> bool:
        """Whether thinking arrived but the answer did not.

        A known failure mode on vLLM: when the chat template and the
        ``--reasoning-parser`` disagree about whether thinking is enabled,
        the entire answer is returned as reasoning with ``content: null``
        (vLLM issue #53284).  Callers surface remediation guidance instead
        of treating it as an empty completion.
        """
        return bool(self.reasoning) and not self.content and not self.tool_calls


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read *key* from a dict-style or attribute-style object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, default)
    return default if value is None else value


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def extract_reasoning(container: Any) -> str | None:
    """Read separated thinking output from a message or a stream delta.

    vLLM originally emitted ``reasoning_content`` and later renamed the
    field to ``reasoning``; both names remain in circulation depending on
    server version, so both are tried, newest first.
    """
    for name in REASONING_FIELDS:
        value = _get(container, name)
        if isinstance(value, str) and value:
            return value
    return None


def _normalize_tool_calls(message: Any) -> tuple[NormalizedToolCall, ...]:
    raw_calls = _get(message, "tool_calls") or ()
    calls: list[NormalizedToolCall] = []
    for index, call in enumerate(raw_calls):
        fn = _get(call, "function")
        name = _get(fn, "name")
        if not isinstance(name, str) or not name:
            continue
        arguments = _get(fn, "arguments", "") or ""
        call_id = _get(call, "id") or f"call_{index}"
        calls.append(
            NormalizedToolCall(
                id=str(call_id),
                name=name,
                arguments=arguments if isinstance(arguments, str) else "",
            )
        )
    return tuple(calls)


def normalize_response(
    response: Any, *, request_id: str | None = None
) -> NormalizedResponse:
    """Flatten an SDK Chat Completions response into a stable shape.

    Args:
        response: The object returned by ``chat.completions.create``.
        request_id: Request id lifted from response headers, if available.
            Falls back to the body's ``id``.

    Returns:
        A :class:`NormalizedResponse`; a response with no choices yields one
        with ``content=None`` rather than raising, so an empty completion is
        handled by the caller as a normal (if unhelpful) outcome.
    """
    choices = _get(response, "choices") or ()
    message: Any = None
    finish_reason: str | None = None

    if choices:
        first = choices[0]
        message = _get(first, "message")
        reason = _get(first, "finish_reason")
        finish_reason = reason if isinstance(reason, str) else None

    content = _get(message, "content")
    if not isinstance(content, str):
        content = None

    reasoning = extract_reasoning(message)

    usage = _get(response, "usage")
    prompt_tokens = _int_or_none(_get(usage, "prompt_tokens"))
    completion_tokens = _int_or_none(_get(usage, "completion_tokens"))
    total_tokens = _int_or_none(_get(usage, "total_tokens"))
    reasoning_tokens = _int_or_none(
        _get(_get(usage, "completion_tokens_details"), "reasoning_tokens")
    )
    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    body_id = _get(response, "id")
    model = _get(response, "model")

    return NormalizedResponse(
        content=content,
        tool_calls=_normalize_tool_calls(message),
        finish_reason=finish_reason,
        request_id=request_id or (body_id if isinstance(body_id, str) else None),
        model=model if isinstance(model, str) else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        usage_reported=usage is not None and prompt_tokens is not None,
        reasoning=reasoning,
        raw=response,
    )
