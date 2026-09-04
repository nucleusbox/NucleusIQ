"""Streaming adapter — SSE chunks to ``StreamEvent``.

Two things make streaming on OpenAI-compatible servers harder than it looks:

* **Tool-call arguments arrive in fragments.** Each chunk carries a partial
  ``arguments`` string keyed by ``index``, and the ``id``/``name`` typically
  appear only in the first fragment. :class:`ToolCallAccumulator` merges them
  into whole calls.
* **Usage is often absent.** Servers omit it on streams unless asked via
  ``stream_options={"include_usage": True}``, and it then arrives in a final
  chunk with an empty ``choices`` list, which must not be mistaken for the
  end of content.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from nucleusiq.streaming.events import StreamEvent

from .._shared.response_models import (
    NormalizedResponse,
    NormalizedToolCall,
    extract_reasoning,
)

__all__ = ["StreamOutcome", "ToolCallAccumulator", "stream_events"]


@dataclass
class _PartialCall:
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass
class ToolCallAccumulator:
    """Merges fragmented tool-call deltas into complete calls."""

    _parts: dict[int, _PartialCall] = field(default_factory=dict)

    def add(self, delta_calls: Any) -> None:
        """Absorb the ``tool_calls`` list from one streamed delta."""
        for position, call in enumerate(delta_calls or ()):
            index = _get(call, "index")
            key = index if isinstance(index, int) else position
            part = self._parts.setdefault(key, _PartialCall())

            call_id = _get(call, "id")
            if isinstance(call_id, str) and call_id:
                part.id = call_id

            fn = _get(call, "function")
            name = _get(fn, "name")
            if isinstance(name, str) and name:
                part.name = name
            arguments = _get(fn, "arguments")
            if isinstance(arguments, str):
                part.arguments += arguments

    def finish(self) -> tuple[NormalizedToolCall, ...]:
        """Return the completed calls, ordered by their stream index."""
        out: list[NormalizedToolCall] = []
        for key in sorted(self._parts):
            part = self._parts[key]
            if not part.name:
                continue
            out.append(
                NormalizedToolCall(
                    id=part.id or f"call_{key}",
                    name=part.name,
                    arguments=part.arguments,
                )
            )
        return tuple(out)

    def __bool__(self) -> bool:
        return bool(self._parts)


@dataclass
class StreamOutcome:
    """Accumulated result of a stream, available once iteration finishes.

    Lets the caller record usage and tool calls on an ``LLMCallRecord``
    without re-reading a consumed stream.
    """

    text: str = ""
    reasoning: str = ""
    response: NormalizedResponse | None = None


async def stream_events(
    chunks: AsyncIterator[Any],
    *,
    outcome: StreamOutcome,
    model: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Convert an SSE chunk stream into ``StreamEvent`` objects.

    Yields a ``TOKEN`` event per content delta and a final ``COMPLETE``
    event carrying the accumulated text. *outcome* is populated in place
    with the full text, tool calls and usage.

    Args:
        chunks: Async iterator of SDK chunk objects.
        outcome: Mutable holder filled in as the stream progresses.
        model: Model name to record on the normalized response.
    """
    accumulator = ToolCallAccumulator()
    pieces: list[str] = []
    reasoning_pieces: list[str] = []
    finish_reason: str | None = None
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    usage_reported = False

    async for chunk in chunks:
        chunk_id = _get(chunk, "id")
        if request_id is None and isinstance(chunk_id, str):
            request_id = chunk_id

        usage = _get(chunk, "usage")
        if usage is not None:
            prompt_tokens = _as_int(_get(usage, "prompt_tokens")) or prompt_tokens
            completion_tokens = (
                _as_int(_get(usage, "completion_tokens")) or completion_tokens
            )
            total_tokens = _as_int(_get(usage, "total_tokens")) or total_tokens
            usage_reported = prompt_tokens is not None

        choices = _get(chunk, "choices") or ()
        if not choices:
            # A usage-only trailer chunk; not the end of content.
            continue

        choice = choices[0]
        reason = _get(choice, "finish_reason")
        if isinstance(reason, str):
            finish_reason = reason

        delta = _get(choice, "delta")

        # Thinking deltas are tagged rather than merged into content, so a UI
        # can render them in a separate pane and a plain consumer can filter
        # them out with one check. The COMPLETE event carries answer text only.
        thinking = extract_reasoning(delta)
        if thinking:
            reasoning_pieces.append(thinking)
            yield StreamEvent.token_event(thinking, metadata={"reasoning": True})

        text = _get(delta, "content")
        if isinstance(text, str) and text:
            pieces.append(text)
            yield StreamEvent.token_event(text)

        accumulator.add(_get(delta, "tool_calls"))

    full_text = "".join(pieces)
    full_reasoning = "".join(reasoning_pieces)
    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    outcome.text = full_text
    outcome.reasoning = full_reasoning
    response = NormalizedResponse(
        content=full_text or None,
        tool_calls=accumulator.finish(),
        finish_reason=finish_reason,
        request_id=request_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        usage_reported=usage_reported,
        reasoning=full_reasoning or None,
    )
    outcome.response = response

    yield StreamEvent.complete_event(full_text, metadata=_complete_metadata(response))


def _complete_metadata(response: NormalizedResponse) -> dict[str, Any] | None:
    """Build the COMPLETE event metadata the streaming agent loop reads.

    On the streaming path a ``NormalizedResponse`` never reaches the agent —
    only ``StreamEvent``s do — so this metadata is the *sole* channel for
    anything that is not answer text. Two consumers depend on it:

    * ``base_mode`` streaming reads ``metadata["tool_calls"]`` to decide
      whether to run the tool loop. Omitting it does not error; the loop
      simply never fires, so a streaming agent silently ignores every tool
      the model asked for.
    * ``UsageTracker.record_from_stream_metadata`` reads ``metadata["usage"]``
      for token accounting and cost.

    Tool calls travel in OpenAI wire form because ``ToolCallRequest.from_raw``
    accepts that shape and it matches what the non-streaming path sends.
    """
    metadata: dict[str, Any] = {}
    if response.tool_calls:
        metadata["tool_calls"] = [tc.to_wire() for tc in response.tool_calls]
    usage = response.usage
    if usage is not None:
        metadata["usage"] = usage
    if response.model:
        metadata["model"] = response.model
    if response.finish_reason:
        metadata["finish_reason"] = response.finish_reason
    if response.reasoning:
        metadata["reasoning_content"] = response.reasoning
    return metadata or None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, default)
    return default if value is None else value


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
