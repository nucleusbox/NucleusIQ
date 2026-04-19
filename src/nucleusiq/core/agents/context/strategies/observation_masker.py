"""ObservationMasker — Tier 0 post-response strategy.

After the model produces a response, tool results that have been
"consumed" (the model saw them and responded) are replaced with
structured markers.  Full content is preserved in ``ContentStore``.

This is NOT part of the ``CompactionPipeline`` — it runs
unconditionally after every LLM response via ``engine.post_response()``.

Responsibility (SRP): identify consumed tool-result messages and
replace their content with a structured marker.  Does NOT decide
*whether* to run — that is ``ContextEngine``'s decision.

Marker format (F1 — structured fact slots)
------------------------------------------
The marker is a multi-line block with five fact slots so that
downstream consumers (Critic, Refiner, weaker models re-reading the
trace) can reason about the masked result without needing to rehydrate
it from ``ContentStore``::

    [observation consumed]
    tool: <tool_name>
    args: <one-line arg preview>
    ref: <store key>
    size: ~<tokens> tokens
    summary: <first 200 chars of content>

Backward compatibility: the marker still starts with
``[observation consumed`` so the idempotency check on subsequent mask
passes continues to work.

Research backing:
    - Morph Research (Feb 2026): Claude Code uses 5.5x fewer tokens
      than Cursor.  Primary mechanism: stripping consumed tool outputs.
    - 80% of context rot comes from stale tool results that the model
      has already incorporated into its reasoning.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nucleusiq.agents.chat_models import ChatMessage
    from nucleusiq.agents.context.counter import TokenCounter
    from nucleusiq.agents.context.store import ContentStore

_MASKED_MARKER_TEMPLATE = (
    "[observation consumed]\n"
    "tool: {tool_name}\n"
    "args: {args_preview}\n"
    "ref: {key}\n"
    "size: ~{tokens} tokens\n"
    "summary: {summary}"
)

#: Marker prefix used for idempotency checks.  Any message whose
#: content begins with this string is already a marker and must not be
#: masked again.  Kept as a module-level constant so callers (notably
#: ``extract_raw_trace``) can use the same prefix without duplication.
MASK_PREFIX = "[observation consumed"

_ARGS_PREVIEW_MAX_CHARS = 200
_SUMMARY_MAX_CHARS = 200
_SUMMARY_UNAVAILABLE = "(empty)"
_ARGS_UNAVAILABLE = "(unavailable)"


def _one_line(text: str, max_chars: int) -> str:
    """Collapse whitespace / newlines and truncate to ``max_chars``.

    Keeps the marker shape single-line-per-slot so downstream parsers
    (e.g. ``extract_raw_trace``) can split on ``\\n`` without ambiguity.
    """
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 3] + "..."


def _build_args_preview(tool_call: object | None) -> str:
    """Extract a compact one-line preview of a tool call's arguments.

    Handles the two shapes a tool_call may take on the wire:

    * A flat dict/object with ``name`` + ``arguments`` (canonical).
    * A nested OpenAI SDK object: ``{"function": {"name", "arguments"}}``.

    ``arguments`` may itself be either a JSON string (OpenAI) or a
    dict (some providers).  Both are normalised to a compact JSON
    string so the marker is deterministic.
    """
    if tool_call is None:
        return _ARGS_UNAVAILABLE

    args: object = ""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function")
        if isinstance(fn, dict):
            args = fn.get("arguments", "")
        else:
            args = tool_call.get("arguments", "")
    else:
        fn = getattr(tool_call, "function", None)
        if fn is not None:
            args = getattr(fn, "arguments", "")
        else:
            args = getattr(tool_call, "arguments", "")

    if not args:
        return _ARGS_UNAVAILABLE

    if isinstance(args, (dict, list)):
        try:
            args_str = json.dumps(args, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            args_str = str(args)
    else:
        args_str = str(args)

    return _one_line(args_str, _ARGS_PREVIEW_MAX_CHARS) or _ARGS_UNAVAILABLE


def _extract_tc_id(tool_call: object) -> str | None:
    """Extract the ``id`` field from a tool_call dict or SDK object."""
    if isinstance(tool_call, dict):
        return tool_call.get("id")
    return getattr(tool_call, "id", None)


def _build_tool_call_index(
    messages: "list[ChatMessage]",
) -> dict[str, object]:
    """Build ``tool_call_id -> tool_call`` index across all assistant turns.

    The index is used to look up the arguments that produced each
    ``role='tool'`` message so the marker can include them.  Without
    this index the marker would only carry the tool's *output*; the
    *input* (which often holds the question the tool was answering)
    would be lost.
    """
    index: dict[str, object] = {}
    for msg in messages:
        if msg.role != "assistant":
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            tc_id = _extract_tc_id(tc)
            if tc_id:
                index[tc_id] = tc
    return index


def build_marker(
    *,
    tool_name: str,
    args_preview: str,
    key: str,
    tokens: int,
    summary: str,
) -> str:
    """Public helper: assemble a masked-observation marker.

    Exposed for tests and for ``extract_raw_trace`` which needs to
    recognise the same layout.  Keeping the format behind a function
    means the template lives in exactly one place.
    """
    return _MASKED_MARKER_TEMPLATE.format(
        tool_name=tool_name,
        args_preview=args_preview,
        key=key,
        tokens=tokens,
        summary=summary,
    )


class ObservationMasker:
    """Replaces consumed tool results with structured markers.

    A tool result is "consumed" when it appears before the most recent
    assistant message — the model has already seen and responded to it.

    Design:
        - Stateless — receives messages, returns modified copy.
        - Does NOT mutate input list (returns a new list).
        - Stores full content in ``ContentStore`` for rehydration
          (see ``extract_raw_trace`` in ``context/store.py``).
        - Marker carries ``tool``, ``args``, ``ref``, ``size``,
          ``summary`` so downstream consumers can reason about the
          masked result without always rehydrating (F1).
    """

    __slots__ = ()

    def mask(
        self,
        messages: "list[ChatMessage]",
        token_counter: "TokenCounter",
        store: "ContentStore",
    ) -> tuple["list[ChatMessage]", int, int]:
        """Mask consumed tool results in the message list.

        A tool result at index *i* is "consumed" if there exists an
        assistant message at index *j > i*.  Tool results after the
        last assistant message are NOT masked (the model hasn't
        responded to them yet).

        Args:
            messages: Current conversation messages.
            token_counter: For counting tokens freed.
            store: Where to offload full content.

        Returns:
            Tuple of ``(new_messages, observations_masked, tokens_freed)``.
        """
        from nucleusiq.agents.chat_models import ChatMessage as CM

        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx < 0:
            return list(messages), 0, 0

        tc_index = _build_tool_call_index(messages)

        result: list[ChatMessage] = []
        masked_count = 0
        total_freed = 0

        for i, msg in enumerate(messages):
            if (
                i < last_assistant_idx
                and msg.role == "tool"
                and isinstance(msg.content, str)
                and not msg.content.startswith(MASK_PREFIX)
                and not msg.content.startswith("[context_ref:")
            ):
                original_tokens = token_counter.count(msg.content)
                if original_tokens < 20:
                    result.append(msg)
                    continue

                tool_name = msg.name or "tool"
                key = f"obs:{tool_name}:{uuid.uuid4().hex[:8]}"
                store.store(
                    key=key,
                    content=msg.content,
                    original_tokens=original_tokens,
                )

                originating_call = (
                    tc_index.get(msg.tool_call_id) if msg.tool_call_id else None
                )
                args_preview = _build_args_preview(originating_call)
                summary = (
                    _one_line(msg.content, _SUMMARY_MAX_CHARS)
                    or _SUMMARY_UNAVAILABLE
                )

                marker = build_marker(
                    tool_name=tool_name,
                    args_preview=args_preview,
                    key=key,
                    tokens=original_tokens,
                    summary=summary,
                )
                marker_tokens = token_counter.count(marker)
                freed = max(0, original_tokens - marker_tokens)

                result.append(
                    CM(
                        role=msg.role,
                        content=marker,
                        name=msg.name,
                        tool_call_id=msg.tool_call_id,
                    )
                )
                masked_count += 1
                total_freed += freed
            else:
                result.append(msg)

        return result, masked_count, total_freed


__all__ = [
    "MASK_PREFIX",
    "ObservationMasker",
    "build_marker",
]
