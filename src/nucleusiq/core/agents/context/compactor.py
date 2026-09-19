"""Compactor — single consolidated context-management engine.

Context Mgmt v2 — Step 3
========================
This module replaces the v1/Step-2 multi-strategy pipeline with one
priority-ordered :class:`Compactor` that handles every flavour of
context reduction the v1 design distributed across four files:

    * ``ToolResultCompactor``   →  size-gated per-message offload.
    * ``ConversationCompactor`` →  drop oldest turn groups.
    * ``EmergencyCompactor``    →  last-resort reduction to head + last group.
    * ``ObservationMasker``     →  replace consumed tool results with markers.

The Compactor is the **only** code path execution modes care about.
:class:`ContextEngine` delegates two operations to it:

    1. :meth:`Compactor.compact` — pre-LLM-call reduction
       (priority-ordered, fired from ``engine.prepare()``).
    2. :meth:`Compactor.mask`    — post-response masking of consumed
       tool results (fired from ``engine.post_response()``).

Behaviour preservation
----------------------
Step 3 is a **pure simplification** (§12 of the v2 redesign): the
external behaviour observed by execution modes, the Critic, the
Refiner, and the test suite is identical to Step 2.  The work was:

    1. Move all logic into one file.
    2. Inline the helpers (``_partition``, ``_group_touches_hot_ref``,
       ``_build_args_preview``, ``_build_structured_summary``) so each
       lives in exactly one place.
    3. Keep the public types (``CompactionResult``,
       ``CompactionStrategy``, ``CompactionEvent``) unchanged so older
       callers and tests do not need touch-ups beyond import paths.

Net delta: −1119 LoC across five deleted files, +~700 LoC here, plus
the removal of ``pipeline.py`` orchestration layer.  See §8 of the
redesign doc for the per-file accounting.

Marker formats
--------------
The two markers — ``[observation consumed]`` (post-response) and
``[N earlier messages compacted ...]`` (prepare-time) — are emitted
verbatim from the Step 2 implementations so callers that grep for
specific prefixes (``store._MASK_PREFIX``, ``extract_raw_trace``)
keep working.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING

from nucleusiq.agents.context.strategies.base import (
    CompactionResult,
    CompactionStrategy,
)
from nucleusiq.agents.context.telemetry import CompactionEvent

if TYPE_CHECKING:
    from nucleusiq.agents.chat_models import ChatMessage
    from nucleusiq.agents.context.budget import ContextBudget
    from nucleusiq.agents.context.config import ContextConfig
    from nucleusiq.agents.context.counter import TokenCounter
    from nucleusiq.agents.context.store import ContentStore


logger = logging.getLogger(__name__)


# =====================================================================
# Re-exported types (defined in ``strategies/base.py`` per §8 of the
# v2 redesign — base.py is kept so callers that imported the types
# from there continue working).
# =====================================================================


# =====================================================================
# Marker constants (kept as module-level so other modules can
# import them — store.py and extract_raw_trace already do).
# =====================================================================

#: Prefix used by the post-response masker to mark a consumed tool
#: result.  Idempotency check: any tool message starting with this is
#: already a marker and must not be re-masked.
MASK_PREFIX = "[observation consumed"

_MASKED_MARKER_TEMPLATE_WITH_PREVIEW = (
    "[observation consumed]\n"
    "tool: {tool_name}\n"
    "args: {args_preview}\n"
    "ref: {key}\n"
    "size: ~{tokens} tokens\n"
    "preview (first {preview_chars} chars of {tokens}-token result):\n"
    "{preview}\n"
    "[end preview — DO NOT re-call this tool with these args; "
    'use the preview above, or call recall_tool_result(ref="{key}") for the full content]'
)

_MASKED_MARKER_TEMPLATE = (
    "[observation consumed]\n"
    "tool: {tool_name}\n"
    "args: {args_preview}\n"
    "ref: {key}\n"
    "size: ~{tokens} tokens\n"
    'To retrieve: call recall_tool_result(ref="{key}")'
)

_REF_LINE_RE = re.compile(r"^ref:\s*(\S+)\s*$", re.MULTILINE)
_CONTEXT_REF_PREFIX = "[context_ref:"
_CONTEXT_REF_RE = re.compile(r"^\[context_ref:\s*([^\]]+)\]", re.MULTILINE)

# One system catalog, not N orphaned tool messages. Eviction (Tier 2
# or Tier 3) may drop bulky turns, but store-backed evidence must
# stay addressable or dedup/recall become a non-convergent loop.
_EVIDENCE_CATALOG_PREFIX = "[evidence available for recall]"
_EVIDENCE_CATALOG_MAX_ENTRIES = 40

_ARGS_PREVIEW_MAX_CHARS = 200
_ARGS_UNAVAILABLE = "(unavailable)"

# Context Mgmt v2 — Step 4 (re-fetch loop fix).
# When the masker offloads a tool result, it now retains the first N
# characters of the original content as a "preview" inline in the
# marker.  Empirically (Task E × gpt-5.2, Apr 2026), the prior
# preview-less marker caused frontier models to re-fetch the same
# content rather than call recall_tool_result — the marker had no
# semantic content to reason from, so the model defaulted to
# re-executing the original tool.  Keep this preview intentionally
# small: Task E showed that 1500-char previews compound across long
# PDF-heavy runs and can leave hundreds of thousands of residual marker
# tokens. A ~300-char preview preserves orientation without turning
# markers back into a second prompt-sized evidence store.
_PREVIEW_MAX_CHARS = 300
_PREVIEW_ENABLE_MIN_CHARS = 1500

_PREVIEW_HEAD_LINES = 8
_PREVIEW_TAIL_LINES = 4
_TRUNCATION_MARKER = (
    "\n[...truncated {freed} tokens — tool result exceeded threshold...]\n"
)

_EMERGENCY_MARKER = (
    "[CONTEXT COMPACTED: emergency reduction triggered at {util:.0%} utilization. "
    "{dropped} messages removed (~{tokens} tokens). "
    "Only system prompt and last {kept} messages preserved.]"
)

# Adaptive Tier-1 squeeze (see ``Compactor._tool_pass``).  A receipt with
# its preview costs on the order of 100-200 tokens, so offloading a
# result smaller than this frees nothing and only adds indirection.
_ADAPTIVE_OFFLOAD_MIN_TOKENS = 256
_ADAPTIVE_RECEIPT_HINT = (
    "[offloaded to fit the context window before you saw it — the preview "
    'above is the head of the result; call recall_tool_result(ref="{key}") '
    "if you need the rest. Do NOT call the original tool again.]"
)


# =====================================================================
# Pure helpers (free functions — no class state)
# =====================================================================


def _one_line(text: str, max_chars: int) -> str:
    """Collapse whitespace and truncate to ``max_chars``.

    Used by the masker to keep marker slots single-line so downstream
    parsers (``extract_raw_trace``) can split on ``\\n`` unambiguously.
    """
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 3] + "..."


def _build_args_preview(tool_call: object | None) -> str:
    """One-line preview of a tool call's arguments for the marker.

    Handles both wire shapes — flat dict and nested OpenAI SDK object
    — and normalises ``arguments`` (which may be a JSON string or
    dict) into a compact deterministic JSON blob.
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
    """Extract the ``id`` field from a tool_call (dict or SDK object)."""
    if isinstance(tool_call, dict):
        return tool_call.get("id")
    return getattr(tool_call, "id", None)


def _build_tool_call_index(
    messages: list[ChatMessage],
) -> dict[str, object]:
    """Build a ``tool_call_id → tool_call`` index across assistant turns.

    Lets the masker stamp the originating call's ``arguments`` into
    the marker, so downstream consumers see *what* the tool was asked
    rather than just *what it returned*.
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


def _build_content_preview(content: str, max_chars: int = _PREVIEW_MAX_CHARS) -> str:
    """Return the first ``max_chars`` of a tool result, sanitised.

    The preview is what the model sees in the marker after the full
    content is offloaded to ``ContentStore``.  It must be:

    * **Short** — typically <100 tokens at the default 300 chars,
      cheap to keep in context across many turns.
    * **Information-dense** — most tool results (PDF page extracts,
      search hits, document fetches) put the answer-bearing prose at
      the top, so a head-slice preserves the most useful content.
    * **Newline-safe** — preserved as-is so paragraph structure
      survives.  Downstream parsers (``extract_raw_trace``) anchor on
      ``[observation consumed]`` and ``[end preview`` so the inner
      newlines don't break parsing.

    If the content is already short (≤ max_chars) it is returned in
    full and the marker effectively becomes a no-op replacement.
    """
    if not content:
        return ""
    if len(content) <= max_chars:
        return content
    head = content[: max_chars - 3].rstrip()
    return f"{head}..."


def build_marker(
    *,
    tool_name: str,
    args_preview: str,
    key: str,
    tokens: int,
    preview: str | None = None,
    summary: str | None = None,  # legacy parameter, ignored (v0.7.8+)
) -> str:
    """Public helper to assemble a masked-observation marker.

    Exposed so ``extract_raw_trace`` and tests can produce or
    recognise the same shape without duplicating the template.  The
    legacy ``summary`` slot was removed in v0.7.8 (see redesign doc
    §3.7) — the parameter is accepted but ignored for back-compat.

    Context Mgmt v2 — Step 4: when ``preview`` is provided, the
    richer template is used so the marker carries answer-bearing
    content inline (not just an opaque ref).  This breaks the
    re-fetch loop where the model would re-execute the tool because
    the bare marker held nothing it could reason from.
    """
    del summary
    if preview:
        return _MASKED_MARKER_TEMPLATE_WITH_PREVIEW.format(
            tool_name=tool_name,
            args_preview=args_preview,
            key=key,
            tokens=tokens,
            preview=preview,
            preview_chars=len(preview),
        )
    return _MASKED_MARKER_TEMPLATE.format(
        tool_name=tool_name,
        args_preview=args_preview,
        key=key,
        tokens=tokens,
    )


def _extract_store_ref(content: str) -> str | None:
    """Return the store key from a known receipt shape, else ``None``.

    Two receipt formats exist today:

    * ``[observation consumed]`` + ``ref: {key}`` (post-response masker)
    * ``[context_ref: {key}]`` (Tier-1 tool-result offload)

    Unknown tool payloads are not treated as receipts — a raw result
    that happens to contain the substring ``ref:`` must still be
    offloaded, not skipped.
    """
    if content.startswith(MASK_PREFIX):
        match = _REF_LINE_RE.search(content)
        return match.group(1) if match else None
    if content.startswith(_CONTEXT_REF_PREFIX):
        match = _CONTEXT_REF_RE.search(content)
        return match.group(1).strip() if match else None
    return None


_ARGS_LINE_RE = re.compile(r"^args:\s*(.+?)\s*$", re.MULTILINE)


def _tool_call_args_by_id(messages: list[ChatMessage]) -> dict[str, str]:
    """``tool_call_id -> args preview`` for every assistant call in ``messages``."""
    out: dict[str, str] = {}
    for msg in messages:
        if getattr(msg, "role", None) != "assistant":
            continue
        for tc in getattr(msg, "tool_calls", None) or []:
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if call_id:
                out[str(call_id)] = _build_args_preview(tc)
    return out


def _collect_addressable_evidence(
    evicted: list[ChatMessage],
    store: ContentStore | None,
    token_counter: TokenCounter,
) -> tuple[list[tuple[str, str, int, str]], int]:
    """Collect store refs from evicted tool turns.

    Already-offloaded receipts keep their key.  Raw tool results are
    written to ``store`` first so eviction does not destroy them.
    Returns ``(entries, newly_offloaded)`` where each entry is
    ``(ref, tool_name, tokens, args_preview)``.

    ``args_preview`` is what makes the catalog actionable: the model's
    own ``tool_calls`` are evicted with the group, so without the
    arguments it could not tell *which* input a ref belongs to and
    would re-fetch it (the 0.7.13 dedup loop).  Arguments come from the
    evicted assistant message (matched by ``tool_call_id``) or from the
    ``args:`` line of an already-masked receipt.
    """
    seen: set[str] = set()
    entries: list[tuple[str, str, int, str]] = []
    offloaded = 0
    args_by_id = _tool_call_args_by_id(evicted)

    for msg in evicted:
        if getattr(msg, "role", None) != "tool":
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, str) or not content:
            continue

        tool_name = getattr(msg, "name", None) or "tool"
        try:
            tokens = token_counter.count(content)
        except Exception:
            tokens = max(1, len(content) // 4)

        args_preview = args_by_id.get(str(getattr(msg, "tool_call_id", "") or ""), "")
        if not args_preview:
            match = _ARGS_LINE_RE.search(content[:600])
            if match:
                args_preview = match.group(1)

        ref = _extract_store_ref(content)
        if ref is None and store is not None:
            ref = f"obs:{tool_name}:{uuid.uuid4().hex[:8]}"
            store.store(
                key=ref,
                content=content,
                original_tokens=tokens,
                tool_name=tool_name,
            )
            offloaded += 1

        if ref and ref not in seen:
            seen.add(ref)
            entries.append((ref, tool_name, tokens, args_preview))

    return entries, offloaded


def _build_evidence_catalog(
    entries: list[tuple[str, str, int, str]],
) -> ChatMessage | None:
    """One system catalog of evicted-but-recallable evidence.

    A catalog is chat-template safe (no orphaned ``role=tool``
    messages) and is the same object for conversation eviction and
    emergency reduction — the model can ``recall_tool_result`` or
    ``list_recalled_evidence`` from either tier.  Each line names the
    call's arguments so the model can map refs back to its inputs.
    """
    if not entries:
        return None

    from nucleusiq.agents.chat_models import ChatMessage as CM

    visible = entries[:_EVIDENCE_CATALOG_MAX_ENTRIES]
    hidden = len(entries) - len(visible)
    lines = [
        _EVIDENCE_CATALOG_PREFIX,
        "Earlier tool results were removed from the live transcript.",
        "They are still in the context store. Do not re-call those",
        "tools with the same arguments — recall instead.",
    ]
    for ref, tool_name, tokens, args_preview in visible:
        args_part = (
            f" args={args_preview}"
            if args_preview and args_preview != _ARGS_UNAVAILABLE
            else ""
        )
        lines.append(f'- ref="{ref}" tool={tool_name}{args_part} ~{tokens} tokens')
    if hidden:
        lines.append(f"... and {hidden} more.")
    lines.append("Call recall_tool_result(ref=...) or list_recalled_evidence().")
    return CM(role="system", content="\n".join(lines))


def _group_touches_hot_ref(
    group: list[ChatMessage], hot_set: frozenset[str] | set[str]
) -> bool:
    """Return ``True`` if any tool message in ``group`` references a hot ref.

    Hot-set rescue (§6.5 of the redesign): the Compactor refuses to
    evict tool turns the model just recalled, otherwise we silently
    undo the recall and force the model to re-fetch the same content.
    """
    if not hot_set:
        return False
    for msg in group:
        if msg.role != "tool":
            continue
        content = msg.content
        if not isinstance(content, str):
            continue
        ref = _extract_store_ref(content)
        if ref and ref in hot_set:
            return True
    return False


def _build_structured_summary(evicted: list[ChatMessage], evicted_tokens: int) -> str:
    """Heuristic working-state summary for the conversation marker.

    Extracts goals (user requests), decisions (assistant first lines),
    and tool findings without an LLM call.  Used when
    ``enable_summarization`` is on; otherwise the marker is a single-
    line "[N earlier messages compacted ...]".
    """
    goals: list[str] = []
    decisions: list[str] = []
    tool_findings: list[str] = []

    for msg in evicted:
        text = msg.content if isinstance(msg.content, str) else ""
        if not text:
            continue

        first_line = text.split("\n")[0][:200]

        if msg.role == "user":
            goals.append(first_line)
        elif msg.role == "assistant":
            decisions.append(first_line)
        elif msg.role == "tool":
            name = msg.name or "tool"
            tool_findings.append(f"{name}: {first_line[:100]}")

    parts = [
        f"[WORKING STATE SUMMARY — {len(evicted)} messages, "
        f"~{evicted_tokens} tokens compacted]",
    ]
    if goals:
        parts.append(f"Goals: {'; '.join(goals[:5])}")
    if decisions:
        parts.append(f"Decisions: {'; '.join(decisions[:5])}")
    if tool_findings:
        parts.append(f"Tool findings: {'; '.join(tool_findings[:5])}")

    return "\n".join(parts)


# =====================================================================
# Internal compaction passes (private; called from Compactor.compact)
# =====================================================================


def _truncate_tool_content(
    content: str,
    original_tokens: int,
    token_counter: TokenCounter,
) -> tuple[str, int]:
    """Lossy fallback when offloading is disabled.

    Keeps head + tail lines (or chars for dense content) and inserts a
    truncation marker.  Returns ``(new_content, freed_tokens)``.
    """
    lines = content.split("\n")
    if len(lines) > _PREVIEW_HEAD_LINES + _PREVIEW_TAIL_LINES:
        head = "\n".join(lines[:_PREVIEW_HEAD_LINES])
        tail = "\n".join(lines[-_PREVIEW_TAIL_LINES:])
        dropped = len(lines) - _PREVIEW_HEAD_LINES - _PREVIEW_TAIL_LINES
        marker = _TRUNCATION_MARKER.format(freed=f"~{dropped} lines")
        truncated = head + marker + tail
        new_tokens = token_counter.count(truncated)
        return truncated, max(0, original_tokens - new_tokens)

    head_chars = max(200, len(content) // 5)
    tail_chars = max(100, len(content) // 10)
    min_content = head_chars + tail_chars + 100
    if len(content) <= min_content:
        return content, 0

    head_text = content[:head_chars]
    tail_text = content[-tail_chars:]
    dropped_chars = len(content) - head_chars - tail_chars
    marker = _TRUNCATION_MARKER.format(freed=f"~{dropped_chars} chars")
    truncated = head_text + marker + tail_text
    new_tokens = token_counter.count(truncated)
    return truncated, max(0, original_tokens - new_tokens)


def _offload_tool_content(
    content: str,
    original_tokens: int,
    tool_name: str,
    token_counter: TokenCounter,
    store: ContentStore,
) -> tuple[str, int]:
    """Lossless offload: full content → ``ContentStore``, return preview marker.

    Marker carries a recall hint so the model can pull the bytes back
    via ``recall_tool_result(ref=...)``.  Returns ``(marker, freed)``.
    """
    key = f"{tool_name}:{uuid.uuid4().hex[:12]}"
    preview_char_budget = max(200, len(content) // 10)
    ref = store.store(
        key=key,
        content=content,
        original_tokens=original_tokens,
        preview_lines=_PREVIEW_HEAD_LINES,
        preview_max_chars=preview_char_budget,
        tool_name=tool_name,
    )
    marker_text = ref.to_marker()
    new_tokens = token_counter.count(marker_text)
    return marker_text, max(0, original_tokens - new_tokens)


def _offload_unseen_tool_content(
    content: str,
    original_tokens: int,
    tool_name: str,
    token_counter: TokenCounter,
    store: ContentStore,
) -> tuple[str, int]:
    """Lossless offload of a result the model has *not yet seen*.

    Same store round-trip as :func:`_offload_tool_content`, but the
    receipt tells the model explicitly that it never saw the full result,
    how to recall it, and that re-running the tool is the wrong move —
    the exact instruction a model needs when its own fresh tool output
    was too large to fit.
    """
    key = f"{tool_name}:{uuid.uuid4().hex[:12]}"
    preview_char_budget = max(200, len(content) // 10)
    ref = store.store(
        key=key,
        content=content,
        original_tokens=original_tokens,
        preview_lines=_PREVIEW_HEAD_LINES,
        preview_max_chars=preview_char_budget,
        tool_name=tool_name,
    )
    marker_text = ref.to_marker() + "\n" + _ADAPTIVE_RECEIPT_HINT.format(key=key)
    new_tokens = token_counter.count(marker_text)
    return marker_text, max(0, original_tokens - new_tokens)


def _tail_needs_squeeze(
    head: list[ChatMessage],
    tail: list[ChatMessage],
    budget: ContextBudget,
    token_counter: TokenCounter,
) -> bool:
    """True when task head + kept tail alone still exceed the working budget."""
    if budget.effective_limit <= 0:
        return False
    return (
        token_counter.count_messages(head) + token_counter.count_messages(tail)
        > budget.effective_limit
    )


def _squeeze_tail(
    tail: list[ChatMessage],
    *,
    head_tokens: int,
    budget: ContextBudget,
    config: ContextConfig,
    token_counter: TokenCounter,
    store: ContentStore | None,
    hot_set: frozenset[str],
    last_assistant: int,
    position: dict[int, int],
) -> tuple[list[ChatMessage], int, int, int]:
    """Shrink the tool results of the kept tail until it fits the budget.

    Returns ``(new_tail, tokens_freed, offloaded_count, lost_unseen_count)``.
    Largest results go first.  With a store the content is offloaded
    losslessly behind a preview + recall hint; without one it is
    truncated head/tail and — if the model had not read it yet — counted
    as lost evidence.
    """
    from nucleusiq.agents.chat_models import ChatMessage as CM

    if budget.effective_limit <= 0:
        return tail, 0, 0, 0
    excess = head_tokens + token_counter.count_messages(tail) - budget.effective_limit
    if excess <= 0:
        return tail, 0, 0, 0

    lossless = config.enable_offloading and store is not None
    candidates: list[tuple[int, int]] = []
    for idx, msg in enumerate(tail):
        if msg.role != "tool" or not isinstance(msg.content, str):
            continue
        if _is_receipt(msg.content):
            continue
        if hot_set and _group_touches_hot_ref([msg], hot_set):
            continue
        tokens = token_counter.count(msg.content)
        if tokens < _ADAPTIVE_OFFLOAD_MIN_TOKENS:
            continue
        candidates.append((-tokens, idx))
    candidates.sort()

    new_tail = list(tail)
    freed_total = 0
    offloaded = 0
    lost_unseen = 0
    for neg_tokens, idx in candidates:
        if excess <= 0:
            break
        msg = new_tail[idx]
        assert isinstance(msg.content, str)
        tokens = -neg_tokens
        unseen = position.get(id(msg), -1) > last_assistant
        if lossless:
            assert store is not None
            offload = _offload_unseen_tool_content if unseen else _offload_tool_content
            new_content, freed = offload(
                msg.content, tokens, msg.name or "tool", token_counter, store
            )
            offloaded += 1
        else:
            new_content, freed = _truncate_tool_content(
                msg.content, tokens, token_counter
            )
            if unseen and freed > 0:
                lost_unseen += 1
        if freed <= 0:
            continue
        new_tail[idx] = CM(
            role=msg.role,
            content=new_content,
            name=msg.name,
            tool_call_id=msg.tool_call_id,
        )
        excess -= freed
        freed_total += freed
    return new_tail, freed_total, offloaded, lost_unseen


def _last_assistant_index(messages: list[ChatMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            return i
    return -1


def _is_receipt(content: object) -> bool:
    return isinstance(content, str) and content.startswith(
        (MASK_PREFIX, _CONTEXT_REF_PREFIX, _EVIDENCE_CATALOG_PREFIX)
    )


def _split_task_head(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """Return ``(task_head, rest)`` — invariant **I1**.

    The task head is everything the model was *given* before it did
    anything: the leading system messages plus every user / system
    message that precedes the first assistant or tool turn.  Prompt
    templates routinely emit several of those (a user preamble, the
    ``Task.resources`` block, the objective itself); pinning only the
    first one — the pre-0.7.14 rule — let emergency compaction evict the
    objective and the resource list, after which the model "forgot" what
    it still had to process.

    If no user message precedes the first assistant turn (memory-first
    transcripts), the first user message anywhere is pinned instead, so
    the historical behaviour is preserved for that layout.
    """
    head: list[ChatMessage] = []
    idx = 0
    for msg in messages:
        if msg.role in ("system", "user"):
            head.append(msg)
            idx += 1
        else:
            break
    rest = messages[idx:]
    if any(m.role == "user" for m in head):
        return head, rest

    for i, msg in enumerate(rest):
        if msg.role == "user":
            head.append(msg)
            return head, rest[:i] + rest[i + 1 :]
    return head, rest


def _partition_for_conversation(
    messages: list[ChatMessage],
    preserve_recent: int,
    *,
    hot_set: frozenset[str] | None = None,
) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage]]:
    """Split messages into ``(pinned_head, evictable, pinned_tail)``.

    Pinning rules (Context Mgmt v2 — invariant **I1** + hot-set rescue):

    * The task head (leading system messages plus the user messages that
      precede the first assistant turn — see :func:`_split_task_head`)
      is always pinned.  Without this pin, synthesis produces
      "I don't have the instructions you're referring to..." refusals
      once the conversation grows past ``preserve_recent`` groups.
    * Assistant messages with ``tool_calls`` and their following
      ``tool`` results form atomic groups (provider validation
      requires it — splitting causes API 4xx).
    * The last ``preserve_recent`` atomic groups are pinned.
    * When ``hot_set`` is supplied, any old group whose tool messages
      reference a hot ref is rescued back into the head, preserving
      original chronological order.
    """
    pinned_head, remaining = _split_task_head(messages)
    if not remaining:
        return pinned_head, [], []

    groups: list[list[ChatMessage]] = []
    i = 0
    while i < len(remaining):
        msg = remaining[i]
        group = [msg]
        if msg.role == "assistant" and getattr(msg, "tool_calls", None):
            j = i + 1
            while j < len(remaining) and remaining[j].role == "tool":
                group.append(remaining[j])
                j += 1
            i = j
        else:
            i += 1
        groups.append(group)

    tail_count = max(preserve_recent, 1)
    if len(groups) <= tail_count:
        return pinned_head, [], remaining

    evict_groups = groups[: len(groups) - tail_count]
    tail_groups = groups[len(groups) - tail_count :]

    # Hot-set rescue — promote groups touching a hot ref back into the
    # pinned head.  Order preserved: rescued groups appear in the same
    # chronological position they had in the source transcript.
    hot = hot_set or frozenset()
    rescued_groups: list[list[ChatMessage]] = []
    truly_evictable_groups: list[list[ChatMessage]] = []
    for g in evict_groups:
        if hot and _group_touches_hot_ref(g, hot):
            rescued_groups.append(g)
        else:
            truly_evictable_groups.append(g)

    if rescued_groups:
        for g in rescued_groups:
            pinned_head.extend(g)

    evictable = [m for g in truly_evictable_groups for m in g]
    pinned_tail = [m for g in tail_groups for m in g]
    return pinned_head, evictable, pinned_tail


# =====================================================================
# The single Compactor
# =====================================================================


class Compactor:
    """Single consolidated compaction engine.

    Responsibilities (formerly split across 5 files):

    * **Per-tool offload / truncation** (was ``ToolResultCompactor``)
      — runs first inside :meth:`compact` so a single oversized tool
      result cannot blow the budget mid-turn.
    * **Conversation eviction** (was ``ConversationCompactor``) —
      drops oldest turn groups while honouring the I1 + hot-set pin
      invariants, and keeps evicted store refs in an evidence catalog.
    * **Emergency reduction** (was ``EmergencyCompactor``) —
      last-resort reduction to ``head + last group`` when utilisation
      crosses the panic threshold. Same addressability catalog as
      conversation eviction — not a special-case marker keep.
    * **Post-response masking** (was ``ObservationMasker``) — replaces
      consumed tool results with structured markers; runs from
      :meth:`mask`.

    Step 3 unification keeps every Step 2 behaviour gate intact (most
    importantly the ``squeeze_threshold`` gate in front of masking
    and the ``preserve_recent_turns`` pin).  See module docstring.
    """

    __slots__ = ()

    # ------------------------------------------------------------------
    # Pre-LLM reduction (replaces CompactionPipeline + 3 strategies)
    # ------------------------------------------------------------------

    async def compact(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None = None,
        *,
        hot_set: frozenset[str] | None = None,
    ) -> tuple[list[ChatMessage], list[CompactionEvent]]:
        """Run priority-ordered compaction.

        Mirrors the v1/Step-2 progressive pipeline tier-by-tier so the
        output is identical to running ``CompactionPipeline.run()``
        with the three strategies in order.  Each tier emits a
        :class:`CompactionEvent` for telemetry; events with
        ``tokens_freed == 0`` are dropped to keep telemetry quiet.
        """
        current_messages = list(messages)
        events: list[CompactionEvent] = []
        current_util = budget.utilization

        # Tier 1 — per-tool-result offload / truncate (Minor GC).
        # Triggered at ``tool_compaction_trigger``.  Cheap, instant.
        if current_util >= config.tool_compaction_trigger:
            current_messages, events_tier, current_util = await self._tool_pass(
                current_messages,
                budget,
                config,
                token_counter,
                store,
                current_util,
                hot_set=hot_set,
            )
            events.extend(events_tier)
            if current_util < config.tool_compaction_trigger:
                return current_messages, events

        # Tier 2 — conversation eviction (Major GC).
        if current_util >= config.compaction_trigger:
            current_messages, events_tier, current_util = await self._conv_pass(
                current_messages,
                budget,
                config,
                token_counter,
                store,
                hot_set,
                current_util,
            )
            events.extend(events_tier)
            if current_util < config.tool_compaction_trigger:
                return current_messages, events

        # Tier 3 — emergency reduction (Full GC).
        if current_util >= config.emergency_trigger:
            current_messages, events_tier, current_util = await self._emergency_pass(
                current_messages,
                budget,
                config,
                token_counter,
                store,
                hot_set,
                current_util,
            )
            events.extend(events_tier)

        return current_messages, events

    # ------------------------------------------------------------------
    # Post-LLM masking (replaces ObservationMasker)
    # ------------------------------------------------------------------

    def mask(
        self,
        messages: list[ChatMessage],
        token_counter: TokenCounter,
        store: ContentStore,
    ) -> tuple[list[ChatMessage], int, int]:
        """Replace consumed tool results with structured markers.

        A tool result is "consumed" when it appears before the most
        recent assistant message — the model has already seen it and
        responded.  Tool results after the last assistant stay
        untouched (the model has not consumed them yet).

        Args:
            messages: Conversation messages (not mutated).
            token_counter: For computing freed tokens.
            store: Where to offload full content for later recall.

        Returns:
            ``(new_messages, observations_masked, tokens_freed)``.
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
                    tool_name=tool_name,
                )

                originating_call = (
                    tc_index.get(msg.tool_call_id) if msg.tool_call_id else None
                )
                args_preview = _build_args_preview(originating_call)

                # Context Mgmt v2 — Step 4: pick marker shape based on
                # original size.  The richer template (with inline
                # preview) is the loop-breaker for LARGE results — it
                # gives the model enough working memory to avoid
                # re-fetching after the result is offloaded.  But for
                # SMALL results, that same template adds ~150 tokens
                # of chrome on top of the existing content, which is a
                # net loss.  Small results don't drive context
                # pressure anyway, so we use the legacy lean marker
                # (just metadata) which freed tokens reliably in v1
                # and v2 Steps 1-3.  Threshold: only attach a preview
                # when the content is large enough that head-slicing
                # actually truncates it.
                use_preview = len(msg.content) > _PREVIEW_ENABLE_MIN_CHARS
                if use_preview:
                    content_preview = _build_content_preview(msg.content)
                    marker = build_marker(
                        tool_name=tool_name,
                        args_preview=args_preview,
                        key=key,
                        tokens=original_tokens,
                        preview=content_preview,
                    )
                else:
                    marker = build_marker(
                        tool_name=tool_name,
                        args_preview=args_preview,
                        key=key,
                        tokens=original_tokens,
                    )
                marker_tokens = token_counter.count(marker)
                # ``max(0, ...)`` keeps the 'no-benefit' case as a
                # zero-saving mask rather than skipping it — preserves
                # the v1/v2-Step-3 contract that masking happens
                # whenever a tool message is consumed, even on
                # degenerate inputs (e.g. tests using "x"*N which
                # tokenizers compress to fewer tokens than the marker
                # chrome itself).
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

    # ------------------------------------------------------------------
    # Internal passes — one per old strategy
    # ------------------------------------------------------------------

    async def _tool_pass(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None,
        starting_util: float,
        *,
        hot_set: frozenset[str] | None = None,
    ) -> tuple[list[ChatMessage], list[CompactionEvent], float]:
        """Per-tool-result offload / truncate (Tier 1, formerly ToolResultCompactor).

        Two steps:

        1. **Static brake** — any single result above
           ``config.tool_result_threshold`` is offloaded (or truncated when
           offloading is off).  Unchanged behaviour.
        2. **Adaptive squeeze** — if the transcript is still above the
           conversation trigger *and* offloading is available, the largest
           remaining raw tool results are offloaded largest-first until the
           transcript is back under the Tier-1 trigger.  The threshold in
           step 1 is absolute; this step is relative to the window, so a
           *round* of medium results (nine 2K documents in a 12K window)
           no longer sails past Tier 1 untouched and lands in emergency
           eviction, where the model loses evidence it never got to read.
           Every squeezed result stays recallable and keeps a preview; the
           receipt tells the model not to re-run the tool.
        """
        from nucleusiq.agents.chat_models import ChatMessage as CM

        t0 = time.perf_counter()
        tokens_before = token_counter.count_messages(messages)

        compacted: list[ChatMessage] = []
        total_freed = 0
        artifacts_offloaded = 0

        for msg in messages:
            if msg.role != "tool" or not isinstance(msg.content, str):
                compacted.append(msg)
                continue

            content_tokens = token_counter.count(msg.content)
            if content_tokens <= config.tool_result_threshold:
                compacted.append(msg)
                continue

            if config.enable_offloading and store is not None:
                new_content, freed = _offload_tool_content(
                    msg.content,
                    content_tokens,
                    msg.name or "tool",
                    token_counter,
                    store,
                )
                artifacts_offloaded += 1
            else:
                new_content, freed = _truncate_tool_content(
                    msg.content, content_tokens, token_counter
                )

            total_freed += freed
            compacted.append(
                CM(
                    role=msg.role,
                    content=new_content,
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                )
            )

        # Step 2 — adaptive squeeze.
        squeezed = 0
        if (
            config.enable_offloading
            and store is not None
            and budget.effective_limit > 0
        ):
            tokens_now = token_counter.count_messages(compacted)
            target = int(config.tool_compaction_trigger * budget.effective_limit)
            excess = tokens_now - target
            if (
                tokens_now / budget.effective_limit >= config.compaction_trigger
                and excess > 0
            ):
                hot = hot_set or frozenset()
                last_assistant = _last_assistant_index(compacted)
                candidates: list[tuple[int, int, int]] = []  # (-tokens, seen_rank, idx)
                for idx, msg in enumerate(compacted):
                    if msg.role != "tool" or not isinstance(msg.content, str):
                        continue
                    if _is_receipt(msg.content):
                        continue
                    if hot and _group_touches_hot_ref([msg], hot):
                        continue
                    tokens = token_counter.count(msg.content)
                    if tokens < _ADAPTIVE_OFFLOAD_MIN_TOKENS:
                        continue
                    # Results the model already responded to go first;
                    # unseen results only when that is not enough.
                    seen_rank = 0 if idx < last_assistant else 1
                    candidates.append((seen_rank, -tokens, idx))
                candidates.sort()
                for seen_rank, neg_tokens, idx in candidates:
                    if excess <= 0:
                        break
                    msg = compacted[idx]
                    assert isinstance(msg.content, str)
                    offload = (
                        _offload_tool_content
                        if seen_rank == 0
                        else _offload_unseen_tool_content
                    )
                    new_content, freed = offload(
                        msg.content,
                        -neg_tokens,
                        msg.name or "tool",
                        token_counter,
                        store,
                    )
                    compacted[idx] = CM(
                        role=msg.role,
                        content=new_content,
                        name=msg.name,
                        tool_call_id=msg.tool_call_id,
                    )
                    excess -= freed
                    total_freed += freed
                    artifacts_offloaded += 1
                    squeezed += 1
                if squeezed:
                    logger.info(
                        "Adaptive tool-result offload: %d result(s) moved to the "
                        "content store to fit the working budget (%d → ~%d tokens)",
                        squeezed,
                        tokens_now,
                        max(0, tokens_now - total_freed),
                    )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        tokens_after = token_counter.count_messages(compacted)

        events = [
            CompactionEvent(
                strategy="tool_result_compactor",
                trigger_utilization=starting_util,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_freed=total_freed,
                artifacts_offloaded=artifacts_offloaded,
                duration_ms=elapsed_ms,
            )
        ]

        new_util = (
            tokens_after / budget.effective_limit if budget.effective_limit > 0 else 1.0
        )
        return compacted, events, new_util

    async def _conv_pass(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None,
        hot_set: frozenset[str] | None,
        starting_util: float,
    ) -> tuple[list[ChatMessage], list[CompactionEvent], float]:
        """Conversation eviction (Tier 2, formerly ConversationCompactor)."""
        from nucleusiq.agents.chat_models import ChatMessage as CM

        t0 = time.perf_counter()
        tokens_before = token_counter.count_messages(messages)

        if len(messages) <= 2:
            return messages, [], starting_util

        pinned_head, evictable, pinned_tail = _partition_for_conversation(
            messages,
            config.preserve_recent_turns,
            hot_set=hot_set,
        )

        if not evictable:
            return messages, [], starting_util

        evicted_tokens = sum(
            token_counter.count(m.content) if isinstance(m.content, str) else 0
            for m in evictable
        )
        evidence, newly_offloaded = _collect_addressable_evidence(
            evictable, store, token_counter
        )
        catalog = _build_evidence_catalog(evidence)

        if config.enable_summarization:
            marker_content = _build_structured_summary(evictable, evicted_tokens)
        else:
            marker_content = (
                f"[{len(evictable)} earlier messages compacted — "
                f"~{evicted_tokens} tokens freed. "
                f"Recent {len(pinned_tail)} messages preserved.]"
            )

        marker = CM(role="system", content=marker_content)
        compacted = list(pinned_head)
        if catalog is not None:
            compacted.append(catalog)
        compacted.extend([marker, *pinned_tail])

        marker_cost = token_counter.count(marker_content)
        catalog_cost = (
            token_counter.count(catalog.content)
            if catalog is not None and isinstance(catalog.content, str)
            else 0
        )
        freed = max(0, evicted_tokens - marker_cost - catalog_cost)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        tokens_after = token_counter.count_messages(compacted)

        events = [
            CompactionEvent(
                strategy="conversation_compactor",
                trigger_utilization=starting_util,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_freed=freed,
                artifacts_offloaded=newly_offloaded,
                duration_ms=elapsed_ms,
            )
        ]
        new_util = (
            tokens_after / budget.effective_limit if budget.effective_limit > 0 else 1.0
        )
        return compacted, events, new_util

    async def _emergency_pass(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None,
        hot_set: frozenset[str] | None,
        starting_util: float,
        *,
        out_warnings: list[str] | None = None,
    ) -> tuple[list[ChatMessage], list[CompactionEvent], float]:
        """Emergency reduction (Tier 3, formerly EmergencyCompactor).

        ``out_warnings`` (optional): caller-supplied list that the
        pass appends a single human-readable warning string to,
        describing how many messages were dropped and how many were
        rescued by the hot set.  Used by the back-compat
        :class:`EmergencyCompactor` wrapper to surface the warning on
        :class:`CompactionResult`; the engine's prepare path doesn't
        consume it (the warning text is also logged).
        """
        from nucleusiq.agents.chat_models import ChatMessage as CM

        t0 = time.perf_counter()
        tokens_before = token_counter.count_messages(messages)

        # I1 — pin the task head (system messages + the user messages
        # that precede the first assistant turn).
        task_head, rest = _split_task_head(messages)

        # Build atomic groups (assistant + tool-result clusters).
        groups: list[list[ChatMessage]] = []
        i = 0
        while i < len(rest):
            msg = rest[i]
            group = [msg]
            if msg.role == "assistant" and getattr(msg, "tool_calls", None):
                j = i + 1
                while j < len(rest) and rest[j].role == "tool":
                    group.append(rest[j])
                    j += 1
                i = j
            else:
                i += 1
            groups.append(group)

        tail_group_count = 1
        kept_groups = groups[-tail_group_count:]
        dropped_groups = groups[:-tail_group_count]
        kept_tail = [m for g in kept_groups for m in g]
        if not dropped_groups and not _tail_needs_squeeze(
            task_head, kept_tail, budget, token_counter
        ):
            # Nothing to drop and the tail already fits: no-op (legacy).
            return messages, [], starting_util

        hot = hot_set or frozenset()
        rescued: list[ChatMessage] = []
        evicted: list[ChatMessage] = []
        for g in dropped_groups:
            if hot and _group_touches_hot_ref(g, hot):
                rescued.extend(g)
                continue
            evicted.extend(g)

        # Tool results that no assistant turn has followed were never read
        # by the model.  Evicting them is the one compaction outcome that
        # can silently turn a grounded answer into a guess, so it is
        # counted and surfaced (``CompactionEvent.unseen_evicted`` →
        # ``EVIDENCE_EVICTED_UNSEEN`` finding).
        last_assistant = _last_assistant_index(messages)
        position = {id(m): i for i, m in enumerate(messages)}
        unseen_evicted = sum(
            1
            for m in evicted
            if m.role == "tool"
            and not _is_receipt(m.content)
            and position.get(id(m), -1) > last_assistant
        )

        evidence, newly_offloaded = _collect_addressable_evidence(
            evicted, store, token_counter
        )
        catalog = _build_evidence_catalog(evidence)

        dropped_tokens = sum(
            token_counter.count(m.content) if isinstance(m.content, str) else 0
            for m in evicted
        )

        marker_text = _EMERGENCY_MARKER.format(
            util=budget.utilization,
            dropped=len(evicted),
            tokens=dropped_tokens,
            kept=len(kept_tail),
        )
        marker = CM(role="system", content=marker_text)
        marker_cost = token_counter.count(marker_text)
        catalog_cost = (
            token_counter.count(catalog.content)
            if catalog is not None and isinstance(catalog.content, str)
            else 0
        )

        head: list[ChatMessage] = list(task_head)
        head.extend(rescued)
        if catalog is not None:
            head.append(catalog)

        # The kept tail can itself be larger than the whole window — one
        # round of tool calls that returned more than the model can read.
        # Dropping everything else does not help then; the tail's own
        # results must shrink.  Lossless (offload + preview + recall hint)
        # when a store is available, lossy head/tail truncation otherwise.
        # Only the lossy path counts as "unseen evicted": the model still
        # gets a preview and a handle in the lossless one.
        squeezed_tail, tail_freed, tail_offloaded, tail_lost_unseen = _squeeze_tail(
            kept_tail,
            head_tokens=token_counter.count_messages(head) + marker_cost,
            budget=budget,
            config=config,
            token_counter=token_counter,
            store=store,
            hot_set=hot,
            last_assistant=last_assistant,
            position=position,
        )
        kept_tail = squeezed_tail
        newly_offloaded += tail_offloaded
        unseen_evicted += tail_lost_unseen

        compacted = head + [marker] + kept_tail if evicted else head + kept_tail
        freed = max(0, dropped_tokens - marker_cost - catalog_cost) + tail_freed

        # Note: ``CompactionEvent`` (telemetry) has no ``warnings``
        # field — the v1 engine's ``getattr(..., 'warnings', ())``
        # call has always returned ``()`` for emergency events.  The
        # warning text is therefore logged here for operator
        # visibility but not propagated through the event object,
        # matching pre-Step-3 behaviour.
        warning_text = (
            f"Emergency compaction: dropped {len(evicted)} messages "
            f"(~{dropped_tokens} tokens) at {budget.utilization:.0%} utilization"
            + (
                f"; rescued {len(rescued)} hot-recalled tool message(s)"
                if rescued
                else ""
            )
            + (f"; catalogued {len(evidence)} recallable ref(s)" if evidence else "")
            + (
                f"; squeezed {tail_offloaded} oversized result(s) in the current turn"
                if tail_offloaded
                else ""
            )
            + (
                f"; {unseen_evicted} tool result(s) lost before the model saw them"
                if unseen_evicted
                else ""
            )
        )
        logger.warning(warning_text)
        if out_warnings is not None:
            out_warnings.append(warning_text)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        tokens_after = token_counter.count_messages(compacted)

        events = [
            CompactionEvent(
                strategy="emergency_compactor",
                trigger_utilization=starting_util,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_freed=freed,
                artifacts_offloaded=newly_offloaded,
                duration_ms=elapsed_ms,
                unseen_evicted=unseen_evicted,
            )
        ]
        new_util = (
            tokens_after / budget.effective_limit if budget.effective_limit > 0 else 1.0
        )
        return compacted, events, new_util


# =====================================================================
# Back-compat wrappers (Step 3 — keep the v1 strategy class names so
# direct callers / tests keep working without rewrites).  Each wrapper
# routes through the consolidated :class:`Compactor`; no logic lives
# here.  These will be deprecated and removed in v0.9.
# =====================================================================


class ObservationMasker:
    """Back-compat wrapper — delegates to :meth:`Compactor.mask`.

    Kept so callers that constructed ``ObservationMasker()`` directly
    (mostly older tests and the v1 engine path) continue to work.
    The masker has no instance state, so the wrapper holds a single
    shared :class:`Compactor` which is itself stateless.
    """

    __slots__ = ("_compactor",)

    def __init__(self) -> None:
        self._compactor = Compactor()

    def mask(
        self,
        messages: list[ChatMessage],
        token_counter: TokenCounter,
        store: ContentStore,
    ) -> tuple[list[ChatMessage], int, int]:
        return self._compactor.mask(messages, token_counter, store)


class ToolResultCompactor(CompactionStrategy):
    """Back-compat wrapper — Tier 1 (Minor GC) per-tool offload pass."""

    __slots__ = ("_compactor",)

    def __init__(self) -> None:
        self._compactor = Compactor()

    @property
    def name(self) -> str:
        return "tool_result_compactor"

    async def compact(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None = None,
        *,
        hot_set: frozenset[str] | None = None,  # noqa: ARG002 — unused
    ) -> CompactionResult:
        compacted, events, _new_util = await self._compactor._tool_pass(
            list(messages),
            budget,
            config,
            token_counter,
            store,
            budget.utilization,
        )
        ev = events[0] if events else None
        return CompactionResult(
            messages=compacted,
            tokens_freed=ev.tokens_freed if ev else 0,
            tokens_remaining=max(0, budget.allocated - (ev.tokens_freed if ev else 0)),
            strategy_used=self.name,
            artifacts_offloaded=ev.artifacts_offloaded if ev else 0,
        )


class ConversationCompactor(CompactionStrategy):
    """Back-compat wrapper — Tier 2 (Major GC) conversation eviction pass."""

    __slots__ = ("_compactor",)

    def __init__(self) -> None:
        self._compactor = Compactor()

    @property
    def name(self) -> str:
        return "conversation_compactor"

    async def compact(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None = None,
        *,
        hot_set: frozenset[str] | None = None,
    ) -> CompactionResult:
        original_count = len(messages)
        compacted, events, _new_util = await self._compactor._conv_pass(
            list(messages),
            budget,
            config,
            token_counter,
            store,
            hot_set,
            budget.utilization,
        )
        ev = events[0] if events else None
        # ``entries_removed`` reports how many of the input messages
        # were removed (the marker line is added separately).  v1
        # exposed this so tests could assert ``entries_removed > 0``.
        if ev is None or compacted is messages:
            return CompactionResult(
                messages=list(messages),
                tokens_freed=0,
                tokens_remaining=budget.allocated,
                strategy_used=self.name,
            )
        # Marker is one extra message inserted; recover original drop
        # count from the length delta + 1.
        entries_removed = max(0, original_count - len(compacted) + 1)
        return CompactionResult(
            messages=compacted,
            tokens_freed=ev.tokens_freed,
            tokens_remaining=max(0, budget.allocated - ev.tokens_freed),
            strategy_used=self.name,
            entries_removed=entries_removed,
            summaries_inserted=1 if config.enable_summarization else 0,
        )

    @staticmethod
    def _partition(
        messages: list[ChatMessage],
        preserve_recent: int,
        *,
        hot_set: frozenset[str] | None = None,
    ) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage]]:
        """Back-compat: tests that called the static helper directly."""
        return _partition_for_conversation(messages, preserve_recent, hot_set=hot_set)

    @staticmethod
    def _build_structured_summary(
        evicted: list[ChatMessage], evicted_tokens: int
    ) -> str:
        """Back-compat: tests that called the static helper directly."""
        return _build_structured_summary(evicted, evicted_tokens)


class EmergencyCompactor(CompactionStrategy):
    """Back-compat wrapper — Tier 3 (Full GC) emergency reduction pass."""

    __slots__ = ("_compactor",)

    def __init__(self) -> None:
        self._compactor = Compactor()

    @property
    def name(self) -> str:
        return "emergency_compactor"

    async def compact(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None = None,
        *,
        hot_set: frozenset[str] | None = None,
    ) -> CompactionResult:
        original_count = len(messages)
        captured_warnings: list[str] = []
        compacted, events, _new_util = await self._compactor._emergency_pass(
            list(messages),
            budget,
            config,
            token_counter,
            store,
            hot_set,
            budget.utilization,
            out_warnings=captured_warnings,
        )
        ev = events[0] if events else None
        if ev is None or compacted is messages or len(compacted) == len(messages):
            # No reduction possible — preserve the v1 short-circuit
            # behaviour (return original messages, surface a warning).
            return CompactionResult(
                messages=list(messages),
                tokens_freed=0,
                tokens_remaining=budget.allocated,
                strategy_used=self.name,
                warnings=("emergency_compactor: nothing to evict",),
            )
        entries_removed = max(0, original_count - len(compacted) + 1)
        return CompactionResult(
            messages=compacted,
            tokens_freed=ev.tokens_freed,
            tokens_remaining=max(0, budget.allocated - ev.tokens_freed),
            strategy_used=self.name,
            entries_removed=entries_removed,
            warnings=tuple(captured_warnings),
        )


class CompactionPipeline:
    """Back-compat wrapper — orchestrates tiers via the Compactor.

    The v1 pipeline accepted ``[(threshold, strategy), ...]`` tuples
    and ran each strategy whose threshold was reached.  Step 3 keeps
    the same constructor signature so existing call sites (and tests
    that build a pipeline with custom strategies) work unchanged.

    Internally the wrapper just runs each strategy in order; the new
    :class:`Compactor` is the preferred path for new code.
    """

    __slots__ = ("_tiers",)

    def __init__(self, tiers: list[tuple[float, CompactionStrategy]]) -> None:
        self._tiers = sorted(tiers, key=lambda t: t[0])

    async def run(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None = None,
        *,
        hot_set: frozenset[str] | None = None,
    ) -> tuple[list[ChatMessage], list[CompactionEvent]]:
        current_messages = list(messages)
        events: list[CompactionEvent] = []
        current_util = budget.utilization

        for trigger_threshold, strategy in self._tiers:
            if current_util < trigger_threshold:
                continue

            t0 = time.perf_counter()
            tokens_before = token_counter.count_messages(current_messages)

            result = await strategy.compact(
                current_messages,
                budget,
                config,
                token_counter,
                store,
                hot_set=hot_set,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            tokens_after = token_counter.count_messages(result.messages)

            events.append(
                CompactionEvent(
                    strategy=strategy.name,
                    trigger_utilization=current_util,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    tokens_freed=result.tokens_freed,
                    artifacts_offloaded=result.artifacts_offloaded,
                    duration_ms=elapsed_ms,
                )
            )

            current_messages = result.messages

            if budget.effective_limit > 0:
                current_util = tokens_after / budget.effective_limit
            else:
                current_util = 1.0

            if current_util < config.tool_compaction_trigger:
                break

        return current_messages, events

    @property
    def tier_count(self) -> int:
        return len(self._tiers)


__all__ = [
    "Compactor",
    "CompactionEvent",
    "CompactionPipeline",
    "CompactionResult",
    "CompactionStrategy",
    "ConversationCompactor",
    "EmergencyCompactor",
    "MASK_PREFIX",
    "ObservationMasker",
    "ToolResultCompactor",
    "build_marker",
]
