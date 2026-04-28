"""Context Mgmt v2 — Step 2: agent-facing memory recall tools.

When :class:`ContextEngine` offloads an evidence-shaped tool result
under memory pressure (Step 3 implementation; Step 2 only sets the
infrastructure up), the conversation is left with a slim marker
that carries a ``ref`` key and a hint::

    [observation consumed]
    tool: read_pdf_page
    args: {"path": "tcs_fy25.pdf", "page": 17}
    ref: obs:read_pdf_page:a3f21b
    size: ~4300 tokens
    To retrieve: call recall_tool_result(ref="obs:read_pdf_page:a3f21b")

The two tools in this module make that hint actionable:

* :func:`build_recall_tool_result` — given a ``ref``, return the full
  original content from the :class:`ContentStore`.
* :func:`build_list_recalled_evidence` — return metadata for every
  offloaded artefact so the model can browse what's recallable.

Both are *factories*: they capture the engine (and thus its store +
recall tracker) in a closure so the tools are bound to one
execution.  Using factories instead of module-level singletons keeps
the lifecycle clean — every ``Agent.execute()`` gets its own pair of
tools, garbage-collected with the engine.

Discovery (Q2 from the design doc — option a + c):

* (a) Tool specs are auto-injected into every LLM call when an
  engine is attached, so the recall tools always appear in the tool
  list — even before the first offload.  An empty store means
  ``list_recalled_evidence`` returns ``[]``; zero overhead.
* (c) The marker itself spells out the call signature, so the model
  learns the pattern from the marker, not by reading tool docs.

Recall results are themselves classified ``EPHEMERAL`` via the
``@tool`` decoration — recursively offloading recall outputs would
defeat the purpose and risk loops.

Error handling is bulletproof (invariant **I2**): missing refs and
empty stores return informative strings rather than raising
exceptions, because an exception in a tool call would surface to the
model as an opaque failure and leave it without a recovery path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nucleusiq.agents.context.policy import ContextPolicy
from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.decorators import DecoratedTool

if TYPE_CHECKING:
    from nucleusiq.agents.context.engine import ContextEngine

logger = logging.getLogger(__name__)


__all__ = [
    "RECALL_TOOL_NAME",
    "LIST_RECALLED_EVIDENCE_TOOL_NAME",
    "build_recall_tool_result",
    "build_list_recalled_evidence",
    "build_recall_tools",
    "is_recall_tool_name",
]


#: Stable name for the recall tool.  Exposed as a constant so the
#: rest of the framework (tool injection, ``max_tool_calls``
#: bookkeeping, etc.) can identify recall calls without string-typing.
RECALL_TOOL_NAME: str = "recall_tool_result"

#: Stable name for the listing tool.
LIST_RECALLED_EVIDENCE_TOOL_NAME: str = "list_recalled_evidence"

#: Tool names that should not count toward ``max_tool_calls`` because
#: they are memory operations, not external actions.  See §6.4 of the
#: redesign document.
_RECALL_TOOL_NAMES: frozenset[str] = frozenset(
    {RECALL_TOOL_NAME, LIST_RECALLED_EVIDENCE_TOOL_NAME}
)


def is_recall_tool_name(tool_name: str | None) -> bool:
    """Return True if ``tool_name`` is one of the auto-injected recall tools.

    Used by the execution modes to skip incrementing
    ``max_tool_calls`` for recall invocations: the budget for recall
    is the *context window*, not the tool-call quota.  See §6.4 of
    ``CONTEXT_MANAGEMENT_V2_REDESIGN.md``.
    """
    return tool_name in _RECALL_TOOL_NAMES if tool_name else False


# ====================================================================== #
# Internal helpers                                                         #
# ====================================================================== #


def _truncate_for_recall(content: str, *, max_chars: int) -> str:
    """Cap recall payload size — never blow the context window.

    A misbehaving model that recalls the same 100K-char artefact ten
    times in a row would otherwise blow the response window in a
    single turn.  The cap is the same one used by Critic/Refiner
    rehydration (``tool_result_per_call_max_chars``) for parity.
    """
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    return (
        content[:max_chars]
        + "\n[... truncated; content exceeded the per-call recall cap]"
    )


def _format_recall_error(message: str) -> str:
    """Wrap an error message in a marker the model can recognise.

    Returning a string (not raising) is the contract — the model
    needs a recovery path.  The ``[recall_error: ...]`` envelope
    makes the failure mode obvious in traces.
    """
    return f"[recall_error: {message}]"


def _safe_keys(engine: ContextEngine) -> list[str]:
    """Return the store's keys, defensively.

    ``ContentStore.keys`` is a normal list method on the bundled
    in-memory backend, but downstream subclasses might fail or
    return weird types.  Wrap defensively because this runs in the
    hot path of every recall.
    """
    try:
        keys = engine.store.keys()
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("ContentStore.keys() raised: %s", exc)
        return []
    if not isinstance(keys, list):
        try:
            keys = list(keys)
        except Exception:
            return []
    return keys


# ====================================================================== #
# Tool factories                                                           #
# ====================================================================== #


def build_recall_tool_result(engine: ContextEngine) -> BaseTool:
    """Return a ``BaseTool`` that retrieves an offloaded tool result.

    The returned tool is bound to ``engine`` via closure: it reads
    from ``engine.store`` and records into ``engine.recall_tracker``.
    Factories (rather than module-level instances) keep the
    lifecycle aligned with the engine: one execution = one engine =
    one set of recall tools.

    Args:
        engine: The :class:`ContextEngine` whose store and tracker
            this tool should bind to.  Required (passing ``None``
            would produce a tool that always returns
            "context store not available").

    Returns:
        A :class:`DecoratedTool` whose ``execute(ref=...)`` returns
        either the recovered content or a ``[recall_error: ...]``
        marker.  Never raises.
    """

    async def recall_tool_result(ref: str) -> str:
        """Retrieve the full content of an earlier offloaded tool result.

        Use this when you see an ``[observation consumed]`` marker
        in your context and need the original content back — for
        example to quote a specific passage, cite a number, or
        re-read evidence before drafting a final answer.

        Args:
            ref: The ``ref`` value from the ``[observation
                consumed]`` marker, e.g. ``"obs:read_pdf_page:a3f21b"``.
                Pass it verbatim — refs are stable for the lifetime
                of this conversation.

        Returns:
            The full original content.  If the ref cannot be
            resolved you get a ``[recall_error: ...]`` marker
            describing what went wrong and listing a few available
            refs so you can self-correct.
        """
        if engine is None:
            return _format_recall_error(
                "context store not available in this execution mode"
            )

        if not isinstance(ref, str) or not ref.strip():
            return _format_recall_error(
                "ref must be a non-empty string from an "
                "[observation consumed] marker (the line beginning 'ref: ')"
            )
        ref = ref.strip()

        try:
            content = engine.store.retrieve(ref)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("ContentStore.retrieve(%r) raised: %s", ref, exc)
            return _format_recall_error(
                f"store retrieval failed for ref={ref!r} ({type(exc).__name__})"
            )

        if content is None:
            available = _safe_keys(engine)[:5]
            hint = ", ".join(repr(k) for k in available) or "(none)"
            return _format_recall_error(
                f"ref {ref!r} not found in the context store. "
                f"Available refs include: [{hint}]. "
                f"Call list_recalled_evidence() to see all options."
            )

        # Per-call cap — protect the context window.
        max_chars = int(
            getattr(engine.config, "tool_result_per_call_max_chars", 50_000)
        )
        truncated = _truncate_for_recall(content, max_chars=max_chars)

        # Record telemetry — recall tracker is per-execution, never None.
        try:
            tokens = engine.token_counter.count(truncated)
        except Exception:
            tokens = len(truncated) // 4  # cheap fallback estimate
        engine.recall_tracker.record_recall(ref, tokens=tokens)

        logger.debug(
            "Recall: ref=%s tokens=%d total_recalls=%d",
            ref,
            tokens,
            engine.recall_tracker.recall_count,
        )

        return truncated

    return DecoratedTool(
        recall_tool_result,
        tool_name=RECALL_TOOL_NAME,
        tool_description=(recall_tool_result.__doc__ or "").strip().split("\n\n")[0],
        # Recall outputs are themselves EPHEMERAL — we do not want
        # the masker to evict and recursively offload them.  See
        # §6.5 (recall-loop guard) in the design doc.
        context_policy=ContextPolicy.EPHEMERAL,
    )


def build_list_recalled_evidence(engine: ContextEngine) -> BaseTool:
    """Return a ``BaseTool`` that lists all offloaded evidence.

    Useful when the model wants to *browse* what it can recall
    without committing to a particular ref yet.  Empty store →
    empty list (no error, no surprises).
    """

    async def list_recalled_evidence() -> list[dict[str, Any]]:
        """List every offloaded tool result available for recall.

        Each entry has the shape::

            {
                "ref": "obs:read_pdf_page:a3f21b",
                "size_chars": 18432,
                "preview": "First page of TCS FY25...",
            }

        Use this to discover what's recallable when you don't
        remember a specific ref — for example when the conversation
        has scrolled past several ``[observation consumed]``
        markers.  The list is in insertion order (newest last).

        Returns:
            A list of dicts.  Empty list if nothing is offloaded.
        """
        if engine is None:
            return []

        keys = _safe_keys(engine)
        out: list[dict[str, Any]] = []
        for k in keys:
            try:
                preview = engine.store.preview(k)
                full = engine.store.retrieve(k)
            except Exception:  # pragma: no cover — defensive
                continue
            if full is None:
                continue
            out.append(
                {
                    "ref": k,
                    "size_chars": len(full),
                    "preview": preview or "",
                }
            )
        return out

    return DecoratedTool(
        list_recalled_evidence,
        tool_name=LIST_RECALLED_EVIDENCE_TOOL_NAME,
        tool_description=(list_recalled_evidence.__doc__ or "")
        .strip()
        .split("\n\n")[0],
        context_policy=ContextPolicy.EPHEMERAL,
    )


def build_recall_tools(engine: ContextEngine) -> list[BaseTool]:
    """Convenience: build both recall tools bound to ``engine``.

    The agent setup code calls this once per execution and merges
    the result into ``agent.tools`` / ``Executor.tools`` so the
    model can call ``recall_tool_result`` like any other tool.
    """
    return [
        build_recall_tool_result(engine),
        build_list_recalled_evidence(engine),
    ]
