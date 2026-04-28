"""Tests for Context Mgmt v2 — Step 2 (recall tools).

The recall tools (``recall_tool_result`` + ``list_recalled_evidence``)
are the model-facing half of the "retrieval, not eviction" loop:

* When :class:`ObservationMasker` offloads an evidence-shaped tool
  result, it leaves a marker carrying a ``ref`` key in the
  conversation.
* The model uses ``recall_tool_result(ref=...)`` to pull the full
  content back when it needs to quote / cite / re-read.
* ``list_recalled_evidence`` lets the model browse what's recallable
  when it has lost track of refs.

Key invariants under test
-------------------------
**I2** — recall tools must never raise.  Every error condition
returns a ``[recall_error: ...]`` string so the model has a recovery
path.

* Telemetry: every successful recall increments
  ``RecallTracker.recall_count`` and ``total_recalled_tokens``.
* Hot-set integration: a recently recalled ref appears in
  ``RecallTracker.hot_set`` and survives the turn boundary.
* Per-call truncation: huge artefacts are capped via
  ``tool_result_per_call_max_chars`` so a single recall cannot blow
  the context window.
"""

from __future__ import annotations

import pytest
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.policy import ContextPolicy
from nucleusiq.agents.context.recall_tools import (
    LIST_RECALLED_EVIDENCE_TOOL_NAME,
    RECALL_TOOL_NAME,
    build_list_recalled_evidence,
    build_recall_tool_result,
    build_recall_tools,
    is_recall_tool_name,
)

# ---------------------------------------------------------------------- #
# Fixtures                                                                 #
# ---------------------------------------------------------------------- #


def _engine(**overrides) -> ContextEngine:
    """Build a fresh engine with sensible defaults for recall tests."""
    cfg = ContextConfig(
        max_context_tokens=100_000,
        tool_result_per_call_max_chars=overrides.pop(
            "tool_result_per_call_max_chars", 50_000
        ),
        **overrides,
    )
    return ContextEngine(cfg)


# ---------------------------------------------------------------------- #
# 1. Tool-name registry helpers                                            #
# ---------------------------------------------------------------------- #


class TestIsRecallToolName:
    """``is_recall_tool_name`` is the cheap predicate execution modes
    use to skip ``max_tool_calls`` accounting for recall calls.  It
    must be tolerant of bad input (None, empty string).
    """

    def test_recognises_recall_tool_result(self):
        assert is_recall_tool_name(RECALL_TOOL_NAME)

    def test_recognises_list_recalled_evidence(self):
        assert is_recall_tool_name(LIST_RECALLED_EVIDENCE_TOOL_NAME)

    def test_rejects_arbitrary_tool_names(self):
        assert not is_recall_tool_name("read_pdf")
        assert not is_recall_tool_name("search_web")

    def test_rejects_none_and_empty(self):
        assert not is_recall_tool_name(None)
        assert not is_recall_tool_name("")


# ---------------------------------------------------------------------- #
# 2. recall_tool_result — happy paths                                     #
# ---------------------------------------------------------------------- #


class TestRecallToolResultHappyPath:
    @pytest.mark.asyncio
    async def test_retrieves_offloaded_content(self):
        engine = _engine()
        engine.store.store(
            "obs:read_pdf:abc123",
            "<full PDF content>\nLine 2\nLine 3",
            original_tokens=42,
        )
        tool = build_recall_tool_result(engine)

        result = await tool.execute(ref="obs:read_pdf:abc123")

        assert "<full PDF content>" in result
        assert "Line 2" in result

    @pytest.mark.asyncio
    async def test_records_recall_in_tracker(self):
        engine = _engine()
        engine.store.store("ref-a", "content-a", original_tokens=10)
        tool = build_recall_tool_result(engine)

        await tool.execute(ref="ref-a")

        assert engine.recall_tracker.recall_count == 1
        assert engine.recall_tracker.total_recalled_tokens > 0

    @pytest.mark.asyncio
    async def test_recall_appears_in_hot_set(self):
        engine = _engine()
        engine.store.store("ref-hot", "content-hot", original_tokens=10)
        tool = build_recall_tool_result(engine)

        await tool.execute(ref="ref-hot")

        # Within the same turn, lookback=0 already includes it.
        assert "ref-hot" in engine.recall_tracker.hot_set(lookback_turns=0)

        # Survives a turn boundary when lookback is wide enough.
        engine.recall_tracker.mark_turn_completed()
        assert "ref-hot" in engine.recall_tracker.hot_set(lookback_turns=2)

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_ref(self):
        """Models occasionally quote refs with stray spaces — accept them."""
        engine = _engine()
        engine.store.store("ref-x", "content-x", original_tokens=5)
        tool = build_recall_tool_result(engine)

        result = await tool.execute(ref="  ref-x  ")
        assert "content-x" in result

    @pytest.mark.asyncio
    async def test_tool_metadata_marks_recall_as_ephemeral(self):
        """Recall outputs are EPHEMERAL — masker must not re-offload them."""
        engine = _engine()
        tool = build_recall_tool_result(engine)
        assert tool.context_policy is ContextPolicy.EPHEMERAL
        assert tool.name == RECALL_TOOL_NAME


# ---------------------------------------------------------------------- #
# 3. recall_tool_result — error paths (I2: never raise)                   #
# ---------------------------------------------------------------------- #


class TestRecallToolResultErrorPaths:
    @pytest.mark.asyncio
    async def test_missing_ref_returns_recall_error_marker(self):
        engine = _engine()
        engine.store.store("ref-real", "content", original_tokens=5)
        tool = build_recall_tool_result(engine)

        result = await tool.execute(ref="ref-does-not-exist")

        assert isinstance(result, str)
        assert result.startswith("[recall_error:")
        # Hint with available refs is part of the recovery contract.
        assert "ref-real" in result

    @pytest.mark.asyncio
    async def test_missing_ref_does_not_increment_recall_count(self):
        engine = _engine()
        tool = build_recall_tool_result(engine)

        await tool.execute(ref="ghost")

        assert engine.recall_tracker.recall_count == 0

    @pytest.mark.asyncio
    async def test_empty_ref_returns_recall_error(self):
        engine = _engine()
        tool = build_recall_tool_result(engine)

        result = await tool.execute(ref="")
        assert result.startswith("[recall_error:")
        assert "non-empty string" in result

    @pytest.mark.asyncio
    async def test_whitespace_ref_returns_recall_error(self):
        engine = _engine()
        tool = build_recall_tool_result(engine)

        result = await tool.execute(ref="   ")
        assert result.startswith("[recall_error:")

    @pytest.mark.asyncio
    async def test_non_string_ref_returns_recall_error(self):
        """Defensive: validators upstream should catch this, but if a
        provider sends ``ref=None`` we still must not raise."""
        engine = _engine()
        tool = build_recall_tool_result(engine)

        result = await tool.execute(ref=None)
        assert result.startswith("[recall_error:")

    @pytest.mark.asyncio
    async def test_empty_store_lists_no_available_refs(self):
        """The error message must still be well-formed when the store is empty."""
        engine = _engine()
        tool = build_recall_tool_result(engine)

        result = await tool.execute(ref="anything")
        assert result.startswith("[recall_error:")
        # No refs available → empty list hint, not a crash.
        assert "(none)" in result or "[]" in result or "Available" in result


# ---------------------------------------------------------------------- #
# 4. recall_tool_result — per-call truncation                             #
# ---------------------------------------------------------------------- #


class TestRecallToolResultTruncation:
    @pytest.mark.asyncio
    async def test_oversize_content_is_truncated(self):
        cap = 1_000
        engine = _engine(tool_result_per_call_max_chars=cap)
        big = "x" * (cap * 4)  # ~4× the cap
        engine.store.store("big", big, original_tokens=cap * 4)
        tool = build_recall_tool_result(engine)

        result = await tool.execute(ref="big")
        # Cap + a short truncation marker.
        assert len(result) <= cap + 200
        assert "truncated" in result.lower()

    @pytest.mark.asyncio
    async def test_under_cap_content_returned_verbatim(self):
        cap = 10_000
        engine = _engine(tool_result_per_call_max_chars=cap)
        small = "abc-123"
        engine.store.store("small", small, original_tokens=2)
        tool = build_recall_tool_result(engine)

        result = await tool.execute(ref="small")
        assert result == small


# ---------------------------------------------------------------------- #
# 5. list_recalled_evidence                                               #
# ---------------------------------------------------------------------- #


class TestListRecalledEvidence:
    @pytest.mark.asyncio
    async def test_empty_store_returns_empty_list(self):
        engine = _engine()
        tool = build_list_recalled_evidence(engine)
        result = await tool.execute()
        assert result == []

    @pytest.mark.asyncio
    async def test_lists_every_offloaded_artefact(self):
        engine = _engine()
        engine.store.store("ref-a", "alpha content", original_tokens=5)
        engine.store.store("ref-b", "beta content", original_tokens=5)
        tool = build_list_recalled_evidence(engine)

        result = await tool.execute()

        assert isinstance(result, list)
        assert len(result) == 2
        refs = {entry["ref"] for entry in result}
        assert refs == {"ref-a", "ref-b"}
        for entry in result:
            assert "size_chars" in entry and entry["size_chars"] > 0
            assert "preview" in entry

    @pytest.mark.asyncio
    async def test_listing_does_not_record_a_recall(self):
        """Listing is a metadata op — must not bump the recall counter."""
        engine = _engine()
        engine.store.store("ref-a", "alpha", original_tokens=3)
        tool = build_list_recalled_evidence(engine)

        await tool.execute()

        assert engine.recall_tracker.recall_count == 0

    @pytest.mark.asyncio
    async def test_tool_metadata_marks_listing_as_ephemeral(self):
        engine = _engine()
        tool = build_list_recalled_evidence(engine)
        assert tool.context_policy is ContextPolicy.EPHEMERAL
        assert tool.name == LIST_RECALLED_EVIDENCE_TOOL_NAME


# ---------------------------------------------------------------------- #
# 6. build_recall_tools convenience                                       #
# ---------------------------------------------------------------------- #


class TestBuildRecallTools:
    def test_returns_both_tools(self):
        engine = _engine()
        tools = build_recall_tools(engine)
        names = {t.name for t in tools}
        assert names == {RECALL_TOOL_NAME, LIST_RECALLED_EVIDENCE_TOOL_NAME}

    def test_each_call_produces_fresh_bindings(self):
        """Factories — different calls → different tool instances."""
        engine = _engine()
        a = build_recall_tools(engine)
        b = build_recall_tools(engine)
        assert a is not b
        assert a[0] is not b[0]
