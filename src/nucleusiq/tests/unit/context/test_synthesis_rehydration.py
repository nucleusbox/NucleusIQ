"""Tests for Context Mgmt v2 — Step 2 (synthesis auto-rehydration).

The synthesis pass calls the LLM with ``tools=None``, so the model
cannot ``recall_tool_result`` to fetch offloaded evidence on its own.
:meth:`ContextEngine.prepare_for_synthesis` closes that gap by walking
the message list newest-first and replacing evidence markers with
their original content — but only as many as fit in the synthesis
budget.

Properties under test (§7 of the v2 redesign doc):

1. **Newest-first**: when only some markers fit, the most recent ones
   are rehydrated first.
2. **Budget-respecting**: rehydration stops as soon as the next
   replacement would exceed ``max_context_tokens - reserve``.
3. **Fail-open**: rehydration never raises.  Missing refs, store
   errors, malformed markers → markers stay as-is, no exception.
4. **Pure**: returns a new list, does not mutate the input.
5. **Idempotent**: re-running on already-rehydrated messages is a
   no-op.
"""

from __future__ import annotations

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.engine import ContextEngine

# ---------------------------------------------------------------------- #
# Fixtures + helpers                                                       #
# ---------------------------------------------------------------------- #


def _engine(
    *,
    max_tokens: int = 100_000,
    response_reserve: int = 8_192,
    per_call_max_chars: int = 50_000,
) -> ContextEngine:
    cfg = ContextConfig(
        max_context_tokens=max_tokens,
        response_reserve=response_reserve,
        tool_result_per_call_max_chars=per_call_max_chars,
    )
    return ContextEngine(cfg)


def _marker(*, tool_name: str, key: str, tokens: int = 4_300) -> str:
    """Build a marker exactly matching what ObservationMasker emits.

    ``prepare_for_synthesis`` parses the ``ref:`` line; the format must
    stay in lock-step with
    :mod:`nucleusiq.agents.context.strategies.observation_masker`.
    """
    return (
        "[observation consumed]\n"
        f"tool: {tool_name}\n"
        f"args: {{}}\n"
        f"ref: {key}\n"
        f"size: ~{tokens} tokens\n"
        f'To retrieve: call recall_tool_result(ref="{key}")'
    )


def _store_and_marker(engine: ContextEngine, key: str, content: str) -> ChatMessage:
    """Offload ``content`` and return a ``tool`` message holding the marker."""
    engine.store.store(key, content, original_tokens=len(content) // 4)
    return ChatMessage(
        role="tool",
        name="search",
        tool_call_id=key,
        content=_marker(tool_name="search", key=key),
    )


# ---------------------------------------------------------------------- #
# 1. Happy path                                                            #
# ---------------------------------------------------------------------- #


class TestRehydrationHappyPath:
    def test_rehydrates_single_marker_when_budget_is_ample(self):
        engine = _engine()
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="task"),
            _store_and_marker(engine, "ref-1", "FULL_EVIDENCE_BODY"),
            ChatMessage(role="user", content="please synthesise"),
        ]

        out = engine.prepare_for_synthesis(msgs)

        rehydrated = next(m for m in out if m.role == "tool")
        assert rehydrated.content == "FULL_EVIDENCE_BODY"
        # tool_call_id propagates so downstream LLMs can match the call.
        assert rehydrated.tool_call_id == "ref-1"

    def test_uses_resolved_window_when_config_window_is_auto(self):
        cfg = ContextConfig(
            max_context_tokens=None,
            response_reserve=1_000,
            tool_result_per_call_max_chars=50_000,
        )
        engine = ContextEngine(cfg, max_tokens=100_000)
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="task"),
            _store_and_marker(engine, "ref-1", "FULL_EVIDENCE_BODY"),
            ChatMessage(role="user", content="please synthesise"),
        ]

        out = engine.prepare_for_synthesis(msgs)

        rehydrated = next(m for m in out if m.role == "tool")
        assert rehydrated.content == "FULL_EVIDENCE_BODY"

    def test_does_not_mutate_input(self):
        """Pure — caller's list and ChatMessage objects survive untouched."""
        engine = _engine()
        original_marker = _marker(tool_name="search", key="ref-1")
        engine.store.store("ref-1", "FULL", original_tokens=1)
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(
                role="tool",
                name="search",
                tool_call_id="ref-1",
                content=original_marker,
            ),
        ]

        before_id = id(msgs)
        before_content = msgs[1].content

        out = engine.prepare_for_synthesis(msgs)

        assert id(out) != before_id, "Must return a new list"
        assert msgs[1].content == before_content, (
            "Original ChatMessage must not be mutated"
        )
        assert out[1].content == "FULL"

    def test_returns_input_unchanged_when_no_markers_present(self):
        engine = _engine()
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="task"),
            ChatMessage(role="assistant", content="answer"),
        ]
        out = engine.prepare_for_synthesis(msgs)
        assert [m.content for m in out] == [m.content for m in msgs]

    def test_empty_message_list_returns_empty(self):
        engine = _engine()
        assert engine.prepare_for_synthesis([]) == []


# ---------------------------------------------------------------------- #
# 2. Newest-first ordering                                                #
# ---------------------------------------------------------------------- #


class TestNewestFirst:
    def test_when_budget_fits_only_one_newest_wins(self):
        """A budget that fits exactly one marker rehydrates the newest."""
        # Tight window: each rehydration costs ~2k tokens; budget allows
        # ~one before the next would overflow.  ``per_call_max_chars``
        # caps each replacement so deltas are deterministic.
        big_chunk = "x" * 8_000  # ~2k tokens per DefaultTokenCounter
        engine = _engine(
            max_tokens=2_500,  # tiny window — only 1 rehydration fits
            response_reserve=200,
            per_call_max_chars=8_000,
        )

        old = _store_and_marker(engine, "ref-old", big_chunk)
        new = _store_and_marker(engine, "ref-new", big_chunk)
        msgs = [
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="t"),
            old,
            ChatMessage(role="assistant", content="a"),
            new,
            ChatMessage(role="user", content="please synthesise"),
        ]

        out = engine.prepare_for_synthesis(msgs)

        tools = [m for m in out if m.role == "tool"]
        # Older marker must stay as marker, newer rehydrated → preserves
        # the model's most recent evidence first.
        assert tools[0].content.startswith("[observation consumed"), (
            "Older marker must remain masked when budget is tight"
        )
        assert big_chunk in tools[1].content, (
            "Newer marker must be rehydrated first under tight budget"
        )

    def test_all_markers_rehydrated_when_budget_is_ample(self):
        engine = _engine(max_tokens=200_000, response_reserve=1_000)

        a = _store_and_marker(engine, "ref-a", "EVIDENCE_A")
        b = _store_and_marker(engine, "ref-b", "EVIDENCE_B")
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="task"),
            a,
            b,
            ChatMessage(role="user", content="synth"),
        ]

        out = engine.prepare_for_synthesis(msgs)

        tools = [m for m in out if m.role == "tool"]
        assert tools[0].content == "EVIDENCE_A"
        assert tools[1].content == "EVIDENCE_B"


# ---------------------------------------------------------------------- #
# 3. Budget gate                                                           #
# ---------------------------------------------------------------------- #


class TestBudgetGate:
    def test_zero_or_negative_budget_returns_unchanged(self):
        """When current usage already exceeds (max - reserve), no-op."""
        # The user message alone is huge — it'll dwarf max_context_tokens.
        # NB: ``max_context_tokens`` is a Pydantic-validated minimum, so we
        # use a small-but-legal window and a still-bigger message.
        engine = _engine(max_tokens=8_000, response_reserve=4_000)
        marker_msg = _store_and_marker(engine, "ref-1", "EVIDENCE_BODY")

        huge_user = ChatMessage(role="user", content="x" * 100_000)
        msgs = [
            ChatMessage(role="system", content="sys"),
            huge_user,
            marker_msg,
            ChatMessage(role="user", content="synth"),
        ]

        out = engine.prepare_for_synthesis(msgs)

        # Marker still a marker — budget didn't allow rehydration.
        tool_msg = next(m for m in out if m.role == "tool")
        assert tool_msg.content.startswith("[observation consumed")

    def test_partial_rehydration_when_budget_runs_out_mid_walk(self):
        """First (newest) fits, second would overflow → stop."""
        big = "y" * 8_000  # ~2k tokens
        engine = _engine(
            max_tokens=4_000,
            response_reserve=200,
            per_call_max_chars=8_000,
        )

        a = _store_and_marker(engine, "ref-a", big)
        b = _store_and_marker(engine, "ref-b", big)
        msgs = [
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="t"),
            a,
            b,
            ChatMessage(role="user", content="synth"),
        ]

        out = engine.prepare_for_synthesis(msgs)

        tools = [m for m in out if m.role == "tool"]
        # Newest rehydrated, older one stays as marker.
        assert big in tools[1].content
        assert tools[0].content.startswith("[observation consumed")


# ---------------------------------------------------------------------- #
# 4. Fail-open                                                             #
# ---------------------------------------------------------------------- #


class TestFailOpen:
    def test_missing_ref_in_store_leaves_marker_untouched(self):
        """Marker references a ref that was never stored → keep marker."""
        engine = _engine()
        ghost = ChatMessage(
            role="tool",
            name="search",
            tool_call_id="ref-ghost",
            content=_marker(tool_name="search", key="ref-ghost"),
        )
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="task"),
            ghost,
            ChatMessage(role="user", content="synth"),
        ]

        out = engine.prepare_for_synthesis(msgs)

        tool_msg = next(m for m in out if m.role == "tool")
        assert tool_msg.content == ghost.content

    def test_malformed_marker_without_ref_line_is_skipped(self):
        engine = _engine()
        bad = ChatMessage(
            role="tool",
            name="search",
            tool_call_id="x",
            content="[observation consumed]\nno ref line here",
        )
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="task"),
            bad,
            ChatMessage(role="user", content="synth"),
        ]

        out = engine.prepare_for_synthesis(msgs)

        tool_msg = next(m for m in out if m.role == "tool")
        assert tool_msg.content == bad.content

    def test_non_tool_role_is_never_rehydrated(self):
        """A user/assistant message that *looks* like a marker stays put."""
        engine = _engine()
        # Even if the marker prefix appears in a user message, we never
        # rehydrate non-tool roles — providers reject mismatched roles.
        engine.store.store("ref-1", "FULL", original_tokens=1)
        looks_like_marker = ChatMessage(
            role="user", content=_marker(tool_name="search", key="ref-1")
        )
        msgs = [
            ChatMessage(role="system", content="sys"),
            looks_like_marker,
            ChatMessage(role="user", content="synth"),
        ]

        out = engine.prepare_for_synthesis(msgs)

        # Untouched.
        assert out[1].content == looks_like_marker.content


# ---------------------------------------------------------------------- #
# 5. Idempotency                                                           #
# ---------------------------------------------------------------------- #


class TestIdempotency:
    def test_already_rehydrated_messages_are_left_alone(self):
        engine = _engine()
        marker_msg = _store_and_marker(engine, "ref-1", "BODY_AFTER_REHYDRATE")
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="task"),
            marker_msg,
            ChatMessage(role="user", content="synth"),
        ]

        first = engine.prepare_for_synthesis(msgs)
        second = engine.prepare_for_synthesis(first)

        assert [m.content for m in second] == [m.content for m in first]
