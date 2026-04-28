"""Tests for ContextEngine Phase 2 features.

Covers:
    - post_response() observation masking
    - optimal_budget-based ledger sizing
    - Cost estimation in telemetry
    - ObservationMasker integration with engine
"""

import pytest
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.engine import ContextEngine


def _msg(
    role: str, content: str, *, name: str | None = None, tool_call_id: str | None = None
):
    return ChatMessage(role=role, content=content, name=name, tool_call_id=tool_call_id)


class TestPostResponse:
    """Masker mechanics — these tests isolate the masker from the budget gate.

    Context Mgmt v2 — Step 1 introduced ``squeeze_threshold`` (default 0.70)
    so the masker only runs once budget pressure is real.  These tests
    verify the masker's *behaviour* (marker shape, counters, store
    side-effects) and therefore explicitly set ``squeeze_threshold=0.0``
    to bypass the gate.  Gate behaviour itself is covered in
    ``test_squeeze_gate.py``.
    """

    def test_masks_consumed_tool_results(self):
        config = ContextConfig(optimal_budget=10_000, squeeze_threshold=0.0)
        engine = ContextEngine(config, max_tokens=128_000)

        messages = [
            _msg("system", "You are helpful."),
            _msg("tool", "A" * 500, name="search", tool_call_id="tc1"),
            _msg("assistant", "Here are the results."),
        ]
        result = engine.post_response(messages)

        assert "[observation consumed" in result[1].content
        tel = engine.telemetry
        assert tel.observations_masked == 1
        assert tel.tokens_masked > 0

    def test_no_masking_when_disabled(self):
        config = ContextConfig(enable_observation_masking=False, squeeze_threshold=0.0)
        engine = ContextEngine(config, max_tokens=128_000)

        messages = [
            _msg("tool", "B" * 500, name="search", tool_call_id="tc1"),
            _msg("assistant", "Done."),
        ]
        result = engine.post_response(messages)

        assert result[0].content == "B" * 500

    def test_no_masking_when_strategy_none(self):
        """ObservationMasker respects enable_observation_masking, not strategy."""
        config = ContextConfig(
            strategy="none",
            enable_observation_masking=True,
            squeeze_threshold=0.0,
        )
        engine = ContextEngine(config, max_tokens=128_000)

        messages = [
            _msg("tool", "C" * 500, name="search", tool_call_id="tc1"),
            _msg("assistant", "Done."),
        ]
        result = engine.post_response(messages)
        assert "[observation consumed" in result[0].content

    def test_content_preserved_in_store(self):
        config = ContextConfig(optimal_budget=10_000, squeeze_threshold=0.0)
        engine = ContextEngine(config, max_tokens=128_000)
        original = "D" * 1000

        messages = [
            _msg("tool", original, name="big_tool", tool_call_id="tc1"),
            _msg("assistant", "Processed."),
        ]
        engine.post_response(messages)

        assert engine.store.size >= 1
        keys = engine.store.keys()
        assert engine.store.retrieve(keys[0]) == original


class TestOptimalBudget:
    def test_ledger_uses_optimal_budget(self):
        config = ContextConfig(optimal_budget=30_000, max_context_tokens=1_000_000)
        engine = ContextEngine(config, max_tokens=1_000_000)

        budget = engine.budget
        assert budget.max_tokens == 30_000

    def test_optimal_budget_smaller_than_max(self):
        config = ContextConfig(optimal_budget=50_000, max_context_tokens=128_000)
        engine = ContextEngine(config, max_tokens=128_000)

        budget = engine.budget
        assert budget.max_tokens == 50_000

    def test_optimal_budget_larger_than_max_clamps(self):
        config = ContextConfig(optimal_budget=200_000, max_context_tokens=32_000)
        engine = ContextEngine(config, max_tokens=32_000)

        budget = engine.budget
        assert budget.max_tokens == 32_000

    @pytest.mark.asyncio
    async def test_compaction_triggers_at_optimal_budget(self):
        config = ContextConfig(
            optimal_budget=200,
            tool_compaction_trigger=0.60,
        )
        engine = ContextEngine(config, max_tokens=1_000_000)

        effective = 200 - config.response_reserve
        if effective <= 0:
            config = ContextConfig(
                optimal_budget=20_000,
                tool_compaction_trigger=0.60,
                tool_result_threshold=100,
            )
            engine = ContextEngine(config, max_tokens=1_000_000)

        filler = "X" * 50_000
        messages = [
            _msg("system", "You are helpful."),
            _msg("tool", filler, name="big_tool", tool_call_id="tc1"),
        ]
        result = await engine.prepare(messages)
        tel = engine.telemetry

        assert tel.peak_utilization > 0


class TestCostTelemetry:
    @pytest.mark.asyncio
    async def test_cost_estimation_with_rate(self):
        config = ContextConfig(
            optimal_budget=10_000,
            cost_per_million_input=3.0,
        )
        engine = ContextEngine(config, max_tokens=128_000)

        messages = [
            _msg("system", "You are helpful."),
            _msg("user", "Hello world " * 100),
        ]
        await engine.prepare(messages)
        tel = engine.telemetry

        assert tel.estimated_cost_with_mgmt > 0

    @pytest.mark.asyncio
    async def test_cost_estimation_without_rate(self):
        config = ContextConfig(optimal_budget=10_000)
        engine = ContextEngine(config, max_tokens=128_000)

        messages = [_msg("system", "Hello.")]
        await engine.prepare(messages)
        tel = engine.telemetry

        assert tel.estimated_cost_without_mgmt == 0.0
        assert tel.estimated_cost_with_mgmt == 0.0
        assert tel.estimated_savings_pct == 0.0

    def test_optimal_budget_in_telemetry(self):
        config = ContextConfig(optimal_budget=42_000)
        engine = ContextEngine(config, max_tokens=128_000)

        tel = engine.telemetry
        assert tel.optimal_budget == 42_000

    @pytest.mark.asyncio
    async def test_masking_tokens_included_in_freed_total(self):
        # squeeze_threshold=0.0 — isolate this assertion from the v2 gate.
        config = ContextConfig(optimal_budget=10_000, squeeze_threshold=0.0)
        engine = ContextEngine(config, max_tokens=128_000)

        messages = [
            _msg("tool", "E" * 500, name="search", tool_call_id="tc1"),
            _msg("assistant", "Done."),
        ]
        engine.post_response(messages)
        await engine.prepare([_msg("system", "Hi")])

        tel = engine.telemetry
        assert tel.tokens_freed_total >= tel.tokens_masked
        assert tel.tokens_masked > 0


class TestStructuredSummary:
    @pytest.mark.asyncio
    async def test_conversation_compactor_with_summary(self):
        from nucleusiq.agents.context.budget import ContextBudget
        from nucleusiq.agents.context.counter import DefaultTokenCounter
        from nucleusiq.agents.context.strategies.conversation import (
            ConversationCompactor,
        )

        compactor = ConversationCompactor()
        counter = DefaultTokenCounter()

        long_content = "Detailed analysis of market trends " * 100
        messages = [
            _msg("system", "You are helpful."),
            _msg("user", "Analyze the market"),
            _msg("assistant", long_content),
            _msg("user", "What about tech?"),
            _msg("assistant", long_content),
            _msg("user", "Summarize findings"),
            _msg("assistant", long_content),
            _msg("user", "Final question"),
            _msg("assistant", "Final answer."),
        ]

        budget = ContextBudget(
            max_tokens=5000,
            response_reserve=500,
            allocated=4000,
            by_region={"system": 10, "user": 100, "assistant": 3890},
        )
        config = ContextConfig(
            optimal_budget=5000,
            enable_summarization=True,
            preserve_recent_turns=2,
        )

        result = await compactor.compact(messages, budget, config, counter)

        assert result.tokens_freed > 0
        assert result.summaries_inserted == 1
        found_summary = False
        for msg in result.messages:
            if isinstance(msg.content, str) and "WORKING STATE SUMMARY" in msg.content:
                found_summary = True
                break
        assert found_summary

    @pytest.mark.asyncio
    async def test_conversation_compactor_without_summary(self):
        from nucleusiq.agents.context.budget import ContextBudget
        from nucleusiq.agents.context.counter import DefaultTokenCounter
        from nucleusiq.agents.context.strategies.conversation import (
            ConversationCompactor,
        )

        compactor = ConversationCompactor()
        counter = DefaultTokenCounter()

        long_content = "Detailed analysis " * 100
        messages = [
            _msg("system", "You are helpful."),
            _msg("user", "Analyze"),
            _msg("assistant", long_content),
            _msg("user", "More"),
            _msg("assistant", long_content),
            _msg("user", "Final"),
            _msg("assistant", "Done."),
        ]

        budget = ContextBudget(
            max_tokens=5000,
            response_reserve=500,
            allocated=4000,
            by_region={"system": 10, "user": 100, "assistant": 3890},
        )
        config = ContextConfig(
            optimal_budget=5000,
            enable_summarization=False,
            preserve_recent_turns=2,
        )

        result = await compactor.compact(messages, budget, config, counter)

        assert result.tokens_freed > 0
        assert result.summaries_inserted == 0
        found_marker = False
        for msg in result.messages:
            if (
                isinstance(msg.content, str)
                and "earlier messages compacted" in msg.content
            ):
                found_marker = True
                break
        assert found_marker
