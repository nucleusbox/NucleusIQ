"""Comprehensive coverage tests for the context management module.

Targets every uncovered line across:
  - telemetry.py: merge(), cost estimation
  - conversation.py: summarization, partition, small-list edge cases
  - emergency.py: full truncation path, nothing-to-evict
  - tool_result.py: truncation path (no offloading), offload path
  - counter.py: list-content messages, tool_calls counting
  - engine.py: checkpoint, strategy=none, cost estimation, post_response no masker
  - observation_masker.py: small-token skip
  - budget.py: region breakdown, utilization
  - store.py: retrieve, contains, keys
"""

import pytest
from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.context import (
    ContextConfig,
    ContextEngine,
    DefaultTokenCounter,
)
from nucleusiq.agents.context.budget import ContextBudget, ContextLedger, Region
from nucleusiq.agents.context.store import ContentStore
from nucleusiq.agents.context.strategies.conversation import ConversationCompactor
from nucleusiq.agents.context.strategies.emergency import EmergencyCompactor
from nucleusiq.agents.context.strategies.observation_masker import ObservationMasker
from nucleusiq.agents.context.strategies.tool_result import ToolResultCompactor
from nucleusiq.agents.context.telemetry import ContextTelemetry

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_counter = DefaultTokenCounter()


def _budget(allocated=1000, max_tokens=2000, reserve=200):
    return ContextBudget(
        allocated=allocated,
        max_tokens=max_tokens,
        response_reserve=reserve,
        by_region={},
    )


# ------------------------------------------------------------------ #
# 1. ContextTelemetry — merge & cost fields                           #
# ------------------------------------------------------------------ #


class TestTelemetryMerge:
    def test_merge_parent_only_no_children(self):
        parent = ContextTelemetry(peak_utilization=0.5, compaction_count=2)
        result = ContextTelemetry.merge(parent, [])
        assert result is parent

    def test_merge_no_parent_no_children(self):
        result = ContextTelemetry.merge(None, [])
        assert result.peak_utilization == 0.0
        assert result.compaction_count == 0

    def test_merge_parent_with_children(self):
        parent = ContextTelemetry(
            peak_utilization=0.3,
            compaction_count=1,
            tokens_freed_total=100,
            artifacts_offloaded=2,
            observations_masked=5,
            tokens_masked=500,
            estimated_cost_without_mgmt=0.10,
            estimated_cost_with_mgmt=0.08,
            region_breakdown={"system": 50, "user": 100},
            warnings=("w1",),
        )
        child1 = ContextTelemetry(
            peak_utilization=0.7,
            compaction_count=2,
            tokens_freed_total=200,
            artifacts_offloaded=3,
            observations_masked=10,
            tokens_masked=1000,
            estimated_cost_without_mgmt=0.05,
            estimated_cost_with_mgmt=0.03,
            region_breakdown={"user": 50, "tool_result": 300},
            warnings=("w2",),
        )
        child2 = ContextTelemetry(
            peak_utilization=0.5,
            compaction_count=0,
            tokens_freed_total=0,
            estimated_cost_without_mgmt=0.0,
            estimated_cost_with_mgmt=0.0,
        )

        merged = ContextTelemetry.merge(parent, [child1, child2])

        assert merged.peak_utilization == 0.7
        assert merged.compaction_count == 3
        assert merged.tokens_freed_total == 300
        assert merged.artifacts_offloaded == 5
        assert merged.observations_masked == 15
        assert merged.tokens_masked == 1500
        assert merged.estimated_cost_without_mgmt == pytest.approx(0.15, abs=0.001)
        assert merged.estimated_cost_with_mgmt == pytest.approx(0.11, abs=0.001)
        assert merged.estimated_savings_pct > 0
        assert merged.region_breakdown["user"] == 150
        assert merged.region_breakdown["tool_result"] == 300
        assert "w1" in merged.warnings
        assert "w2" in merged.warnings

    def test_merge_none_parent_with_children(self):
        child = ContextTelemetry(
            peak_utilization=0.9,
            compaction_count=5,
            estimated_cost_without_mgmt=1.0,
            estimated_cost_with_mgmt=0.5,
        )
        merged = ContextTelemetry.merge(None, [child])
        assert merged.peak_utilization == 0.9
        assert merged.compaction_count == 5
        assert merged.estimated_savings_pct == pytest.approx(50.0)


# ------------------------------------------------------------------ #
# 2. ConversationCompactor — full coverage                             #
# ------------------------------------------------------------------ #


class TestConversationCompactor:
    @pytest.mark.asyncio
    async def test_tiny_message_list_skipped(self):
        cc = ConversationCompactor()
        msgs = [ChatMessage(role="system", content="sys")]
        result = await cc.compact(msgs, _budget(), ContextConfig(), _counter, None)
        assert result.tokens_freed == 0
        assert len(result.messages) == 1

    @pytest.mark.asyncio
    async def test_no_evictable_messages(self):
        cc = ConversationCompactor()
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
        result = await cc.compact(
            msgs, _budget(), ContextConfig(preserve_recent_turns=10), _counter, None
        )
        assert result.tokens_freed == 0

    @pytest.mark.asyncio
    async def test_truncation_mode_evicts_old_turns(self):
        cc = ConversationCompactor()
        filler = "word " * 100
        msgs = [ChatMessage(role="system", content="sys")]
        for i in range(8):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        config = ContextConfig(preserve_recent_turns=2, enable_summarization=False)
        result = await cc.compact(msgs, _budget(), config, _counter, None)

        assert result.tokens_freed > 0
        assert result.entries_removed > 0
        assert any("compacted" in m.content for m in result.messages if m.role == "system")

    @pytest.mark.asyncio
    async def test_summarization_mode_builds_working_state(self):
        cc = ConversationCompactor()
        filler = "word " * 100
        msgs = [ChatMessage(role="system", content="sys")]
        for _i in range(8):
            msgs.append(ChatMessage(role="user", content=f"Analyze {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"Result: {filler}"))

        config = ContextConfig(preserve_recent_turns=2, enable_summarization=True)
        result = await cc.compact(msgs, _budget(), config, _counter, None)

        assert result.tokens_freed > 0
        assert result.summaries_inserted == 1
        summary_msgs = [
            m for m in result.messages
            if m.role == "system" and "WORKING STATE SUMMARY" in (m.content or "")
        ]
        assert len(summary_msgs) == 1
        assert "Goals:" in summary_msgs[0].content
        assert "Decisions:" in summary_msgs[0].content

    @pytest.mark.asyncio
    async def test_partition_respects_tool_call_groups(self):
        cc = ConversationCompactor()
        tc = ToolCallRequest(id="tc1", name="search", arguments="{}")
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="search for X"),
            ChatMessage(role="assistant", content=None, tool_calls=[tc]),
            ChatMessage(role="tool", name="search", tool_call_id="tc1", content="result"),
            ChatMessage(role="user", content="follow up"),
            ChatMessage(role="assistant", content="final answer"),
        ]
        head, evictable, tail = cc._partition(msgs, preserve_recent=2)
        assert head[0].role == "system"
        for m in tail:
            assert m.role != "tool" or any(
                t.role == "assistant" and t.tool_calls for t in tail
            )

    @pytest.mark.asyncio
    async def test_structured_summary_with_tool_findings(self):
        evicted = [
            ChatMessage(role="user", content="What is the revenue?"),
            ChatMessage(role="assistant", content="Let me check the data."),
            ChatMessage(role="tool", name="db_query", content="Revenue: $10M\nGrowth: 15%"),
            ChatMessage(role="assistant", content="Revenue is $10M with 15% growth."),
        ]
        summary = ConversationCompactor._build_structured_summary(evicted, 500)
        assert "WORKING STATE SUMMARY" in summary
        assert "Goals:" in summary
        assert "Decisions:" in summary
        assert "Tool findings:" in summary
        assert "db_query" in summary


# ------------------------------------------------------------------ #
# 3. EmergencyCompactor — full coverage                                #
# ------------------------------------------------------------------ #


class TestEmergencyCompactor:
    @pytest.mark.asyncio
    async def test_nothing_to_evict(self):
        ec = EmergencyCompactor()
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hi"),
        ]
        result = await ec.compact(msgs, _budget(), ContextConfig(), _counter, None)
        assert result.tokens_freed == 0
        assert "nothing to evict" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_emergency_drops_all_but_last_group(self):
        ec = EmergencyCompactor()
        filler = "data " * 200
        tc = ToolCallRequest(id="tc1", name="tool", arguments="{}")
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"task {filler}"),
            ChatMessage(role="assistant", content=f"thinking {filler}"),
            ChatMessage(role="user", content=f"more {filler}"),
            ChatMessage(role="assistant", content=None, tool_calls=[tc]),
            ChatMessage(role="tool", name="tool", tool_call_id="tc1", content=f"result {filler}"),
        ]
        budget = _budget(allocated=5000, max_tokens=5000, reserve=500)
        result = await ec.compact(msgs, budget, ContextConfig(), _counter, None)

        assert result.tokens_freed > 0
        assert result.entries_removed > 0
        assert any("CONTEXT COMPACTED" in m.content for m in result.messages if m.role == "system")
        assert any("Emergency compaction" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_emergency_keeps_system_and_last_group(self):
        ec = EmergencyCompactor()
        msgs = [
            ChatMessage(role="system", content="sys prompt"),
            ChatMessage(role="user", content="q1"),
            ChatMessage(role="assistant", content="a1"),
            ChatMessage(role="user", content="q2"),
            ChatMessage(role="assistant", content="a2"),
        ]
        result = await ec.compact(msgs, _budget(), ContextConfig(), _counter, None)
        roles = [m.role for m in result.messages]
        assert roles[0] == "system"
        assert "system" in roles[1]


# ------------------------------------------------------------------ #
# 4. ToolResultCompactor — truncation & offloading                     #
# ------------------------------------------------------------------ #


class TestToolResultCompactor:
    @pytest.mark.asyncio
    async def test_truncation_mode_no_offloading(self):
        trc = ToolResultCompactor()
        long_content = "\n".join(f"line {i}: {'x' * 80}" for i in range(30))
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="tool", name="search", content=long_content),
        ]
        config = ContextConfig(
            tool_result_threshold=20,
            enable_offloading=False,
        )
        result = await trc.compact(msgs, _budget(), config, _counter, None)

        assert result.tokens_freed > 0
        assert result.artifacts_offloaded == 0
        tool_msg = [m for m in result.messages if m.role == "tool"][0]
        assert "truncated" in tool_msg.content.lower()

    @pytest.mark.asyncio
    async def test_offloading_mode(self):
        trc = ToolResultCompactor()
        store = ContentStore()
        long_content = "\n".join(f"line {i}: {'x' * 80}" for i in range(30))
        msgs = [
            ChatMessage(role="tool", name="api", content=long_content),
        ]
        config = ContextConfig(
            tool_result_threshold=20,
            enable_offloading=True,
        )
        result = await trc.compact(msgs, _budget(), config, _counter, store)

        assert result.tokens_freed > 0
        assert result.artifacts_offloaded == 1
        assert store.size == 1

    @pytest.mark.asyncio
    async def test_small_result_passes_through(self):
        trc = ToolResultCompactor()
        msgs = [
            ChatMessage(role="tool", name="calc", content='{"result": 42}'),
        ]
        config = ContextConfig(tool_result_threshold=1000)
        result = await trc.compact(msgs, _budget(), config, _counter, None)

        assert result.tokens_freed == 0
        assert result.messages[0].content == '{"result": 42}'

    @pytest.mark.asyncio
    async def test_truncation_few_lines_no_truncation(self):
        trc = ToolResultCompactor()
        short_content = "line1\nline2\nline3"
        msgs = [
            ChatMessage(role="tool", name="t", content=short_content),
        ]
        config = ContextConfig(tool_result_threshold=1, enable_offloading=False)
        result = await trc.compact(msgs, _budget(), config, _counter, None)
        assert result.tokens_freed == 0

    @pytest.mark.asyncio
    async def test_non_tool_messages_pass_through(self):
        trc = ToolResultCompactor()
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi there"),
        ]
        result = await trc.compact(msgs, _budget(), ContextConfig(), _counter, None)
        assert len(result.messages) == 3
        assert result.tokens_freed == 0


# ------------------------------------------------------------------ #
# 5. DefaultTokenCounter — list content & tool_calls                   #
# ------------------------------------------------------------------ #


class TestDefaultTokenCounter:
    def test_count_basic(self):
        c = DefaultTokenCounter()
        assert c.count("hello world") > 0
        assert c.count("") == 1

    def test_count_messages_with_list_content(self):
        c = DefaultTokenCounter()
        msg = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "http://..."}},
            ],
        )
        tokens = c.count_messages([msg])
        assert tokens > 4

    def test_count_messages_with_tool_calls(self):
        c = DefaultTokenCounter()
        tc = ToolCallRequest(id="tc1", name="search", arguments='{"q": "test"}')
        msg = ChatMessage(role="assistant", content="Let me search.", tool_calls=[tc])
        tokens = c.count_messages([msg])
        assert tokens > 4

    def test_count_messages_with_name(self):
        c = DefaultTokenCounter()
        msg = ChatMessage(role="tool", name="calculator", content="42")
        tokens = c.count_messages([msg])
        assert tokens > 4


# ------------------------------------------------------------------ #
# 6. ContextEngine — checkpoint, strategy=none, cost estimation        #
# ------------------------------------------------------------------ #


class TestContextEngineExtended:
    def test_strategy_none_skips_everything(self):
        engine = ContextEngine(
            config=ContextConfig(strategy="none"),
            token_counter=DefaultTokenCounter(),
            max_tokens=100_000,
        )
        large = "x" * 10000
        result = engine.ingest_tool_result(large, "big_tool")
        assert result == large

    @pytest.mark.asyncio
    async def test_prepare_strategy_none_passthrough(self):
        engine = ContextEngine(
            config=ContextConfig(strategy="none"),
            token_counter=DefaultTokenCounter(),
            max_tokens=100_000,
        )
        msgs = [ChatMessage(role="user", content="hi")]
        prepared = await engine.prepare(msgs)
        assert prepared is msgs

    def test_post_response_no_masker(self):
        engine = ContextEngine(
            config=ContextConfig(
                strategy="progressive",
                enable_observation_masking=False,
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=100_000,
        )
        msgs = [
            ChatMessage(role="assistant", content="ok"),
            ChatMessage(role="tool", name="t", content="data " * 100),
        ]
        result = engine.post_response(msgs)
        assert all(
            not m.content.startswith("[observation consumed")
            for m in result if m.role == "tool"
        )

    def test_checkpoint_recorded(self):
        engine = ContextEngine(
            config=ContextConfig(max_context_tokens=10_000),
            token_counter=DefaultTokenCounter(),
            max_tokens=10_000,
        )
        engine.checkpoint("task_boundary_1")
        assert "task_boundary_1" in engine._checkpoints

    def test_telemetry_cost_estimation(self):
        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=50_000,
                cost_per_million_input=3.0,
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=50_000,
        )
        engine._total_tokens_sent = 100_000
        engine._tokens_masked = 20_000

        tel = engine.telemetry
        assert tel.estimated_cost_without_mgmt > 0
        assert tel.estimated_cost_with_mgmt > 0
        assert tel.estimated_savings_pct > 0
        assert tel.estimated_cost_without_mgmt > tel.estimated_cost_with_mgmt

    def test_budget_property(self):
        engine = ContextEngine(
            config=ContextConfig(max_context_tokens=10_000),
            token_counter=DefaultTokenCounter(),
            max_tokens=10_000,
        )
        b = engine.budget
        assert isinstance(b, ContextBudget)
        assert b.max_tokens == 10_000

    def test_store_property(self):
        engine = ContextEngine(
            config=ContextConfig(max_context_tokens=10_000),
            token_counter=DefaultTokenCounter(),
            max_tokens=10_000,
        )
        assert isinstance(engine.store, ContentStore)


# ------------------------------------------------------------------ #
# 7. ObservationMasker — small-token skip path                        #
# ------------------------------------------------------------------ #


class TestObservationMaskerEdgeCases:
    def test_small_tool_result_not_masked(self):
        masker = ObservationMasker()
        store = ContentStore()
        msgs = [
            ChatMessage(role="tool", name="t", tool_call_id="tc1", content="ok"),
            ChatMessage(role="assistant", content="Got it."),
        ]
        result, count, freed = masker.mask(msgs, _counter, store)
        assert count == 0

    def test_already_masked_skipped(self):
        masker = ObservationMasker()
        store = ContentStore()
        msgs = [
            ChatMessage(role="tool", name="t", content="[observation consumed — 500 tokens]"),
            ChatMessage(role="assistant", content="ok"),
        ]
        result, count, freed = masker.mask(msgs, _counter, store)
        assert count == 0

    def test_no_assistant_message_no_masking(self):
        masker = ObservationMasker()
        store = ContentStore()
        msgs = [
            ChatMessage(role="tool", name="t", content="big data " * 100),
            ChatMessage(role="user", content="analyze"),
        ]
        result, count, freed = masker.mask(msgs, _counter, store)
        assert count == 0

    def test_context_ref_skipped(self):
        masker = ObservationMasker()
        store = ContentStore()
        msgs = [
            ChatMessage(role="tool", name="t", content="[context_ref: some_key]"),
            ChatMessage(role="assistant", content="ok"),
        ]
        result, count, freed = masker.mask(msgs, _counter, store)
        assert count == 0


# ------------------------------------------------------------------ #
# 8. ContentStore — retrieve, contains, keys                           #
# ------------------------------------------------------------------ #


class TestContentStore:
    def test_store_and_retrieve(self):
        store = ContentStore()
        store.store(key="k1", content="data", original_tokens=100)
        assert store.contains("k1")
        assert store.retrieve("k1") == "data"
        keys = store.keys()
        assert "k1" in keys
        assert store.size == 1

    def test_retrieve_missing_key(self):
        store = ContentStore()
        assert store.retrieve("nonexistent") is None

    def test_contains_false(self):
        store = ContentStore()
        assert not store.contains("missing")


# ------------------------------------------------------------------ #
# 9. ContextLedger / ContextBudget — region breakdown                  #
# ------------------------------------------------------------------ #


class TestContextLedger:
    def test_allocate_and_snapshot(self):
        ledger = ContextLedger(max_tokens=1000, response_reserve=100)
        ledger.allocate(
            msg_id="m1", tokens=50, region=Region.SYSTEM,
            source_type="system", importance=1.0,
        )
        ledger.allocate(
            msg_id="m2", tokens=200, region=Region.USER,
            source_type="user", importance=0.5,
        )
        snapshot = ledger.snapshot()
        assert snapshot.allocated == 250
        assert snapshot.by_region.get("system", 0) == 50
        assert snapshot.by_region.get("user", 0) == 200
        assert snapshot.utilization > 0

    def test_reset(self):
        ledger = ContextLedger(max_tokens=1000, response_reserve=100)
        ledger.allocate(
            msg_id="m1", tokens=500, region=Region.SYSTEM,
            source_type="system", importance=1.0,
        )
        ledger.reset()
        snapshot = ledger.snapshot()
        assert snapshot.allocated == 0

    def test_deallocate(self):
        ledger = ContextLedger(max_tokens=1000, response_reserve=100)
        ledger.allocate(msg_id="m1", tokens=200, region=Region.USER,
                        source_type="user", importance=0.5)
        freed = ledger.deallocate("m1")
        assert freed == 200
        assert ledger.total_allocated == 0

    def test_deallocate_nonexistent(self):
        ledger = ContextLedger(max_tokens=1000, response_reserve=100)
        freed = ledger.deallocate("missing")
        assert freed == 0

    def test_reallocate_replaces_existing(self):
        ledger = ContextLedger(max_tokens=1000, response_reserve=100)
        ledger.allocate(msg_id="m1", tokens=100, region=Region.USER,
                        source_type="user")
        ledger.allocate(msg_id="m1", tokens=50, region=Region.USER,
                        source_type="user")
        assert ledger.total_allocated == 50

    def test_get_entry(self):
        ledger = ContextLedger(max_tokens=1000, response_reserve=100)
        ledger.allocate(msg_id="m1", tokens=100, region=Region.SYSTEM,
                        source_type="system", importance=1.0)
        entry = ledger.get_entry("m1")
        assert entry is not None
        assert entry.tokens == 100
        assert ledger.get_entry("nonexistent") is None

    def test_entries_by_region(self):
        ledger = ContextLedger(max_tokens=1000, response_reserve=100)
        ledger.allocate(msg_id="m1", tokens=100, region=Region.SYSTEM,
                        source_type="system")
        ledger.allocate(msg_id="m2", tokens=200, region=Region.USER,
                        source_type="user")
        ledger.allocate(msg_id="m3", tokens=300, region=Region.USER,
                        source_type="user")
        user_entries = ledger.entries_by_region(Region.USER)
        assert len(user_entries) == 2

    def test_entry_count(self):
        ledger = ContextLedger(max_tokens=1000, response_reserve=100)
        assert ledger.entry_count == 0
        ledger.allocate(msg_id="m1", tokens=100, region=Region.SYSTEM,
                        source_type="system")
        assert ledger.entry_count == 1


# ------------------------------------------------------------------ #
# 10. ContextBudget — edge cases                                       #
# ------------------------------------------------------------------ #


class TestContextBudgetEdgeCases:
    def test_utilization_when_effective_limit_zero(self):
        budget = ContextBudget(
            max_tokens=100, response_reserve=100, allocated=50, by_region={},
        )
        assert budget.utilization == 1.0

    def test_effective_limit(self):
        budget = ContextBudget(
            max_tokens=1000, response_reserve=200, allocated=0, by_region={},
        )
        assert budget.effective_limit == 800

    def test_available(self):
        budget = ContextBudget(
            max_tokens=1000, response_reserve=200, allocated=500, by_region={},
        )
        assert budget.available == 300

    def test_can_fit(self):
        budget = ContextBudget(
            max_tokens=1000, response_reserve=200, allocated=500, by_region={},
        )
        assert budget.can_fit(200) is True
        assert budget.can_fit(500) is False


# ------------------------------------------------------------------ #
# 11. ContentStore — preview, remove, clear                            #
# ------------------------------------------------------------------ #


class TestContentStoreExtended:
    def test_preview(self):
        store = ContentStore()
        store.store(key="k1", content="line1\nline2\nline3", original_tokens=50)
        assert store.preview("k1") is not None
        assert "line1" in store.preview("k1")
        assert store.preview("missing") is None

    def test_remove(self):
        store = ContentStore()
        store.store(key="k1", content="data", original_tokens=10)
        assert store.remove("k1") is True
        assert store.size == 0
        assert store.remove("k1") is False

    def test_clear(self):
        store = ContentStore()
        store.store(key="k1", content="a", original_tokens=5)
        store.store(key="k2", content="b", original_tokens=5)
        assert store.size == 2
        store.clear()
        assert store.size == 0


# ------------------------------------------------------------------ #
# 12. CompactionPipeline — tier_count, zero effective_limit            #
# ------------------------------------------------------------------ #


class TestCompactionPipelineEdgeCases:
    def test_tier_count(self):
        from nucleusiq.agents.context.pipeline import CompactionPipeline
        p = CompactionPipeline([
            (0.7, ToolResultCompactor()),
            (0.95, EmergencyCompactor()),
        ])
        assert p.tier_count == 2

    @pytest.mark.asyncio
    async def test_zero_effective_limit_sets_util_to_one(self):
        from nucleusiq.agents.context.pipeline import CompactionPipeline
        budget = ContextBudget(
            max_tokens=100, response_reserve=100, allocated=50, by_region={},
        )
        config = ContextConfig(tool_compaction_trigger=0.0)
        pipeline = CompactionPipeline([
            (0.0, ToolResultCompactor()),
        ])
        msgs = [ChatMessage(role="tool", name="t", content="x")]
        result_msgs, events = await pipeline.run(
            msgs, budget, config, _counter, None,
        )
        assert len(events) > 0


# ------------------------------------------------------------------ #
# 13. ContextEngine — truncate_only strategy, list-content tokens      #
# ------------------------------------------------------------------ #


class TestContextEngineTruncateOnly:
    @pytest.mark.asyncio
    async def test_truncate_only_strategy_builds_correct_pipeline(self):
        engine = ContextEngine(
            config=ContextConfig(strategy="truncate_only", max_context_tokens=5000),
            token_counter=DefaultTokenCounter(),
            max_tokens=5000,
        )
        assert engine._pipeline.tier_count == 2

    @pytest.mark.asyncio
    async def test_prepare_recount_with_list_content(self):
        engine = ContextEngine(
            config=ContextConfig(max_context_tokens=100_000),
            token_counter=DefaultTokenCounter(),
            max_tokens=100_000,
        )
        msgs = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url"},
                ],
            ),
        ]
        result = await engine.prepare(msgs)
        assert result is msgs
        budget = engine.budget
        assert budget.allocated > 0


# ------------------------------------------------------------------ #
# 14. ConversationCompactor — edge cases                               #
# ------------------------------------------------------------------ #


class TestConversationCompactorEdgeCases:
    @pytest.mark.asyncio
    async def test_message_with_non_string_content(self):
        cc = ConversationCompactor()
        filler = "word " * 100
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=None),
            ChatMessage(role="assistant", content=filler),
            ChatMessage(role="user", content=filler),
            ChatMessage(role="assistant", content=filler),
            ChatMessage(role="user", content="last q"),
            ChatMessage(role="assistant", content="last a"),
        ]
        config = ContextConfig(preserve_recent_turns=1, enable_summarization=True)
        result = await cc.compact(msgs, _budget(), config, _counter, None)
        assert result.tokens_freed > 0

    @pytest.mark.asyncio
    async def test_only_system_messages(self):
        cc = ConversationCompactor()
        msgs = [
            ChatMessage(role="system", content="a"),
            ChatMessage(role="system", content="b"),
        ]
        result = await cc.compact(msgs, _budget(), ContextConfig(), _counter, None)
        assert result.tokens_freed == 0

    def test_partition_all_system_messages(self):
        cc = ConversationCompactor()
        msgs = [
            ChatMessage(role="system", content="a"),
            ChatMessage(role="system", content="b"),
            ChatMessage(role="system", content="c"),
        ]
        head, evictable, tail = cc._partition(msgs, preserve_recent=2)
        assert len(head) == 3
        assert evictable == []
        assert tail == []


# ------------------------------------------------------------------ #
# 15. Engine warning extraction from compaction events                 #
# ------------------------------------------------------------------ #


class TestEngineWarningPropagation:
    @pytest.mark.asyncio
    async def test_warnings_extracted_from_events(self):
        """Force the warning extraction path in engine.prepare() by using
        emergency compaction which produces warnings."""
        filler = "x " * 500
        msgs = [
            ChatMessage(role="system", content="system prompt"),
        ]
        for i in range(20):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=600,
                optimal_budget=600,
                response_reserve=50,
                tool_compaction_trigger=0.3,
                compaction_trigger=0.5,
                emergency_trigger=0.7,
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=600,
        )
        await engine.prepare(msgs)
        tel = engine.telemetry
        assert tel.compaction_count > 0
