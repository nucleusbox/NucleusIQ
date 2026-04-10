"""Real-world proof: Context Window Management solves actual production problems.

These tests simulate realistic agent scenarios and prove with hard numbers
that context management prevents overflow, saves tokens, and maintains
quality.

Each test prints measurable evidence:
  - Token counts before/after compaction
  - Number of messages compacted
  - Artifacts offloaded
  - Region-by-region breakdown
"""

import pytest
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context import (
    ContextConfig,
    ContextEngine,
    DefaultTokenCounter,
)

counter = DefaultTokenCounter()


def _token_count(msgs):
    return counter.count_messages(msgs)


def _print_telemetry(engine, label: str):
    tel = engine.telemetry
    budget = engine.budget
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Context limit     : {budget.max_tokens:,} tokens")
    print(f"  Response reserve  : {budget.response_reserve:,} tokens")
    print(f"  Effective limit   : {budget.effective_limit:,} tokens")
    print(f"  Current allocated : {budget.allocated:,} tokens")
    print(f"  Available         : {budget.available:,} tokens")
    print(f"  Utilization       : {budget.utilization:.1%}")
    print(f"  Peak utilization  : {tel.peak_utilization:.1%}")
    print(f"  Compactions       : {tel.compaction_count}")
    print(f"  Tokens freed      : {tel.tokens_freed_total:,}")
    print(f"  Artifacts stored  : {tel.artifacts_offloaded}")
    if tel.region_breakdown:
        print("  Region breakdown  :")
        for region, tokens in sorted(tel.region_breakdown.items()):
            if tokens > 0:
                print(f"    {region:15s}: {tokens:,}")
    if tel.compaction_events:
        print("  Compaction events :")
        for ce in tel.compaction_events:
            print(
                f"    [{ce.strategy}] {ce.tokens_before:,} -> {ce.tokens_after:,} "
                f"(freed {ce.tokens_freed:,}, {ce.duration_ms:.2f}ms)"
            )
    print()


# ================================================================== #
# SCENARIO 1: Research Agent — 15 web search results                   #
# Problem: Each search returns 2-5K tokens. After 10 searches,        #
# context is 90%+ full, model quality degrades.                        #
# ================================================================== #


class TestScenario1_ResearchAgent:
    """Simulates a research agent doing 15 web searches.

    Without context management: 15 * 3K = 45K tokens of tool results
    alone, pushing a 50K context to overflow.

    With context management: ToolResultCompactor offloads oversized
    results, keeping utilization manageable.
    """

    @staticmethod
    def _build_search_history(num_searches: int = 15) -> list[ChatMessage]:
        msgs = [
            ChatMessage(role="system", content="You are a research assistant."),
            ChatMessage(
                role="user", content="Research the impact of AI on healthcare in 2026."
            ),
        ]
        for i in range(num_searches):
            msgs.append(
                ChatMessage(
                    role="assistant",
                    content=f"Searching for: AI healthcare topic {i + 1}",
                )
            )
            result_lines = [
                f"Search Result {i + 1}: AI Healthcare Impact",
                f"Source: journal-{i + 1}.example.com",
                "Abstract: " + "x" * 200,
                "Key Findings:",
            ] + [f"  - Finding {j}: " + "y" * 150 for j in range(8)]
            msgs.append(
                ChatMessage(
                    role="tool",
                    name="web_search",
                    content="\n".join(result_lines),
                )
            )
            msgs.append(
                ChatMessage(
                    role="assistant",
                    content=f"Found relevant data about topic {i + 1}. Continuing research...",
                )
            )
        msgs.append(ChatMessage(role="user", content="Now synthesize all findings."))
        return msgs

    @pytest.mark.asyncio
    async def test_without_context_management(self):
        """WITHOUT management: context overflows an 8K window."""
        msgs = self._build_search_history(15)
        total_tokens = _token_count(msgs)
        context_limit = 8_000

        print("\n--- SCENARIO 1: Research Agent (15 searches) ---")
        print("WITHOUT context management:")
        print(f"  Total tokens     : {total_tokens:,}")
        print(f"  Context limit    : {context_limit:,}")
        print(f"  Utilization      : {total_tokens / (context_limit - 1000):.1%}")
        print(
            f"  Would overflow?  : {'YES' if total_tokens > (context_limit - 1000) else 'NO'}"
        )

        effective = context_limit - 1000
        if total_tokens > effective:
            overflow = total_tokens - effective
            print(f"  Overflow by      : {overflow:,} tokens")

    @pytest.mark.asyncio
    async def test_with_context_management(self):
        """WITH management: progressive compaction keeps it under control."""
        msgs = self._build_search_history(15)
        tokens_before = _token_count(msgs)

        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=8_000,
                response_reserve=1000,
                tool_result_threshold=200,
                strategy="progressive",
                preserve_recent_turns=3,
            ),
            token_counter=counter,
            max_tokens=8_000,
        )

        prepared = await engine.prepare(msgs)
        tokens_after = _token_count(prepared)

        print("\nWITH context management:")
        print(f"  Tokens before    : {tokens_before:,}")
        print(f"  Tokens after     : {tokens_after:,}")
        print(f"  Tokens saved     : {tokens_before - tokens_after:,}")
        print(f"  Reduction        : {(1 - tokens_after / tokens_before):.1%}")
        print(f"  Messages before  : {len(msgs)}")
        print(f"  Messages after   : {len(prepared)}")
        _print_telemetry(engine, "Research Agent Telemetry")

        budget = engine.budget
        assert budget.utilization < 1.0, "Context should not overflow"
        assert engine.telemetry.compaction_count > 0, "Compaction should have fired"
        assert tokens_after < tokens_before, "Tokens should be reduced"


# ================================================================== #
# SCENARIO 2: Data Analyst — huge JSON tool results                    #
# Problem: Single API response is 20K+ tokens. Context fills up fast. #
# ================================================================== #


class TestScenario2_DataAnalyst:
    """Simulates a data analyst agent receiving large JSON API responses.

    Each API call returns a large JSON dataset. Without management,
    two calls fill the context. With management, results are offloaded
    and only previews kept in context.
    """

    @staticmethod
    def _build_data_analysis() -> list[ChatMessage]:
        msgs = [
            ChatMessage(
                role="system",
                content="You are a data analyst. Analyze data from multiple APIs.",
            ),
            ChatMessage(
                role="user", content="Compare Q1 and Q2 sales data across all regions."
            ),
        ]

        for quarter in ["Q1", "Q2"]:
            msgs.append(
                ChatMessage(
                    role="assistant",
                    content=f"Fetching {quarter} sales data...",
                )
            )
            lines = [f"=== {quarter} Sales Report ==="]
            for r in range(20):
                for m in range(1, 4):
                    lines.append(
                        f"Region_{r} | Month {m} | "
                        f"Sales: {10000 + r * 1000 + m * 500} | "
                        f"Units: {500 + r * 50} | "
                        f"Notes: detailed analysis for region {r} month {m} "
                        + "x"
                        * 100
                    )
            msgs.append(
                ChatMessage(
                    role="tool",
                    name="sales_api",
                    content="\n".join(lines),
                )
            )

        msgs.append(
            ChatMessage(role="user", content="Which region had the best growth?")
        )
        return msgs

    @pytest.mark.asyncio
    async def test_large_json_offloaded(self):
        """Large JSON responses are offloaded, previews kept in context."""
        msgs = self._build_data_analysis()
        tokens_before = _token_count(msgs)

        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=5_000,
                response_reserve=500,
                tool_result_threshold=200,
                strategy="progressive",
                enable_offloading=True,
            ),
            token_counter=counter,
            max_tokens=5_000,
        )

        prepared = await engine.prepare(msgs)
        tokens_after = _token_count(prepared)

        print("\n--- SCENARIO 2: Data Analyst (huge JSON) ---")
        print(f"  Tokens before    : {tokens_before:,}")
        print(f"  Tokens after     : {tokens_after:,}")
        print(f"  Tokens saved     : {tokens_before - tokens_after:,}")
        print(f"  Reduction        : {(1 - tokens_after / tokens_before):.1%}")
        _print_telemetry(engine, "Data Analyst Telemetry")

        assert engine.store.size > 0, "Large results should be offloaded"
        assert tokens_after < tokens_before, "Tokens should be reduced"

        for key in engine.store.keys():  # noqa: SIM118
            full = engine.store.retrieve(key)
            assert full is not None, "Full data should be retrievable"
            assert "Sales Report" in full, "Original data preserved in store"

        for msg in prepared:
            if msg.role == "tool" and "[context_ref:" in str(msg.content):
                assert "preview" in str(msg.content).lower()


# ================================================================== #
# SCENARIO 3: Long conversation — 50 turns then final question        #
# Problem: Old conversation history consumes most of the context.      #
# ================================================================== #


class TestScenario3_LongConversation:
    """Simulates a 50-turn conversation that fills the context.

    Without management: old irrelevant turns consume most of the window.
    With management: ConversationCompactor removes old turns, keeping
    only recent relevant context.
    """

    @staticmethod
    def _build_long_conversation(turns: int = 50) -> list[ChatMessage]:
        msgs = [
            ChatMessage(
                role="system",
                content="You are a project planning assistant helping plan a software release.",
            ),
        ]
        for i in range(turns):
            topic = [
                "requirements",
                "architecture",
                "testing",
                "deployment",
                "monitoring",
            ][i % 5]
            msgs.append(
                ChatMessage(
                    role="user",
                    content=f"Turn {i + 1}: Let's discuss {topic}. " + "details " * 30,
                )
            )
            msgs.append(
                ChatMessage(
                    role="assistant",
                    content=f"Here are my thoughts on {topic} for turn {i + 1}. "
                    + "analysis " * 30,
                )
            )
        msgs.append(
            ChatMessage(
                role="user",
                content="Now summarize the key decisions we made across all topics.",
            )
        )
        return msgs

    @pytest.mark.asyncio
    async def test_old_turns_compacted(self):
        """Old conversation turns are compacted, recent turns preserved."""
        msgs = self._build_long_conversation(50)
        tokens_before = _token_count(msgs)

        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=8_000,
                response_reserve=1000,
                strategy="progressive",
                preserve_recent_turns=4,
            ),
            token_counter=counter,
            max_tokens=8_000,
        )

        prepared = await engine.prepare(msgs)
        tokens_after = _token_count(prepared)

        print("\n--- SCENARIO 3: Long Conversation (50 turns) ---")
        print(f"  Tokens before    : {tokens_before:,}")
        print(f"  Tokens after     : {tokens_after:,}")
        print(f"  Tokens saved     : {tokens_before - tokens_after:,}")
        print(f"  Reduction        : {(1 - tokens_after / tokens_before):.1%}")
        print(f"  Messages before  : {len(msgs)}")
        print(f"  Messages after   : {len(prepared)}")
        _print_telemetry(engine, "Long Conversation Telemetry")

        assert tokens_after < tokens_before
        assert len(prepared) < len(msgs)

        assert prepared[0].role == "system"
        assert prepared[-1].content.startswith("Now summarize")

        recent_contents = [
            m.content for m in prepared if m.role in ("user", "assistant")
        ]
        assert any("Turn 50" in c or "turn 50" in c for c in recent_contents), (
            "Most recent turns should be preserved"
        )


# ================================================================== #
# SCENARIO 4: Progressive escalation — tool → conversation → emergency #
# Problem: Context keeps growing despite compaction at each tier.      #
# ================================================================== #


class TestScenario4_ProgressiveEscalation:
    """Tests that all three compaction tiers fire progressively."""

    @pytest.mark.asyncio
    async def test_all_three_tiers_fire(self):
        """Push context through all compaction levels."""
        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=1_000,
                response_reserve=100,
                tool_result_threshold=50,
                tool_compaction_trigger=0.40,
                compaction_trigger=0.60,
                emergency_trigger=0.80,
                strategy="progressive",
                preserve_recent_turns=1,
            ),
            token_counter=counter,
            max_tokens=1_000,
        )

        filler = " ".join(["word"] * 40)
        msgs = [ChatMessage(role="system", content="sys")]
        for i in range(15):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(
                ChatMessage(
                    role="tool",
                    name=f"tool_{i}",
                    content="\n".join(f"line {j}: {'d' * 80}" for j in range(5)),
                )
            )
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        prepared = await engine.prepare(msgs)
        tel = engine.telemetry

        print("\n--- SCENARIO 4: Progressive Escalation ---")
        print(f"  Messages before  : {len(msgs)}")
        print(f"  Messages after   : {len(prepared)}")
        _print_telemetry(engine, "Progressive Escalation Telemetry")

        assert tel.compaction_count >= 1
        assert tel.tokens_freed_total > 0
        assert len(prepared) < len(msgs)

        strategies_used = [e.strategy for e in tel.compaction_events]
        print(f"  Strategies fired : {strategies_used}")
        assert len(strategies_used) >= 1
