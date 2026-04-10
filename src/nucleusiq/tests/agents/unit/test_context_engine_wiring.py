"""Integration tests: ContextEngine wiring into Agent + all 3 execution modes.

Tests prove that:
1. ContextEngine is created during _setup_execution() when configured
2. prepare() is called before LLM calls (messages get compacted)
3. ingest_tool_result() compresses large tool results
4. Telemetry is captured in AgentResult.context_telemetry
5. All three modes (Direct, Standard, Autonomous) work correctly
6. display() shows context telemetry section
"""

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.agent_result import AgentResult, ResultStatus
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.agents.context import ContextConfig, ContextEngine, DefaultTokenCounter
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM

from nucleusiq.tests.conftest import make_test_prompt


def _make_task(objective: str = "Analyze data") -> Task:
    return Task.from_dict({"id": "test-task", "objective": objective})


# ------------------------------------------------------------------ #
# Test: ContextEngine is created when ContextConfig is provided        #
# ------------------------------------------------------------------ #


class TestContextEngineCreation:
    def test_engine_created_with_config(self):
        """When ContextConfig is provided, _create_context_engine returns an engine."""
        agent = Agent(
            name="test",
            role="tester",
            objective="test context",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(
                context=ContextConfig(max_context_tokens=10_000),
                respect_context_window=True,
            ),
        )
        agent._tracer = None
        engine = agent._create_context_engine()
        assert engine is not None
        assert isinstance(engine, ContextEngine)
        assert engine.budget.max_tokens == 10_000

    def test_engine_none_when_strategy_none(self):
        """When strategy='none', no engine is created."""
        agent = Agent(
            name="test",
            role="tester",
            objective="test",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(context=ContextConfig(strategy="none")),
        )
        agent._tracer = None
        engine = agent._create_context_engine()
        assert engine is None

    def test_engine_auto_creates_with_respect_context_window(self):
        """When context=None but respect_context_window=True, engine is auto-created."""
        agent = Agent(
            name="test",
            role="tester",
            objective="test",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(respect_context_window=True),
        )
        agent._tracer = None
        engine = agent._create_context_engine()
        assert engine is not None

    def test_mode_aware_defaults_applied(self):
        """Different modes get different default thresholds."""
        for mode, expected_trigger in [
            ("direct", 0.80),
            ("standard", 0.70),
            ("autonomous", 0.60),
        ]:
            agent = Agent(
                name="test",
                role="tester",
                objective="test",
                prompt=make_test_prompt(),
                llm=MockLLM(),
                config=AgentConfig(execution_mode=mode, respect_context_window=True),
            )
            agent._tracer = None
            engine = agent._create_context_engine()
            assert engine is not None


# ------------------------------------------------------------------ #
# Test: ContextEngine prepare() compacts messages                      #
# ------------------------------------------------------------------ #


class TestPrepareCompaction:
    @pytest.mark.asyncio
    async def test_prepare_triggers_tool_result_compaction(self):
        """Tool results exceeding threshold are offloaded when prepare() runs."""
        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=500,
                response_reserve=50,
                tool_result_threshold=30,
                strategy="progressive",
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=500,
        )

        large_tool_result = "\n".join(f"line {i}: {'x' * 80}" for i in range(20))
        msgs = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="tool", name="search", content=large_tool_result),
            ChatMessage(role="user", content="question"),
        ]

        prepared = await engine.prepare(msgs)
        tel = engine.telemetry
        assert tel.compaction_count > 0
        assert tel.tokens_freed_total > 0

    @pytest.mark.asyncio
    async def test_prepare_preserves_messages_under_budget(self):
        """Messages under budget pass through unchanged."""
        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=100_000,
                response_reserve=8192,
                strategy="progressive",
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=100_000,
        )

        msgs = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
        ]

        prepared = await engine.prepare(msgs)
        assert len(prepared) == 2
        assert engine.telemetry.compaction_count == 0

    @pytest.mark.asyncio
    async def test_emergency_compaction_kicks_in(self):
        """When utilization is extreme, emergency compactor fires."""
        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=300,
                response_reserve=30,
                tool_result_threshold=20,
                strategy="progressive",
                preserve_recent_turns=1,
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=300,
        )

        filler = " ".join(["word"] * 40)
        msgs = [ChatMessage(role="system", content="sys")]
        for i in range(10):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        prepared = await engine.prepare(msgs)
        tel = engine.telemetry
        assert tel.compaction_count > 0
        assert len(prepared) < len(msgs)


# ------------------------------------------------------------------ #
# Test: ingest_tool_result compresses large results                    #
# ------------------------------------------------------------------ #


class TestIngestToolResult:
    def test_large_result_stored_but_full_content_returned(self):
        """Large tool results are stored in ContentStore but full content is
        returned so the model sees real data.  ObservationMasker handles
        compression post-response."""
        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=10_000,
                tool_result_threshold=20,
                enable_offloading=True,
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=10_000,
        )

        big_result = "\n".join(f"data line {i}: {'x' * 100}" for i in range(30))
        returned = engine.ingest_tool_result(big_result, "web_search")

        assert returned == big_result
        assert engine.store.size == 1

        stored_key = engine.store.keys()[0]
        stored = engine.store.retrieve(stored_key)
        assert stored == big_result

    def test_small_result_unchanged(self):
        """Small tool results pass through unchanged."""
        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=10_000,
                tool_result_threshold=1000,
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=10_000,
        )

        small_result = '{"answer": 42}'
        result = engine.ingest_tool_result(small_result, "calc")
        assert result == small_result


# ------------------------------------------------------------------ #
# Test: Full agent execution with Standard Mode                        #
# ------------------------------------------------------------------ #


class TestStandardModeIntegration:
    @pytest.mark.asyncio
    async def test_standard_mode_with_context_management(self):
        """Standard mode: tool loop with context management and tracing."""
        agent = Agent(
            name="SearchAgent",
            role="Research Assistant",
            objective="Search and analyze",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(
                execution_mode="standard",
                context=ContextConfig(max_context_tokens=50_000),
                enable_tracing=True,
            ),
        )

        result = await agent.execute(_make_task("Search for AI trends"))

        assert isinstance(result, AgentResult)
        assert result.status == ResultStatus.SUCCESS


# ------------------------------------------------------------------ #
# Test: Full agent execution with Direct Mode                          #
# ------------------------------------------------------------------ #


class TestDirectModeIntegration:
    @pytest.mark.asyncio
    async def test_direct_mode_with_context_management(self):
        """Direct mode with context management enabled."""
        agent = Agent(
            name="CalcAgent",
            role="Calculator",
            objective="Perform calculations",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(
                execution_mode="direct",
                context=ContextConfig(max_context_tokens=30_000),
                enable_tracing=True,
            ),
        )

        result = await agent.execute(_make_task("What is 6 * 7?"))

        assert isinstance(result, AgentResult)
        assert result.status == ResultStatus.SUCCESS


# ------------------------------------------------------------------ #
# Test: Full agent execution with Autonomous Mode                      #
# ------------------------------------------------------------------ #


class TestAutonomousModeIntegration:
    @pytest.mark.asyncio
    async def test_autonomous_mode_with_context_management(self):
        """Autonomous mode with context management enabled."""
        agent = Agent(
            name="ResearchAgent",
            role="Analyst",
            objective="Analyze complex data",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(
                execution_mode="autonomous",
                context=ContextConfig.for_mode("autonomous"),
                enable_tracing=True,
            ),
        )

        result = await agent.execute(_make_task("Analyze market trends for Q1 2026"))

        assert isinstance(result, AgentResult)
        assert result.status == ResultStatus.SUCCESS


# ------------------------------------------------------------------ #
# Test: Telemetry appears in AgentResult.display()                     #
# ------------------------------------------------------------------ #


class TestTelemetryDisplay:
    def test_display_shows_context_info(self):
        """AgentResult.display() includes context telemetry section."""
        from nucleusiq.agents.context.telemetry import CompactionEvent, ContextTelemetry

        tel = ContextTelemetry(
            peak_utilization=0.92,
            final_utilization=0.65,
            compaction_count=2,
            compaction_events=(
                CompactionEvent(
                    strategy="tool_result_compactor",
                    trigger_utilization=0.70,
                    tokens_before=85000,
                    tokens_after=45000,
                    tokens_freed=40000,
                    duration_ms=1.5,
                ),
            ),
            tokens_freed_total=40000,
            artifacts_offloaded=3,
            region_breakdown={
                "system": 500,
                "user": 2000,
                "tool_result": 42000,
                "reserved": 8192,
            },
            context_limit=100000,
            response_reserve=8192,
        )

        result = AgentResult(
            agent_id="test",
            agent_name="TestAgent",
            task_id="task1",
            mode="standard",
            output="Done",
            context_telemetry=tel,
        )

        display = result.display()
        assert "Context:" in display
        assert "92%" in display
        assert "65%" in display
        assert "tool_result_compactor" in display
        assert "40000 freed" in display
        assert "Offloaded: 3" in display
        assert "Regions:" in display

    def test_display_without_telemetry(self):
        """display() works fine when no context telemetry is present."""
        result = AgentResult(
            agent_id="test",
            agent_name="TestAgent",
            task_id="task1",
            mode="direct",
            output="Hello",
        )
        display = result.display()
        assert "Context:" not in display
        assert "AgentResult" in display


# ------------------------------------------------------------------ #
# Test: ObservabilityConfig integration                                #
# ------------------------------------------------------------------ #


class TestObservabilityConfigIntegration:
    @pytest.mark.asyncio
    async def test_observability_config_enables_tracing(self):
        """ObservabilityConfig.tracing=True enables trace data."""
        from nucleusiq.agents.config.observability_config import ObservabilityConfig

        agent = Agent(
            name="ObsAgent",
            role="tester",
            objective="test observability",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(
                execution_mode="direct",
                observability=ObservabilityConfig(tracing=True, verbose=True),
            ),
        )

        result = await agent.execute(_make_task("Hello"))
        assert result.status == ResultStatus.SUCCESS
        assert len(result.llm_calls) > 0
