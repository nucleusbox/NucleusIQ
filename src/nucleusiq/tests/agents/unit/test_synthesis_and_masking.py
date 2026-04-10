"""Tests for synthesis pass + context management interaction.

Covers the bug where call_llm's post_response masking would degrade
messages before synthesis could use them, and verifies the snapshot
approach ensures synthesis always receives full context.

Test Groups:
  1. TestCallLlmMaskingConsistency — call_llm always masks the caller's list
  2. TestSynthesisSnapshot — synthesis receives pre-masking messages
  3. TestSynthesisE2E — end-to-end synthesis flow with mocked context engine
  4. TestStreamingSynthesisSnapshot — streaming path snapshot verification
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.config.agent_config import AgentConfig, AgentState
from nucleusiq.agents.context import ContextConfig, ContextEngine, DefaultTokenCounter
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.agents.task import Task
from nucleusiq.streaming.events import StreamEvent, StreamEventType


def _make_agent(**overrides):
    """Build a minimal mock agent with context engine support."""
    config = overrides.pop("config", AgentConfig(enable_synthesis=True))
    agent = MagicMock()
    agent.name = "test_agent"
    agent.config = config
    agent.state = AgentState.INITIALIZING
    agent.tools = overrides.pop("tools", [])
    agent.memory = overrides.pop("memory", None)
    agent.prompt = overrides.pop("prompt", None)
    agent.role = "tester"
    agent.objective = "test"
    agent._logger = MagicMock()
    agent._plugin_manager = None
    agent._usage_tracker = None
    agent._tracer = None
    agent._current_llm_overrides = {}
    agent._resolve_response_format = MagicMock(return_value=None)
    agent._get_structured_output_kwargs = MagicMock(return_value={})

    llm = AsyncMock()
    llm.model_name = "test-model"
    agent.llm = llm

    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


def _make_response(content=None, tool_calls=None):
    """Build a mock LLM response."""
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        refusal=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        ),
        model="test-model",
    )


def _build_messages_with_tool_results(n_tool_rounds=3, content_per_result=500):
    """Build a realistic message list simulating multiple tool-calling rounds.

    Structure per round: assistant(tool_calls) -> tool(result)
    """
    msgs = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Analyze data."),
    ]
    for i in range(n_tool_rounds):
        tc = ToolCallRequest(id=f"tc_{i}", name=f"tool_{i}", arguments="{}")
        msgs.append(
            ChatMessage(role="assistant", content=None, tool_calls=[tc])
        )
        msgs.append(
            ChatMessage(
                role="tool",
                name=f"tool_{i}",
                tool_call_id=f"tc_{i}",
                content="x" * content_per_result,
            )
        )
    return msgs


# ------------------------------------------------------------------ #
# 1. call_llm masking consistency                                      #
# ------------------------------------------------------------------ #


class TestCallLlmMaskingConsistency:
    """Verify post_response always updates the caller's message list."""

    @pytest.mark.asyncio
    async def test_masking_applied_when_no_compaction(self):
        """When prepare() returns the SAME list (no compaction), caller is masked."""
        agent = _make_agent()
        agent.llm.call = AsyncMock(return_value=_make_response("ok"))

        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=200_000,
                strategy="progressive",
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=200_000,
        )
        agent._context_engine = engine

        msgs = _build_messages_with_tool_results(n_tool_rounds=3, content_per_result=500)
        original_tool_contents = [
            m.content for m in msgs if m.role == "tool"
        ]

        mode = StandardMode()
        await mode.call_llm(agent, {"model": "m", "messages": []}, msgs, None)

        masked_tool_contents = [m.content for m in msgs if m.role == "tool"]
        masked_count = sum(
            1 for c in masked_tool_contents
            if c.startswith("[observation consumed")
        )

        assert masked_count > 0, (
            "post_response must mask consumed tool results even when no compaction fires"
        )

    @pytest.mark.asyncio
    async def test_masking_applied_when_compaction_fires(self):
        """When prepare() returns a NEW list (compaction fires), caller is STILL masked.

        This is the bug that existed before the fix: prepare() returned a new
        compacted list, so messages[:] = masked only updated the new list,
        leaving the caller's list untouched.
        """
        agent = _make_agent()
        agent.llm.call = AsyncMock(return_value=_make_response("ok"))

        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=200,
                response_reserve=20,
                tool_result_threshold=20,
                strategy="progressive",
                preserve_recent_turns=1,
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=200,
        )
        agent._context_engine = engine

        msgs = _build_messages_with_tool_results(n_tool_rounds=3, content_per_result=500)

        mode = StandardMode()
        await mode.call_llm(agent, {"model": "m", "messages": []}, msgs, None)

        tool_contents = [m.content for m in msgs if m.role == "tool"]
        masked_count = sum(
            1 for c in tool_contents
            if isinstance(c, str) and c.startswith("[observation consumed")
        )

        assert masked_count > 0, (
            "post_response must mask caller's messages even when compaction created a new list"
        )

    @pytest.mark.asyncio
    async def test_no_engine_no_masking(self):
        """Without a context engine, messages are not modified."""
        agent = _make_agent()
        agent.llm.call = AsyncMock(return_value=_make_response("ok"))
        agent._context_engine = None

        msgs = _build_messages_with_tool_results(n_tool_rounds=2)
        original_contents = [m.content for m in msgs]

        mode = StandardMode()
        await mode.call_llm(agent, {"model": "m", "messages": []}, msgs, None)

        current_contents = [m.content for m in msgs]
        assert current_contents == original_contents


# ------------------------------------------------------------------ #
# 2. Synthesis receives pre-masking snapshot                           #
# ------------------------------------------------------------------ #


class TestSynthesisSnapshot:
    """Verify _tool_call_loop passes unmasked snapshot to _synthesis_pass."""

    @pytest.mark.asyncio
    async def test_synthesis_receives_unmasked_messages(self):
        """The snapshot passed to _synthesis_pass must contain full tool results,
        not the masked versions that call_llm's post_response produces.
        """
        agent = _make_agent()

        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=200_000,
                strategy="progressive",
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=200_000,
        )
        agent._context_engine = engine

        call_count = 0
        long_result = "DATA " * 200

        def make_tool_call(name):
            return SimpleNamespace(
                id=f"tc_{name}",
                function=SimpleNamespace(name=name, arguments="{}"),
            )

        async def fake_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(tool_calls=[make_tool_call("tool_a")])
            if call_count == 2:
                return _make_response(tool_calls=[make_tool_call("tool_b")])
            if call_count == 3:
                return _make_response(tool_calls=[make_tool_call("tool_c")])
            if call_count == 4:
                return _make_response(content="Short summary.")
            return _make_response(content="FULL SYNTHESIS REPORT " * 100)

        agent.llm.call = fake_llm_call

        from nucleusiq.tools import tool

        @tool(name="tool_a", description="A")
        def tool_a():
            return long_result

        @tool(name="tool_b", description="B")
        def tool_b():
            return long_result

        @tool(name="tool_c", description="C")
        def tool_c():
            return long_result

        agent.tools = [tool_a, tool_b, tool_c]

        captured_snapshot = None
        original_synthesis = StandardMode._synthesis_pass

        async def capture_synthesis(self, ag, messages):
            nonlocal captured_snapshot
            captured_snapshot = list(messages)
            return await original_synthesis(self, ag, messages)

        mode = StandardMode()

        from nucleusiq.agents.components.executor import Executor
        agent._executor = Executor(agent.llm, agent.tools)

        task = Task.from_dict({"id": "t1", "objective": "Analyze"})

        with patch.object(StandardMode, "_synthesis_pass", capture_synthesis):
            await mode._tool_call_loop(agent, task, mode.build_messages(agent, task), [])

        assert captured_snapshot is not None, "Synthesis pass must have been called"

        # The snapshot is taken BEFORE the final call_llm's masking.
        # Older tool results may have been masked in previous rounds
        # (that's expected), but the MOST RECENT round's tool results
        # must still be unmasked — that's the data synthesis needs.
        tool_msgs = [m for m in captured_snapshot if m.role == "tool"]
        assert len(tool_msgs) > 0, "Snapshot must contain tool results"

        last_tool = tool_msgs[-1]
        assert not last_tool.content.startswith("[observation consumed"), (
            "Most recent tool result in snapshot must NOT be masked. "
            "Synthesis would lose the most recent data."
        )

        unmasked_count = sum(
            1 for m in tool_msgs
            if not m.content.startswith("[observation consumed")
        )
        assert unmasked_count >= 1, (
            "Snapshot must have at least one unmasked tool result"
        )

    @pytest.mark.asyncio
    async def test_no_synthesis_when_disabled(self):
        """When enable_synthesis=False, synthesis pass is skipped."""
        agent = _make_agent(config=AgentConfig(enable_synthesis=False))
        agent._context_engine = None

        call_count = 0

        def make_tool_call(name):
            return SimpleNamespace(
                id=f"tc_{name}",
                function=SimpleNamespace(name=name, arguments="{}"),
            )

        async def fake_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return _make_response(tool_calls=[make_tool_call(f"t{call_count}")])
            return _make_response(content="Direct answer.")

        agent.llm.call = fake_llm_call

        from nucleusiq.tools import tool

        @tool(name="t1", description="T1")
        def t1():
            return "r1"

        @tool(name="t2", description="T2")
        def t2():
            return "r2"

        @tool(name="t3", description="T3")
        def t3():
            return "r3"

        agent.tools = [t1, t2, t3]

        from nucleusiq.agents.components.executor import Executor
        agent._executor = Executor(agent.llm, agent.tools)

        mode = StandardMode()
        task = Task.from_dict({"id": "t1", "objective": "Analyze"})

        result = await mode._tool_call_loop(
            agent, task, mode.build_messages(agent, task), []
        )

        assert result == "Direct answer."
        assert call_count == 4, "Should NOT make a 5th synthesis call"

    @pytest.mark.asyncio
    async def test_no_synthesis_when_few_rounds(self):
        """Synthesis requires call_round > 2. Single tool call should not trigger it."""
        agent = _make_agent()
        agent._context_engine = None

        call_count = 0

        def make_tool_call(name):
            return SimpleNamespace(
                id=f"tc_{name}",
                function=SimpleNamespace(name=name, arguments="{}"),
            )

        async def fake_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(tool_calls=[make_tool_call("t1")])
            return _make_response(content="Answer after one tool.")

        agent.llm.call = fake_llm_call

        from nucleusiq.tools import tool

        @tool(name="t1", description="T")
        def t1():
            return "result"

        agent.tools = [t1]

        from nucleusiq.agents.components.executor import Executor
        agent._executor = Executor(agent.llm, agent.tools)

        mode = StandardMode()
        task = Task.from_dict({"id": "t1", "objective": "Quick question"})

        result = await mode._tool_call_loop(
            agent, task, mode.build_messages(agent, task), []
        )

        assert result == "Answer after one tool."
        assert call_count == 2, "Should NOT make a 3rd synthesis call"


# ------------------------------------------------------------------ #
# 3. End-to-end synthesis flow                                         #
# ------------------------------------------------------------------ #


class TestSynthesisE2E:
    """Full synthesis flow: multi-round tool calls → synthesis pass."""

    @pytest.mark.asyncio
    async def test_synthesis_replaces_terse_output(self):
        """After 3+ rounds of tool calls, synthesis produces the final output."""
        agent = _make_agent()
        agent._context_engine = None

        call_count = 0

        def make_tool_call(name):
            return SimpleNamespace(
                id=f"tc_{name}",
                function=SimpleNamespace(name=name, arguments="{}"),
            )

        async def fake_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return _make_response(tool_calls=[make_tool_call(f"t{call_count}")])
            if call_count == 4:
                return _make_response(content="Please confirm.")
            return _make_response(content="FULL REPORT " * 50)

        agent.llm.call = fake_llm_call

        from nucleusiq.tools import tool

        @tool(name="t1", description="T1")
        def t1():
            return "data1"

        @tool(name="t2", description="T2")
        def t2():
            return "data2"

        @tool(name="t3", description="T3")
        def t3():
            return "data3"

        agent.tools = [t1, t2, t3]

        from nucleusiq.agents.components.executor import Executor
        agent._executor = Executor(agent.llm, agent.tools)

        mode = StandardMode()
        task = Task.from_dict({"id": "t1", "objective": "Analyze data"})

        result = await mode._tool_call_loop(
            agent, task, mode.build_messages(agent, task), []
        )

        assert call_count == 5, "Should make 5 calls: 3 tool + 1 terse + 1 synthesis"
        assert "FULL REPORT" in result
        assert agent.state == AgentState.COMPLETED


# ------------------------------------------------------------------ #
# 4. Streaming synthesis snapshot                                      #
# ------------------------------------------------------------------ #


class TestStreamingSynthesisSnapshot:
    """Verify the streaming tool call loop uses snapshot for synthesis."""

    @pytest.mark.asyncio
    async def test_streaming_synthesis_uses_snapshot(self):
        """The streaming synthesis builds call_kwargs from the pre-masking snapshot,
        not from the masked messages list.
        """
        agent = _make_agent()

        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=200_000,
                strategy="progressive",
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=200_000,
        )
        agent._context_engine = engine

        stream_call_count = 0
        captured_synth_messages = None

        long_result = "DATA " * 200

        async def fake_call_stream(**kwargs):
            nonlocal stream_call_count
            stream_call_count += 1
            if stream_call_count <= 3:
                tc_name = f"tool_{stream_call_count}"
                yield StreamEvent.complete_event(
                    "",
                    metadata={
                        "tool_calls": [
                            {
                                "id": f"tc_{stream_call_count}",
                                "function": {"name": tc_name, "arguments": "{}"},
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    },
                )
            elif stream_call_count == 4:
                yield StreamEvent.complete_event(
                    "Short summary.",
                    metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
                )
            else:
                yield StreamEvent.complete_event(
                    "SYNTHESIZED REPORT",
                    metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
                )

        agent.llm.call_stream = fake_call_stream

        from nucleusiq.tools import tool

        @tool(name="tool_1", description="T1")
        def tool_1():
            return long_result

        @tool(name="tool_2", description="T2")
        def tool_2():
            return long_result

        @tool(name="tool_3", description="T3")
        def tool_3():
            return long_result

        agent.tools = [tool_1, tool_2, tool_3]

        from nucleusiq.agents.components.executor import Executor
        agent._executor = Executor(agent.llm, agent.tools)

        original_build = StandardMode.build_call_kwargs
        synth_messages_captured = None

        def capture_build(self, ag, messages, tool_specs, **kw):
            nonlocal synth_messages_captured
            if tool_specs is None and stream_call_count >= 4:
                synth_messages_captured = list(messages)
            return original_build(self, ag, messages, tool_specs, **kw)

        mode = StandardMode()
        task = Task.from_dict({"id": "t1", "objective": "Analyze"})
        msgs = mode.build_messages(agent, task)

        events = []
        with patch.object(StandardMode, "build_call_kwargs", capture_build):
            async for ev in mode._streaming_tool_call_loop(
                agent, msgs, [], max_tool_calls=30, max_output_tokens=2048
            ):
                events.append(ev)

        complete_events = [e for e in events if e.type == StreamEventType.COMPLETE]
        assert len(complete_events) == 1
        assert complete_events[0].content == "SYNTHESIZED REPORT"

        if synth_messages_captured is not None:
            tool_msgs = [
                m for m in synth_messages_captured if m.role == "tool"
            ]
            assert len(tool_msgs) > 0
            last_tool = tool_msgs[-1]
            assert not last_tool.content.startswith("[observation consumed"), (
                "Most recent tool result in streaming snapshot must NOT be masked"
            )
            unmasked = sum(
                1 for m in tool_msgs
                if not m.content.startswith("[observation consumed")
            )
            assert unmasked >= 1, (
                "Streaming snapshot must preserve at least the most recent tool results"
            )

    @pytest.mark.asyncio
    async def test_streaming_calls_engine_prepare(self):
        """Verify engine.prepare() is invoked in the streaming path."""
        agent = _make_agent()

        engine = ContextEngine(
            config=ContextConfig(
                max_context_tokens=200_000,
                strategy="progressive",
            ),
            token_counter=DefaultTokenCounter(),
            max_tokens=200_000,
        )
        agent._context_engine = engine

        prepare_call_count = 0
        original_prepare = ContextEngine.prepare

        async def counting_prepare(self_engine, messages):
            nonlocal prepare_call_count
            prepare_call_count += 1
            return await original_prepare(self_engine, messages)

        call_count = 0

        async def fake_call_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamEvent.complete_event(
                    "",
                    metadata={
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "function": {"name": "tool_a", "arguments": "{}"},
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    },
                )
            else:
                yield StreamEvent.complete_event(
                    "Done.",
                    metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
                )

        agent.llm.call_stream = fake_call_stream

        from nucleusiq.tools import tool

        @tool(name="tool_a", description="A")
        def tool_a():
            return "result"

        agent.tools = [tool_a]

        from nucleusiq.agents.components.executor import Executor
        agent._executor = Executor(agent.llm, agent.tools)

        mode = StandardMode()
        task = Task.from_dict({"id": "t1", "objective": "Test"})
        msgs = mode.build_messages(agent, task)

        events = []
        with patch.object(ContextEngine, "prepare", counting_prepare):
            async for ev in mode._streaming_tool_call_loop(
                agent, msgs, [], max_tool_calls=30, max_output_tokens=2048
            ):
                events.append(ev)

        assert prepare_call_count >= 2, (
            f"engine.prepare() must be called for each LLM round in streaming "
            f"(expected >=2, got {prepare_call_count})"
        )
