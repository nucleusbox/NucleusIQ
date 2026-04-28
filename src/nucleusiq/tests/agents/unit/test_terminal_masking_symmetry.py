"""F7 — Terminal post_response symmetry between streaming and non-streaming.

Before F7, both ``StandardMode._tool_call_loop`` and
``BaseExecutionMode._streaming_tool_call_loop`` skipped
``ContextEngine.post_response`` on the terminal state — i.e. after the
LAST assistant message was appended to ``messages``.  Streaming also
skipped it on its synthesis branch entirely.

Consequences:

* The last round's tool result (which now has an assistant after it)
  was **never masked**, even though intermediate tool results were.
* Autonomous mode's Critic / Refiner inspected this un-normalised
  terminal state, and streaming vs non-streaming runs of the SAME
  conversation produced DIFFERENT masked message lists — a quiet
  source of divergence in telemetry and in Critic input.

These tests pin down the post-fix contract: every terminal path —
non-streaming tool-then-content, streaming tool-then-content, and
streaming synthesis — runs ``post_response`` once on the final
messages list, so the last round's tool results are masked and both
paths end in an identical conversation state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.config.agent_config import AgentConfig, AgentState
from nucleusiq.agents.context import (
    ContextConfig,
    ContextEngine,
    DefaultTokenCounter,
)
from nucleusiq.agents.context.strategies.observation_masker import (
    MASK_PREFIX,
)
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.agents.task import Task
from nucleusiq.streaming.events import StreamEvent

# ------------------------------------------------------------------ #
# Test helpers                                                        #
# ------------------------------------------------------------------ #


class _RecordingEngine:
    """Minimal engine stub that counts post_response invocations.

    Wraps an optional real ``ContextEngine`` so tests can assert on
    both call counts AND final masked message content.
    """

    def __init__(self, inner: ContextEngine | None = None) -> None:
        self._inner = inner
        self.prepare_calls = 0
        self.post_response_calls = 0
        self.ingest_calls = 0
        self.store = getattr(inner, "store", None)

    async def prepare(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        self.prepare_calls += 1
        if self._inner is not None:
            return await self._inner.prepare(messages)
        return messages

    def post_response(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        self.post_response_calls += 1
        if self._inner is not None:
            return self._inner.post_response(messages)
        return messages

    def ingest_tool_result(self, content: str, tool_name: str) -> str:
        self.ingest_calls += 1
        if self._inner is not None:
            return self._inner.ingest_tool_result(content, tool_name)
        return content


def _make_agent(context_engine=None) -> MagicMock:
    agent = MagicMock()
    agent.name = "t7-agent"
    agent.config = AgentConfig(enable_synthesis=True)
    agent.state = AgentState.INITIALIZING
    agent.tools = []
    agent.memory = None
    agent.prompt = None
    agent.role = "tester"
    agent.objective = "test"
    agent._logger = MagicMock()
    agent._plugin_manager = None
    agent._usage_tracker = None
    agent._tracer = None
    agent._current_llm_overrides = {}
    agent._resolve_response_format = MagicMock(return_value=None)
    agent._get_structured_output_kwargs = MagicMock(return_value={})
    agent._context_engine = context_engine

    llm = AsyncMock()
    llm.model_name = "test-model"
    agent.llm = llm
    return agent


def _response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, refusal=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=20, total_tokens=120
        ),
        model="test-model",
    )


def _tc(name: str, idx: int):
    return SimpleNamespace(
        id=f"tc_{idx}",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _real_engine() -> ContextEngine:
    # v2 step 1: terminal-masking symmetry tests verify F7 (streaming
    # and non-streaming paths both run post_response on the terminal
    # state).  They depend on the masker firing deterministically, so
    # the squeeze gate is bypassed here.  Gate behaviour itself is
    # covered separately in ``test_squeeze_gate.py``.
    return ContextEngine(
        config=ContextConfig(
            max_context_tokens=200_000,
            strategy="progressive",
            squeeze_threshold=0.0,
        ),
        token_counter=DefaultTokenCounter(),
        max_tokens=200_000,
    )


# ------------------------------------------------------------------ #
# 1. Terminal masking — non-streaming                                 #
# ------------------------------------------------------------------ #


class TestNonStreamingTerminalMasking:
    """``_tool_call_loop`` masks the final round's tool results."""

    @pytest.mark.asyncio
    async def test_terminal_tool_result_is_masked(self):
        agent = _make_agent(context_engine=_RecordingEngine(_real_engine()))

        call_count = 0

        async def fake_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _response(tool_calls=[_tc("search", 1)])
            return _response(content="Final synthesized answer.")

        agent.llm.call = fake_llm

        from nucleusiq.agents.components.executor import Executor
        from nucleusiq.tools import tool

        @tool(name="search", description="S")
        def search():
            return "x" * 1500

        agent.tools = [search]
        agent._executor = Executor(agent.llm, agent.tools)

        mode = StandardMode()
        task = Task.from_dict({"id": "t", "objective": "Run search"})
        messages = mode.build_messages(agent, task)
        result = await mode._tool_call_loop(agent, task, messages, [])

        assert result == "Final synthesized answer."
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert isinstance(tool_msgs[0].content, str)
        assert tool_msgs[0].content.startswith(MASK_PREFIX), (
            "F7: terminal tool result must be masked once the final "
            "assistant message is appended."
        )

    @pytest.mark.asyncio
    async def test_terminal_post_response_invoked_exactly_once_extra(self):
        """Per-round post_response + terminal post_response == N+1 total."""
        recording = _RecordingEngine(_real_engine())
        agent = _make_agent(context_engine=recording)

        call_count = 0

        async def fake_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _response(tool_calls=[_tc("search", 1)])
            return _response(content="done.")

        agent.llm.call = fake_llm

        from nucleusiq.agents.components.executor import Executor
        from nucleusiq.tools import tool

        @tool(name="search", description="S")
        def search():
            return "x" * 1500

        agent.tools = [search]
        agent._executor = Executor(agent.llm, agent.tools)

        mode = StandardMode()
        task = Task.from_dict({"id": "t", "objective": "x"})
        messages = mode.build_messages(agent, task)
        await mode._tool_call_loop(agent, task, messages, [])

        # 2 LLM calls → 2 per-round post_response (from call_llm) +
        # 1 terminal post_response (F7) = 3.
        assert recording.post_response_calls == 3, (
            f"Expected 3 post_response calls (2 per-round + 1 terminal); "
            f"got {recording.post_response_calls}."
        )


# ------------------------------------------------------------------ #
# 2. Terminal masking — streaming                                     #
# ------------------------------------------------------------------ #


class TestStreamingTerminalMasking:
    """Streaming loop masks the final round's tool results too."""

    @pytest.mark.asyncio
    async def test_terminal_tool_result_is_masked(self):
        agent = _make_agent(context_engine=_RecordingEngine(_real_engine()))

        call_count = 0

        async def fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamEvent.complete_event(
                    "",
                    metadata={
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    },
                )
            else:
                yield StreamEvent.complete_event(
                    "Final streamed answer.",
                    metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
                )

        agent.llm.call_stream = fake_stream

        from nucleusiq.agents.components.executor import Executor
        from nucleusiq.tools import tool

        @tool(name="search", description="S")
        def search():
            return "x" * 1500

        agent.tools = [search]
        agent._executor = Executor(agent.llm, agent.tools)

        mode = StandardMode()
        task = Task.from_dict({"id": "t", "objective": "Run search"})
        messages = mode.build_messages(agent, task)

        from nucleusiq.streaming.events import StreamEventType

        final_complete = None
        async for ev in mode._streaming_tool_call_loop(
            agent,
            messages,
            tool_specs=[],
            max_tool_calls=10,
            max_output_tokens=512,
        ):
            if ev.type == StreamEventType.COMPLETE:
                final_complete = ev

        assert final_complete is not None
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert isinstance(tool_msgs[0].content, str)
        assert tool_msgs[0].content.startswith(MASK_PREFIX), (
            "F7: streaming terminal tool result must be masked (parity "
            "with the non-streaming path)."
        )


# ------------------------------------------------------------------ #
# 3. Streaming synthesis now runs post_response                       #
# ------------------------------------------------------------------ #


class TestStreamingSynthesisFinalize:
    """Streaming synthesis no longer bypasses terminal masking."""

    @pytest.mark.asyncio
    async def test_synthesis_terminal_invokes_post_response(self):
        recording = _RecordingEngine(_real_engine())
        agent = _make_agent(context_engine=recording)

        call_count = 0

        async def fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                yield StreamEvent.complete_event(
                    "",
                    metadata={
                        "tool_calls": [
                            {
                                "id": f"tc_{call_count}",
                                "function": {
                                    "name": f"tool_{call_count}",
                                    "arguments": "{}",
                                },
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                    },
                )
            elif call_count == 4:
                yield StreamEvent.complete_event(
                    "Short.",
                    metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 5}},
                )
            else:
                yield StreamEvent.complete_event(
                    "SYNTHESIZED FULL REPORT " * 30,
                    metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 90}},
                )

        agent.llm.call_stream = fake_stream

        from nucleusiq.agents.components.executor import Executor
        from nucleusiq.tools import tool

        @tool(name="tool_1", description="T1")
        def tool_1():
            return "d1" * 800

        @tool(name="tool_2", description="T2")
        def tool_2():
            return "d2" * 800

        @tool(name="tool_3", description="T3")
        def tool_3():
            return "d3" * 800

        agent.tools = [tool_1, tool_2, tool_3]
        agent._executor = Executor(agent.llm, agent.tools)

        mode = StandardMode()
        task = Task.from_dict({"id": "t", "objective": "Analyze"})
        messages = mode.build_messages(agent, task)

        calls_before_run = recording.post_response_calls

        async for _ in mode._streaming_tool_call_loop(
            agent,
            messages,
            tool_specs=[],
            max_tool_calls=20,
            max_output_tokens=2048,
        ):
            pass

        # Structure: 3 tool rounds (each fires per-round post_response),
        # 1 terse content round, 1 synthesis.  F7 adds a terminal
        # post_response at the synthesis completion.  Before F7 the
        # streaming path had **zero** synthesis post_response calls.
        assert recording.post_response_calls > calls_before_run
        assert agent._context_engine.post_response_calls >= 4, (
            "F7: streaming synthesis must run post_response on the "
            "terminal state (previously it was skipped entirely)."
        )


# ------------------------------------------------------------------ #
# 4. Symmetry — streaming == non-streaming for same conversation       #
# ------------------------------------------------------------------ #


class TestStreamingNonStreamingSymmetry:
    """Same tool script produces an equivalent masked final state."""

    @staticmethod
    def _mask_shape(messages: list[ChatMessage]) -> list[tuple[str, bool]]:
        """Return (role, is_masked) tuples — an order-preserving fingerprint.

        Tool-result content is compared by "starts-with mask prefix"
        rather than exact bytes, since the masker stamps a random uuid
        in each marker.
        """
        out: list[tuple[str, bool]] = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else ""
            masked = m.role == "tool" and content.startswith(MASK_PREFIX)
            out.append((m.role, masked))
        return out

    @pytest.mark.asyncio
    async def test_same_shape_after_tool_then_content(self):
        # Non-streaming run
        ns_engine = _RecordingEngine(_real_engine())
        ns_agent = _make_agent(context_engine=ns_engine)

        ns_count = 0

        async def ns_llm(**kwargs):
            nonlocal ns_count
            ns_count += 1
            if ns_count == 1:
                return _response(tool_calls=[_tc("search", 1)])
            return _response(content="done.")

        ns_agent.llm.call = ns_llm

        from nucleusiq.agents.components.executor import Executor
        from nucleusiq.tools import tool

        @tool(name="search", description="S")
        def ns_search():
            return "x" * 1500

        ns_agent.tools = [ns_search]
        ns_agent._executor = Executor(ns_agent.llm, ns_agent.tools)

        ns_mode = StandardMode()
        ns_task = Task.from_dict({"id": "t", "objective": "Run search"})
        ns_messages = ns_mode.build_messages(ns_agent, ns_task)
        await ns_mode._tool_call_loop(ns_agent, ns_task, ns_messages, [])

        # Streaming run (same script, same agent config)
        st_engine = _RecordingEngine(_real_engine())
        st_agent = _make_agent(context_engine=st_engine)

        st_count = 0

        async def st_stream(**kwargs):
            nonlocal st_count
            st_count += 1
            if st_count == 1:
                yield StreamEvent.complete_event(
                    "",
                    metadata={
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    },
                )
            else:
                yield StreamEvent.complete_event(
                    "done.",
                    metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
                )

        st_agent.llm.call_stream = st_stream

        @tool(name="search", description="S")
        def st_search():
            return "x" * 1500

        st_agent.tools = [st_search]
        st_agent._executor = Executor(st_agent.llm, st_agent.tools)

        st_mode = StandardMode()
        st_task = Task.from_dict({"id": "t", "objective": "Run search"})
        st_messages = st_mode.build_messages(st_agent, st_task)
        async for _ in st_mode._streaming_tool_call_loop(
            st_agent,
            st_messages,
            tool_specs=[],
            max_tool_calls=10,
            max_output_tokens=512,
        ):
            pass

        ns_shape = self._mask_shape(ns_messages)
        st_shape = self._mask_shape(st_messages)
        assert ns_shape == st_shape, (
            "F7: streaming and non-streaming must end in the same masked "
            f"conversation shape.\n"
            f"non-streaming: {ns_shape}\nstreaming:     {st_shape}"
        )


# ------------------------------------------------------------------ #
# 5. No-engine runs stay crash-free                                   #
# ------------------------------------------------------------------ #


class TestFinalizeHelperIsDefensive:
    """``_finalize_post_response`` must tolerate missing / broken engines."""

    def test_no_engine_is_noop(self):
        from nucleusiq.agents.modes.base_mode import BaseExecutionMode

        agent = _make_agent(context_engine=None)
        messages = [ChatMessage(role="user", content="hi")]
        BaseExecutionMode._finalize_post_response(agent, messages)
        assert messages[0].content == "hi"

    def test_engine_exception_is_swallowed(self):
        from nucleusiq.agents.modes.base_mode import BaseExecutionMode

        class _Exploding:
            def post_response(self, _messages):
                raise RuntimeError("boom")

        agent = _make_agent(context_engine=_Exploding())
        messages = [ChatMessage(role="user", content="hi")]
        BaseExecutionMode._finalize_post_response(agent, messages)
        assert messages[0].content == "hi", (
            "F7 helper must never propagate exceptions — masking "
            "failures must not break the user's task."
        )

    def test_empty_messages_is_noop(self):
        from nucleusiq.agents.modes.base_mode import BaseExecutionMode

        engine = _RecordingEngine()
        agent = _make_agent(context_engine=engine)
        BaseExecutionMode._finalize_post_response(agent, [])
        assert engine.post_response_calls == 0
