"""PR A — termination invariants and the always-on run report.

Covers docs/design/AUTONOMOUS_HARNESS_HARDENING.md §11 Phase 0:

* every loop exit records a ``TerminationReason``;
* context-management tools have their own finite budget;
* a model that repeats itself is nudged once, then stopped;
* ``execute_stream`` never lets a runtime exception escape;
* ``AgentConfig`` rejects reply budgets that cannot fit the window;
* ``enable_decomposition=False`` skips the classifier;
* ``AgentResult.diagnostics`` is populated without tracing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.agent_result import ResultStatus
from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.diagnostics import (
    RunRecorder,
    RunReport,
    TerminationReason,
    recorder_for,
)
from nucleusiq.agents.modes.loop_guards import (
    NO_PROGRESS_NUDGE,
    NO_PROGRESS_ROUNDS,
    ContextToolBudget,
    ProgressTracker,
    is_stalled_tool_content,
)
from nucleusiq.agents.modes.standard_mode import _IDEMPOTENT_DEDUP_BANNER
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.streaming.events import StreamEventType
from nucleusiq.tools import BaseTool

from nucleusiq.tests.conftest import make_test_prompt

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


class ReadDocTool(BaseTool):
    """Idempotent document reader returning a stable payload per path."""

    def __init__(self, *, idempotent: bool = True):
        super().__init__(
            name="read_doc",
            description="Read a document",
            idempotent=idempotent,
        )
        self.calls: list[str] = []

    async def initialize(self) -> None:
        pass

    async def execute(self, path: str) -> str:
        self.calls.append(path)
        return f"CONTENT OF {path} " + ("x" * 200)

    def get_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }


class ScriptedLLM(MockLLM):
    """MockLLM that replays a scripted sequence of responses.

    Each script entry is either ``("tool", [(name, args_dict), ...])`` or
    ``("text", "final answer")``.  Once the script is exhausted the last
    entry repeats, which lets tests model a model that never stops.
    """

    def __init__(self, script: list[tuple[str, Any]]):
        super().__init__(model_name="scripted")
        self.script = script
        self.calls = 0
        self.saw_messages: list[list[dict[str, Any]]] = []

    async def call(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        self.saw_messages.append(messages)
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        kind, payload = self.script[idx]
        if kind == "text":
            return self.LLMResponse([self.Choice(self.Message(content=payload))])
        tool_calls = [
            {
                "id": f"call_{self.calls}_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(payload)
        ]
        return self.LLMResponse(
            [self.Choice(self.Message(content=None, tool_calls=tool_calls))]
        )


def _agent(
    llm: MockLLM, config: AgentConfig, tools: list[BaseTool] | None = None
) -> Agent:
    return Agent(
        name="HarnessAgent",
        role="Analyst",
        objective="Extract data",
        prompt=make_test_prompt(),
        llm=llm,
        tools=tools or [],
        config=config,
    )


def _tool_round(name: str, args: str, result: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCallRequest(id="c1", name=name, arguments=args)],
        ),
        ChatMessage(role="tool", name=name, tool_call_id="c1", content=result),
    ]


# --------------------------------------------------------------------------- #
# loop_guards — pure unit tests                                                #
# --------------------------------------------------------------------------- #


class TestProgressTracker:
    def test_identical_rounds_nudge_then_stop(self):
        tracker = ProgressTracker()
        verdicts = [
            tracker.observe(_tool_round("read_doc", '{"path": "a"}', "same"))
            for _ in range(NO_PROGRESS_ROUNDS)
        ]
        assert not verdicts[0].no_progress and not verdicts[0].should_nudge
        assert verdicts[1].should_nudge and not verdicts[1].no_progress
        assert verdicts[-1].no_progress
        assert "same results" in verdicts[-1].reason

    def test_changing_results_never_trip(self):
        tracker = ProgressTracker()
        for i in range(10):
            v = tracker.observe(_tool_round("poll", "{}", f"tick {i}"))
            assert not v.no_progress
            assert not v.should_nudge

    def test_changing_arguments_never_trip(self):
        tracker = ProgressTracker()
        for i in range(10):
            v = tracker.observe(_tool_round("read_doc", f'{{"path": "d{i}"}}', "same"))
            assert not v.no_progress

    def test_stalled_rounds_trip_even_when_calls_differ(self):
        tracker = ProgressTracker()
        banner = _IDEMPOTENT_DEDUP_BANNER.format(
            tool_name="read_doc", args_preview="{}", original_call_id="c0"
        )
        rounds = [
            _tool_round("read_doc", '{"path": "a"}', banner),
            _tool_round("recall_tool_result", '{"ref": "x"}', "[recall_error] gone"),
            _tool_round("read_doc", '{"path": "b"}', banner),
        ]
        verdicts = [tracker.observe(r) for r in rounds]
        assert all(v.stalled_round for v in verdicts)
        assert verdicts[-1].no_progress
        assert "duplicate-call banners" in verdicts[-1].reason

    def test_stalled_streak_resets_on_real_result(self):
        tracker = ProgressTracker()
        tracker.observe(_tool_round("a", "{}", "[recall_error] x"))
        tracker.observe(_tool_round("a", "{}", "[recall_error] x"))
        v = tracker.observe(_tool_round("b", "{}", "real payload"))
        assert v.stalled_streak == 0
        assert not v.no_progress

    def test_nudge_message_is_sent_once(self):
        tracker = ProgressTracker()
        msg = tracker.nudge_message()
        assert msg is not None and msg.role == "user"
        assert msg.content == NO_PROGRESS_NUDGE
        assert tracker.nudge_message() is None

    def test_is_stalled_tool_content(self):
        assert is_stalled_tool_content("[recall_error] nope")
        assert is_stalled_tool_content(
            "  [duplicate idempotent call — short-circuited]"
        )
        assert not is_stalled_tool_content("real data")
        assert not is_stalled_tool_content(None)


class TestContextToolBudget:
    def test_exhaustion(self):
        b = ContextToolBudget(3)
        b.consume(2)
        assert not b.exhausted
        b.consume()
        assert b.exhausted

    def test_config_default_is_twice_tool_budget(self):
        cfg = AgentConfig(execution_mode=ExecutionMode.STANDARD, max_tool_calls=40)
        assert cfg.get_effective_max_context_tool_calls() == 80
        cfg = AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS)
        assert cfg.get_effective_max_context_tool_calls() == 600

    def test_config_explicit_wins(self):
        cfg = AgentConfig(max_tool_calls=40, max_context_tool_calls=5)
        assert cfg.get_effective_max_context_tool_calls() == 5


# --------------------------------------------------------------------------- #
# AgentConfig validators                                                       #
# --------------------------------------------------------------------------- #


class TestConfigValidators:
    def test_defaults_do_not_raise(self):
        AgentConfig(context=ContextConfig())
        AgentConfig(context=ContextConfig(max_context_tokens=8_000))
        AgentConfig(llm_max_output_tokens=16_000, context=ContextConfig())

    def test_reserve_must_be_below_window(self):
        with pytest.raises(ValueError, match="response_reserve"):
            AgentConfig(
                context=ContextConfig(max_context_tokens=8_000, response_reserve=8_000)
            )

    def test_output_tokens_must_fit_explicit_reserve(self):
        with pytest.raises(ValueError, match="llm_max_output_tokens"):
            AgentConfig(
                llm_max_output_tokens=4_096,
                context=ContextConfig(
                    max_context_tokens=65_536, response_reserve=2_048
                ),
            )

    def test_valid_explicit_budget(self):
        cfg = AgentConfig(
            llm_max_output_tokens=2_048,
            context=ContextConfig(max_context_tokens=65_536, response_reserve=4_096),
        )
        assert cfg.context is not None and cfg.context.response_reserve == 4_096


# --------------------------------------------------------------------------- #
# StandardMode loop — termination reasons end to end                           #
# --------------------------------------------------------------------------- #


class TestLoopTermination:
    @pytest.mark.asyncio
    async def test_clean_completion_reports_completed(self):
        tool = ReadDocTool()
        llm = ScriptedLLM(
            [("tool", [("read_doc", {"path": "a.pdf"})]), ("text", "Done: a.pdf")]
        )
        agent = _agent(llm, AgentConfig(execution_mode=ExecutionMode.STANDARD), [tool])
        result = await agent.execute(Task(id="t", objective="read a"))

        assert result.status == ResultStatus.SUCCESS
        assert result.termination_reason == TerminationReason.COMPLETED.value
        assert isinstance(result.diagnostics, RunReport)
        assert result.diagnostics.counters.llm_calls == 2
        assert result.diagnostics.counters.tool_calls_business == 1
        assert result.diagnostics.counters.rounds == 2
        assert result.diagnostics.config_resolved["tool_count"] == 1
        assert result.diagnostics.config_resolved["idempotent_tool_count"] == 1
        assert agent.last_run_report is result.diagnostics

    @pytest.mark.asyncio
    async def test_repeated_identical_calls_are_nudged_then_stopped(self):
        """Non-idempotent tool, same call, same result forever → no_progress."""
        tool = ReadDocTool(idempotent=False)
        llm = ScriptedLLM([("tool", [("read_doc", {"path": "a.pdf"})])])
        agent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.STANDARD,
                max_tool_calls=50,
                enable_synthesis=False,
            ),
            [tool],
        )
        result = await agent.execute(Task(id="t", objective="loop"))

        assert result.termination_reason == TerminationReason.NO_PROGRESS.value
        assert result.status == ResultStatus.ERROR
        assert "No progress" in (result.error or "")
        # Stopped well before the tool budget.
        assert len(tool.calls) == NO_PROGRESS_ROUNDS
        # The nudge was injected one round before the stop.
        nudged = [
            m
            for msgs in llm.saw_messages
            for m in msgs
            if m.get("role") == "user" and "[loop guard]" in str(m.get("content"))
        ]
        assert nudged, "expected the loop-guard nudge in the transcript"
        assert result.diagnostics.counters.no_progress_rounds >= 1

    @pytest.mark.asyncio
    async def test_dedup_banner_spam_is_stopped(self):
        """Idempotent tool re-fetched forever → banners only → no_progress."""
        tool = ReadDocTool(idempotent=True)
        llm = ScriptedLLM([("tool", [("read_doc", {"path": "a.pdf"})])])
        agent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.STANDARD,
                max_tool_calls=50,
                enable_synthesis=False,
            ),
            [tool],
        )
        result = await agent.execute(Task(id="t", objective="loop"))

        assert result.termination_reason == TerminationReason.NO_PROGRESS.value
        assert len(tool.calls) == 1  # dedup short-circuited every repeat
        assert result.diagnostics.counters.dedup_banners >= NO_PROGRESS_ROUNDS - 1

    @pytest.mark.asyncio
    async def test_context_tool_budget_is_finite(self):
        """Recall calls never touch max_tool_calls but are still bounded."""
        tool = ReadDocTool()
        # Distinct refs each round; every recall fails with [recall_error].
        # A cap of 2 is reached before the stalled-round guard (3) so this
        # isolates the context-tool budget.
        script: list[tuple[str, Any]] = [
            ("tool", [("recall_tool_result", {"ref": f"ref-{i}"})]) for i in range(100)
        ]
        llm = ScriptedLLM(script)
        agent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.STANDARD,
                max_tool_calls=50,
                max_context_tool_calls=2,
                enable_synthesis=False,
            ),
            [tool],
        )
        result = await agent.execute(Task(id="t", objective="recall forever"))

        assert result.termination_reason == TerminationReason.CONTEXT_TOOL_BUDGET.value
        assert result.diagnostics.counters.tool_calls_context == 2
        assert result.diagnostics.counters.tool_calls_business == 0
        assert result.diagnostics.counters.recall_errors == 2

    @pytest.mark.asyncio
    async def test_recall_error_spam_is_stopped_by_stall_guard(self):
        """Without an explicit cap, endless recall errors still terminate."""
        tool = ReadDocTool()
        script: list[tuple[str, Any]] = [
            ("tool", [("recall_tool_result", {"ref": f"ref-{i}"})]) for i in range(100)
        ]
        llm = ScriptedLLM(script)
        agent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.STANDARD,
                max_tool_calls=50,
                enable_synthesis=False,
            ),
            [tool],
        )
        result = await agent.execute(Task(id="t", objective="recall forever"))
        assert result.termination_reason == TerminationReason.NO_PROGRESS.value
        assert result.diagnostics.counters.tool_calls_context == NO_PROGRESS_ROUNDS
        assert result.diagnostics.counters.stalled_rounds == NO_PROGRESS_ROUNDS

    @pytest.mark.asyncio
    async def test_tool_budget_reason_preserved(self):
        tool = ReadDocTool(idempotent=False)
        script: list[tuple[str, Any]] = [
            ("tool", [("read_doc", {"path": f"d{i}.pdf"})]) for i in range(100)
        ]
        llm = ScriptedLLM(script)
        agent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.STANDARD,
                max_tool_calls=3,
                enable_synthesis=False,
            ),
            [tool],
        )
        result = await agent.execute(Task(id="t", objective="many docs"))
        assert result.termination_reason == TerminationReason.TOOL_BUDGET.value
        assert result.error_type == "ToolCallLimitError"
        assert result.diagnostics.termination.at_tool_call == 3

    @pytest.mark.asyncio
    async def test_guard_stop_still_synthesises_when_allowed(self):
        """A guard stop is not a failure if the model can still answer."""
        tool = ReadDocTool(idempotent=False)
        script: list[tuple[str, Any]] = [
            ("tool", [("read_doc", {"path": "a.pdf"})])
        ] * NO_PROGRESS_ROUNDS + [("text", "Final answer from evidence")]
        llm = ScriptedLLM(script)
        agent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.STANDARD,
                max_tool_calls=50,
                enable_synthesis=True,
            ),
            [tool],
        )
        result = await agent.execute(Task(id="t", objective="loop then answer"))
        assert result.status == ResultStatus.SUCCESS
        assert str(result.output) == "Final answer from evidence"
        # The report keeps the guard reason: the run *was* rescued.
        assert result.termination_reason == TerminationReason.NO_PROGRESS.value


# --------------------------------------------------------------------------- #
# execute_stream never leaks exceptions                                        #
# --------------------------------------------------------------------------- #


class TestStreamHardening:
    @pytest.mark.asyncio
    async def test_runtime_exception_becomes_error_event(self):
        class ExplodingLLM(MockLLM):
            async def call_stream(self, **kwargs: Any):
                raise RuntimeError("provider exploded")
                yield  # pragma: no cover

        agent = _agent(ExplodingLLM(), AgentConfig(execution_mode=ExecutionMode.DIRECT))
        events = [e async for e in agent.execute_stream(Task(id="t", objective="hi"))]
        assert events, "stream must yield at least one event"
        assert any(e.type == StreamEventType.ERROR for e in events)
        assert agent.last_run_report is not None
        assert agent.last_run_report.termination.reason == TerminationReason.ERROR


# --------------------------------------------------------------------------- #
# enable_decomposition                                                         #
# --------------------------------------------------------------------------- #


class TestEnableDecomposition:
    @pytest.mark.asyncio
    async def test_classifier_skipped_when_disabled(self, monkeypatch):
        from nucleusiq.agents.components import decomposer as decomposer_mod

        async def _boom(self, agent, task):  # pragma: no cover - must not run
            raise AssertionError("Decomposer.analyze must not be called")

        monkeypatch.setattr(decomposer_mod.Decomposer, "analyze", _boom)

        llm = ScriptedLLM([("text", "single-agent answer")])
        agent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                enable_decomposition=False,
                max_retries=1,
            ),
        )
        result = await agent.execute(Task(id="t", objective="do the whole job"))
        assert result.status in (ResultStatus.SUCCESS, ResultStatus.ABSTAINED)
        decision = result.diagnostics.decisions["classification"]
        assert decision["classifier_called"] is False
        assert decision["is_complex"] is False
        assert decision["reason"] == "enable_decomposition=False"

    @pytest.mark.asyncio
    async def test_classifier_skipped_when_single_sub_agent(self, monkeypatch):
        from nucleusiq.agents.components import decomposer as decomposer_mod

        called = []

        async def _spy(self, agent, task):  # pragma: no cover - must not run
            called.append(1)
            raise AssertionError("must not be called")

        monkeypatch.setattr(decomposer_mod.Decomposer, "analyze", _spy)
        llm = ScriptedLLM([("text", "answer")])
        agent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS, max_sub_agents=1, max_retries=1
            ),
        )
        await agent.execute(Task(id="t", objective="x"))
        assert not called


# --------------------------------------------------------------------------- #
# RunReport / RunRecorder                                                      #
# --------------------------------------------------------------------------- #


class TestRunReport:
    def test_last_termination_wins_and_default_applies(self):
        rec = RunRecorder()
        rec.set_termination(TerminationReason.TOOL_BUDGET, "first attempt")
        rec.set_termination("completed", "retry succeeded")
        assert rec.termination_reason == TerminationReason.COMPLETED

        report = rec.build(
            framework_version="x",
            provider=None,
            model=None,
            mode="standard",
            agent_name="a",
            task_id="t",
            status="success",
            config_resolved={},
            default_reason=TerminationReason.ERROR,
        )
        assert report.termination.reason == TerminationReason.COMPLETED

        empty = RunRecorder().build(
            framework_version="x",
            provider=None,
            model=None,
            mode="standard",
            agent_name="a",
            task_id="t",
            status="error",
            config_resolved={},
            default_reason=TerminationReason.ERROR,
        )
        assert empty.termination.reason == TerminationReason.ERROR

    def test_unknown_reason_string_maps_to_error(self):
        rec = RunRecorder()
        rec.set_termination("something-new")
        assert rec.termination_reason == TerminationReason.ERROR

    def test_redacted_and_explain(self):
        rec = RunRecorder()
        rec.record_decision(
            "classification", {"is_complex": True, "reasoning": "secret task text"}
        )
        rec.coverage = {"unprocessed": ["/private/a.pdf"], "total": 1}
        rec.record_child({"id": "s1", "touched_resources": ["/private/a.pdf"]})
        rec.set_termination(TerminationReason.COMPLETED)
        report = rec.build(
            framework_version="0.0",
            provider="openai_compatible",
            model="gemma",
            mode="autonomous",
            agent_name="a",
            task_id="my-task",
            status="success",
            config_resolved={"context_window": 65_536},
            default_reason=TerminationReason.COMPLETED,
        )
        red = report.redacted()
        dumped = json.dumps(red.to_dict())
        assert "secret task text" not in dumped
        assert "/private/a.pdf" not in dumped
        assert red.task_id != "my-task"
        text = report.explain()
        assert "termination **completed**" in text
        assert "window 65536" in text

    def test_recorder_for_ignores_mocks(self):
        from unittest.mock import MagicMock

        assert recorder_for(MagicMock()) is None
        agent = MagicMock()
        agent._run_recorder = RunRecorder()
        assert recorder_for(agent) is agent._run_recorder
