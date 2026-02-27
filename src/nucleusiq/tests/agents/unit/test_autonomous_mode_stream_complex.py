"""Additional coverage for AutonomousMode streaming complex path."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq.agents.components.critic import CritiqueResult, Verdict
from nucleusiq.agents.components.decomposer import TaskAnalysis
from nucleusiq.agents.components.validation import ValidationResult
from nucleusiq.agents.config.agent_config import AgentConfig, AgentState
from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
from nucleusiq.agents.task import Task
from nucleusiq.streaming.events import StreamEvent


def _agent() -> MagicMock:
    a = MagicMock()
    a._logger = MagicMock()
    a.config = AgentConfig(max_retries=2, max_sub_agents=3)
    a.state = AgentState.INITIALIZING
    a._execution_progress = None
    a.prompt = None
    a.memory = None
    a.role = "assistant"
    a.objective = "help"
    a.llm = MagicMock()
    a.llm.model_name = "dummy"
    return a


def _analysis() -> TaskAnalysis:
    return TaskAnalysis(
        is_complex=True,
        sub_tasks=[{"id": "s1", "objective": "one"}, {"id": "s2", "objective": "two"}],
    )


@pytest.mark.asyncio
async def test_stream_complex_pass_path(monkeypatch):
    agent = _agent()
    mode = AutonomousMode()
    task = Task(id="t1", objective="complex")
    decomposer = MagicMock()
    decomposer.run_sub_tasks = AsyncMock(return_value=[{"id": "s1", "result": "r1"}])
    decomposer.build_synthesis_prompt = MagicMock(return_value="synth")

    async def _loop(*_args, **_kwargs):
        yield StreamEvent.token_event("hello")
        yield StreamEvent.complete_event("hello")

    monkeypatch.setattr(mode, "_streaming_tool_call_loop", _loop)

    class _VP:
        async def validate(self, *_a, **_k):
            return ValidationResult(valid=True, layer="all", reason="ok")

    monkeypatch.setattr(
        "nucleusiq.agents.modes.autonomous_mode.ValidationPipeline",
        lambda logger=None: _VP(),
    )
    monkeypatch.setattr(
        mode,
        "_run_critic",
        AsyncMock(return_value=CritiqueResult(verdict=Verdict.PASS, score=0.9)),
    )

    events = []
    async for e in mode._stream_complex(agent, task, decomposer, _analysis()):
        events.append(e)

    assert any(e.type == "thinking" for e in events)
    assert any(e.type == "complete" for e in events)
    assert agent.state == AgentState.COMPLETED


@pytest.mark.asyncio
async def test_stream_complex_validation_retry_then_pass(monkeypatch):
    agent = _agent()
    mode = AutonomousMode()
    task = Task(id="t2", objective="complex")
    decomposer = MagicMock()
    decomposer.run_sub_tasks = AsyncMock(return_value=[{"id": "s1", "result": "r1"}])
    decomposer.build_synthesis_prompt = MagicMock(return_value="synth")

    call = {"n": 0}

    async def _loop(*_args, **_kwargs):
        call["n"] += 1
        if call["n"] == 1:
            yield StreamEvent.complete_event("first")
        else:
            yield StreamEvent.complete_event("second")

    monkeypatch.setattr(mode, "_streaming_tool_call_loop", _loop)

    class _VP:
        def __init__(self):
            self.n = 0

        async def validate(self, *_a, **_k):
            self.n += 1
            if self.n == 1:
                return ValidationResult(
                    valid=False, layer="tool_output", reason="bad", details=["d"]
                )
            return ValidationResult(valid=True, layer="all", reason="ok")

    vp = _VP()
    monkeypatch.setattr(
        "nucleusiq.agents.modes.autonomous_mode.ValidationPipeline",
        lambda logger=None: vp,
    )
    monkeypatch.setattr(
        mode,
        "_run_critic",
        AsyncMock(return_value=CritiqueResult(verdict=Verdict.PASS, score=0.9)),
    )

    events = []
    async for e in mode._stream_complex(agent, task, decomposer, _analysis()):
        events.append(e)

    assert any(
        e.type == "thinking" and "Validation failed" in (e.message or "")
        for e in events
    )
    assert agent.state == AgentState.COMPLETED


@pytest.mark.asyncio
async def test_stream_complex_handles_loop_exception(monkeypatch):
    agent = _agent()
    mode = AutonomousMode()
    task = Task(id="t3", objective="complex")
    decomposer = MagicMock()
    decomposer.run_sub_tasks = AsyncMock(return_value=[])
    decomposer.build_synthesis_prompt = MagicMock(return_value="synth")

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("synthesis failed")
        yield  # pragma: no cover

    monkeypatch.setattr(mode, "_streaming_tool_call_loop", _boom)

    events = []
    async for e in mode._stream_complex(agent, task, decomposer, _analysis()):
        events.append(e)

    assert events[-1].type == "error"
    assert "synthesis failed" in (events[-1].message or "")
    assert agent.state == AgentState.ERROR
