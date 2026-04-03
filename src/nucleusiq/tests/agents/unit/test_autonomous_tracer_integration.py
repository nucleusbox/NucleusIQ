"""Autonomous mode + ExecutionTracer integration tests.

Proves that autonomous mode correctly populates AgentResult.llm_calls
(including critic calls) via the shared base_mode.call_llm / call_tool
pipeline, even though AutonomousMode delegates to a StandardMode
instance internally.

Known gap (v0.7.4): Decomposer.analyze() calls agent.llm.call() directly,
bypassing the tracer.  That call will be captured in v0.7.6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

src_dir = Path(__file__).resolve().parent.parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM


@pytest.mark.asyncio
async def test_autonomous_simple_path_traces_llm_and_critic():
    """Autonomous simple path: tool-call loop LLM + Critic LLM are traced."""
    agent = Agent(
        name="Auto",
        role="r",
        objective="o",
        narrative="n",
        llm=MockLLM(),
        config=AgentConfig(
            execution_mode="autonomous", verbose=False, enable_tracing=True
        ),
    )
    await agent.initialize()
    result = await agent.execute(Task(id="auto-1", objective="explain gravity"))

    assert result.status.value == "success"

    assert len(result.llm_calls) >= 2, (
        f"Expected >=2 llm_calls (tool_call_loop + critic), got {len(result.llm_calls)}"
    )

    purposes = [lc.purpose for lc in result.llm_calls]
    assert "critic" in purposes, (
        f"Expected a 'critic' purpose LLM call, got purposes: {purposes}"
    )

    for lc in result.llm_calls:
        assert lc.duration_ms >= 0, "LLM call duration should be non-negative"

    assert isinstance(result.warnings, tuple)
    assert isinstance(result.tool_calls, tuple)


@pytest.mark.asyncio
async def test_autonomous_tracer_resets_between_executions():
    """Each execute() call gets a fresh tracer — no data leakage."""
    agent = Agent(
        name="Auto2",
        role="r",
        objective="o",
        narrative="n",
        llm=MockLLM(),
        config=AgentConfig(
            execution_mode="autonomous", verbose=False, enable_tracing=True
        ),
    )
    await agent.initialize()

    r1 = await agent.execute(Task(id="a1", objective="question one"))
    n_llm_1 = len(r1.llm_calls)

    r2 = await agent.execute(Task(id="a2", objective="question two"))
    n_llm_2 = len(r2.llm_calls)

    assert n_llm_1 == n_llm_2, f"Tracer should reset: run1={n_llm_1}, run2={n_llm_2}"


@pytest.mark.asyncio
async def test_autonomous_critic_call_has_correct_purpose():
    """Critic LLM call records purpose as 'critic' not 'main'."""
    agent = Agent(
        name="CriticCheck",
        role="r",
        objective="o",
        narrative="n",
        llm=MockLLM(),
        config=AgentConfig(
            execution_mode="autonomous", verbose=False, enable_tracing=True
        ),
    )
    await agent.initialize()
    result = await agent.execute(Task(id="cc-1", objective="test critic purpose"))

    critic_calls = [lc for lc in result.llm_calls if lc.purpose == "critic"]
    assert len(critic_calls) >= 1, "Should have at least one critic LLM call"

    main_calls = [lc for lc in result.llm_calls if lc.purpose == "main"]
    assert len(main_calls) >= 1, "Should have at least one main LLM call"
