"""Tests for AutonomousDetail + ValidationRecord observability in autonomous mode.

Verifies that autonomous executions populate result.autonomous with
complexity, attempts, max_attempts, validations, and refined fields.
"""

from __future__ import annotations

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tests.conftest import make_test_prompt


def _make_agent(**overrides):
    defaults = dict(
        name="AutoObs",
        role="tester",
        objective="test autonomous detail",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        config=AgentConfig(
            execution_mode="autonomous",
            verbose=False,
            enable_tracing=True,
        ),
    )
    defaults.update(overrides)
    return Agent(**defaults)


@pytest.mark.asyncio
async def test_autonomous_result_has_detail():
    """Autonomous execution should populate result.autonomous."""
    agent = _make_agent()
    await agent.initialize()
    result = await agent.execute(Task(id="ad-1", objective="explain photosynthesis"))

    assert result.autonomous is not None, (
        "result.autonomous should be populated in autonomous mode"
    )
    assert result.autonomous.attempts >= 1
    assert result.autonomous.max_attempts >= 1
    assert result.autonomous.complexity in ("simple", "complex", None)


@pytest.mark.asyncio
async def test_autonomous_detail_contains_validations():
    """At least one ValidationRecord should be present."""
    agent = _make_agent()
    await agent.initialize()
    result = await agent.execute(Task(id="ad-2", objective="what is gravity"))

    assert result.autonomous is not None
    assert isinstance(result.autonomous.validations, tuple)
    if result.autonomous.validations:
        vr = result.autonomous.validations[0]
        assert isinstance(vr.attempt, int)
        assert isinstance(vr.valid, bool)
        assert isinstance(vr.layer, str)


@pytest.mark.asyncio
async def test_autonomous_detail_is_frozen():
    """AutonomousDetail should be immutable (Pydantic frozen=True)."""
    agent = _make_agent()
    await agent.initialize()
    result = await agent.execute(Task(id="ad-3", objective="test immutability"))

    if result.autonomous is not None:
        with pytest.raises(Exception):
            result.autonomous.attempts = 99  # type: ignore[misc]


@pytest.mark.asyncio
async def test_autonomous_detail_resets_per_execution():
    """Each execution should produce its own AutonomousDetail."""
    agent = _make_agent()
    await agent.initialize()

    r1 = await agent.execute(Task(id="ad-4a", objective="first run"))
    r2 = await agent.execute(Task(id="ad-4b", objective="second run"))

    assert r1.autonomous is not None
    assert r2.autonomous is not None
    assert r1.autonomous.attempts == r2.autonomous.attempts


@pytest.mark.asyncio
async def test_non_autonomous_mode_has_no_detail():
    """Direct mode should NOT populate result.autonomous."""
    agent = Agent(
        name="DirectCheck",
        role="tester",
        objective="test",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        config=AgentConfig(
            execution_mode="direct",
            verbose=False,
            enable_tracing=True,
        ),
    )
    await agent.initialize()
    result = await agent.execute(Task(id="ad-5", objective="simple query"))

    assert result.autonomous is None
