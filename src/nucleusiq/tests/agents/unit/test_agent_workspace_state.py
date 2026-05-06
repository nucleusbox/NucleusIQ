"""Agent integration tests for L2 run-local workspace state."""

from __future__ import annotations

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.llms.mock_llm import MockLLM

from nucleusiq.tests.conftest import make_test_prompt


def _agent() -> Agent:
    return Agent(
        name="WorkspaceAgent",
        role="Assistant",
        objective="Use run-local workspace",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        config=AgentConfig(verbose=False),
    )


@pytest.mark.asyncio
async def test_workspace_starts_empty_per_agent_run() -> None:
    agent = _agent()
    await agent.initialize()

    first = await agent.execute({"id": "t1", "objective": "hello"})
    agent.workspace.write_note(title="Between runs", content="must be cleared")
    second = await agent.execute({"id": "t2", "objective": "hello again"})

    assert first.metadata["workspace"]["entry_count"] == 0
    assert second.metadata["workspace"]["entry_count"] == 0
    assert agent.workspace.list() == []


@pytest.mark.asyncio
async def test_agent_result_reports_workspace_counts() -> None:
    agent = _agent()
    await agent.initialize()

    task, _mode, _agent_ctx = await agent._setup_execution(
        {"id": "t1", "objective": "hello"}
    )
    agent.workspace.write_note(title="Progress", content="Checked TCS FY25.")
    result = agent._build_result(
        task=task,
        status="success",
        output="done",
        error=None,
        error_type=None,
        t0=0.0,
    )

    workspace_meta = result.metadata["workspace"]
    assert workspace_meta["entry_count"] == 1
    assert workspace_meta["total_chars"] > 0
    assert workspace_meta["backend"] == "memory"
