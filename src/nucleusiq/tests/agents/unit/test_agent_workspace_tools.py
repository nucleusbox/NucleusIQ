"""Agent integration tests for L2.5 workspace tools."""

from __future__ import annotations

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.agents.context.workspace_tools import (
    WRITE_WORKSPACE_NOTE_TOOL_NAME,
    is_context_management_tool_name,
    is_workspace_tool_name,
)
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools.decorators import tool

from nucleusiq.tests.conftest import make_test_prompt


@tool
async def user_lookup(query: str = "default") -> str:
    """User-provided lookup tool."""
    return f"result for {query}"


def _agent(*, tools: list | None = None, max_tool_calls: int | None = None) -> Agent:
    return Agent(
        name="WorkspaceToolAgent",
        role="Assistant",
        objective="Use workspace tools",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        tools=list(tools or []),
        config=AgentConfig(verbose=False, max_tool_calls=max_tool_calls),
    )


@pytest.mark.asyncio
async def test_workspace_tools_auto_injected_with_user_tools() -> None:
    agent = _agent(tools=[user_lookup])
    await agent.initialize()

    await agent._setup_execution({"id": "t1", "objective": "hello"})

    names = {tool.name for tool in agent.tools}
    assert "user_lookup" in names
    assert WRITE_WORKSPACE_NOTE_TOOL_NAME in names


@pytest.mark.asyncio
async def test_workspace_tools_not_injected_when_no_user_tools() -> None:
    agent = _agent()
    await agent.initialize()

    await agent._setup_execution({"id": "t1", "objective": "hello"})

    assert not any(
        is_workspace_tool_name(getattr(tool, "name", None)) for tool in agent.tools
    )


@pytest.mark.asyncio
async def test_workspace_tools_are_run_local() -> None:
    agent = _agent(tools=[user_lookup])
    await agent.initialize()

    await agent._setup_execution({"id": "t1", "objective": "first"})
    note_tool = next(
        tool for tool in agent.tools if tool.name == WRITE_WORKSPACE_NOTE_TOOL_NAME
    )
    await note_tool.execute(title="Progress", content="first run note")
    assert agent.workspace.stats().entry_count == 1

    await agent._setup_execution({"id": "t2", "objective": "second"})

    assert agent.workspace.list() == []


@pytest.mark.asyncio
async def test_workspace_tools_do_not_count_as_user_tool_budget() -> None:
    agent = _agent(tools=[user_lookup], max_tool_calls=1)
    await agent.initialize()

    await agent._setup_execution({"id": "t1", "objective": "hello"})

    user_tool_count = sum(
        1
        for tool in agent.tools
        if not is_context_management_tool_name(getattr(tool, "name", None))
    )
    assert user_tool_count == 1
    assert WRITE_WORKSPACE_NOTE_TOOL_NAME in {tool.name for tool in agent.tools}
