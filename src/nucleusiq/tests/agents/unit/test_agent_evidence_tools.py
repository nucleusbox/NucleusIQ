"""Agent integration tests for L3 evidence tools."""

from __future__ import annotations

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.agents.context.document_corpus_tools import (
    SEARCH_DOCUMENT_CORPUS_TOOL_NAME,
)
from nucleusiq.agents.context.evidence_tools import (
    ADD_EVIDENCE_TOOL_NAME,
    is_evidence_tool_name,
)
from nucleusiq.agents.context.workspace_tools import is_context_management_tool_name
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools.decorators import tool

from nucleusiq.tests.conftest import make_test_prompt


@tool
async def user_lookup(query: str = "default") -> str:
    """User-provided lookup tool."""
    return f"result for {query}"


def _agent(*, tools: list | None = None, max_tool_calls: int | None = None) -> Agent:
    return Agent(
        name="EvidenceToolAgent",
        role="Assistant",
        objective="Use evidence tools",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        tools=list(tools or []),
        config=AgentConfig(verbose=False, max_tool_calls=max_tool_calls),
    )


@pytest.mark.asyncio
async def test_evidence_tools_auto_injected_with_user_tools() -> None:
    agent = _agent(tools=[user_lookup])
    await agent.initialize()

    await agent._setup_execution({"id": "t1", "objective": "hello"})

    names = {tool.name for tool in agent.tools}
    assert "user_lookup" in names
    assert ADD_EVIDENCE_TOOL_NAME in names


@pytest.mark.asyncio
async def test_document_corpus_tools_auto_injected_with_user_tools() -> None:
    agent = _agent(tools=[user_lookup])
    await agent.initialize()

    await agent._setup_execution({"id": "t1", "objective": "hello"})

    names = {tool.name for tool in agent.tools}
    assert SEARCH_DOCUMENT_CORPUS_TOOL_NAME in names


@pytest.mark.asyncio
async def test_evidence_tools_write_to_current_run_dossier() -> None:
    agent = _agent(tools=[user_lookup])
    await agent.initialize()

    await agent._setup_execution({"id": "t1", "objective": "first"})
    add_tool = next(tool for tool in agent.tools if tool.name == ADD_EVIDENCE_TOOL_NAME)
    result = await add_tool.execute(
        claim="TCS revenue found.",
        source_ref="obs:tcs",
        tags=["company:tcs", "metric:revenue"],
    )

    assert result["status"] == "supported"
    assert agent.evidence_dossier.stats().supported_count == 1

    await agent._setup_execution({"id": "t2", "objective": "second"})
    assert agent.evidence_dossier.list() == []


@pytest.mark.asyncio
async def test_evidence_dossier_metadata_visible_in_result() -> None:
    agent = _agent(tools=[user_lookup])
    await agent.initialize()

    task, _mode, _agent_ctx = await agent._setup_execution(
        {"id": "t1", "objective": "hello"}
    )
    agent.evidence_dossier.add_evidence(
        claim="TCS revenue found.",
        source_ref="obs:tcs",
        tags=("company:tcs",),
    )
    agent.evidence_dossier.add_gap(
        question="Need Wipro revenue.",
        reason="Not collected yet.",
        tags=("company:wipro",),
    )
    result = agent._build_result(
        task=task,
        status="success",
        output="done",
        error=None,
        error_type=None,
        t0=0.0,
    )

    evidence_meta = result.metadata["evidence"]
    assert evidence_meta["item_count"] == 2
    assert evidence_meta["supported_count"] == 1
    assert evidence_meta["gap_count"] == 1


@pytest.mark.asyncio
async def test_evidence_tools_do_not_count_as_user_tool_budget() -> None:
    agent = _agent(tools=[user_lookup], max_tool_calls=1)
    await agent.initialize()

    await agent._setup_execution({"id": "t1", "objective": "hello"})

    user_tool_count = sum(
        1
        for tool in agent.tools
        if not is_context_management_tool_name(getattr(tool, "name", None))
    )
    assert user_tool_count == 1
    assert any(
        is_evidence_tool_name(getattr(tool, "name", None)) for tool in agent.tools
    )
