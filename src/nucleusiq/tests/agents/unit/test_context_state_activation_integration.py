"""Agent integration tests for L4.5 context state activation."""

from __future__ import annotations

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools.decorators import tool

from nucleusiq.tests.conftest import make_test_prompt


@tool
async def read_annual_report_excerpt() -> str:
    """Return a deterministic annual-report-like excerpt."""
    return (
        "TCS FY25 annual report excerpt. "
        "Revenue from operations was Rs 255,324 crore. "
        "Operating margin was 24.3%."
    )


def _agent() -> Agent:
    return Agent(
        name="ActivationAgent",
        role="Research assistant",
        objective="Collect evidence",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        tools=[read_annual_report_excerpt],
        config=AgentConfig(verbose=False, max_tool_calls=2),
    )


@pytest.mark.asyncio
async def test_business_tool_result_activates_workspace_and_evidence() -> None:
    agent = _agent()
    await agent.initialize()

    result = await agent.execute(
        {
            "id": "task",
            "objective": "Read the annual report excerpt and answer.",
        }
    )

    assert result.metadata["workspace"]["entry_count"] > 0
    assert result.metadata["evidence"]["supported_count"] > 0
    activation = result.metadata["context_activation"]
    assert activation["tool_results_seen"] >= 1
    assert activation["tool_results_activated"] >= 1
    assert activation["evidence_items_promoted"] >= 1
    doc_search = result.metadata.get("document_search") or {}
    assert doc_search.get("documents_indexed", 0) >= 1


@pytest.mark.asyncio
async def test_result_reports_synthesis_package_when_state_exists() -> None:
    agent = _agent()
    await agent.initialize()

    task, _mode, _ctx = await agent._setup_execution(
        {"id": "task", "objective": "Compare companies."}
    )
    agent.evidence_dossier.add_evidence(
        claim="TCS FY25 revenue was Rs 255,324 crore.",
        source_ref="obs:tcs",
        tags=("company:tcs", "metric:revenue"),
    )

    package_messages = agent._build_synthesis_messages_from_context(
        task="Compare companies.",
        output_shape="Return a concise answer.",
    )
    result = agent._build_result(
        task=task,
        status="success",
        output="done",
        error=None,
        error_type=None,
        t0=0.0,
    )

    assert package_messages is not None
    assert "TCS FY25 revenue" in package_messages[0].content
    assert result.metadata["synthesis_package"]["char_count"] > 0
    assert result.metadata["context_activation"]["synthesis_package_used"] is True
