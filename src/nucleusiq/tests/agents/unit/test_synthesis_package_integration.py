"""Agent integration tests for L4 synthesis package helper."""

from __future__ import annotations

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.llms.mock_llm import MockLLM

from nucleusiq.tests.conftest import make_test_prompt


def _agent() -> Agent:
    return Agent(
        name="SynthesisPackageAgent",
        role="Assistant",
        objective="Build synthesis package",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        config=AgentConfig(verbose=False),
    )


@pytest.mark.asyncio
async def test_agent_can_build_synthesis_package_from_run_state() -> None:
    agent = _agent()
    await agent.initialize()
    await agent._setup_execution({"id": "t1", "objective": "Compare companies"})

    agent.workspace.write_note(title="Progress", content="Checked TCS.")
    agent.evidence_dossier.add_evidence(
        claim="TCS revenue found.",
        source_ref="obs:tcs",
        tags=("company:tcs",),
    )

    package = agent.build_synthesis_package(
        task="Compare companies",
        output_shape="Table plus recommendations",
        max_chars=2000,
    )

    assert "Compare companies" in package.text
    assert "TCS revenue found." in package.text
    assert "Progress" in package.text
    assert package.metadata["char_count"] <= 2000
