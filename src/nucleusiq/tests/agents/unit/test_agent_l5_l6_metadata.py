"""Agent metadata tests for L5/L6 framework state."""

from __future__ import annotations

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.llms.mock_llm import MockLLM

from nucleusiq.tests.conftest import make_test_prompt


def _agent() -> Agent:
    return Agent(
        name="ContextStateAgent",
        role="Assistant",
        objective="Use framework context state",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        tools=[],
        config=AgentConfig(verbose=False),
    )


@pytest.mark.asyncio
async def test_synthesis_records_gaps_for_missing_required_evidence_tags() -> None:
    agent = Agent(
        name="ContextStateAgent",
        role="Assistant",
        objective="Use framework context state",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        tools=[],
        config=AgentConfig(
            verbose=False,
            evidence_gate_required_tags=("metric:profit",),
        ),
    )
    await agent.initialize()
    task, _mode, _ctx = await agent._setup_execution({"id": "t1", "objective": "hello"})
    agent.evidence_dossier.add_evidence(
        claim="Revenue was Rs 100 crore.",
        source_ref="obs:revenue",
        tags=("metric:revenue",),
    )

    agent._build_synthesis_messages_from_context(task="hello")

    gaps = agent.evidence_dossier.list(status="gap")
    assert any("metric:profit" in item.tags for item in gaps)


@pytest.mark.asyncio
async def test_agent_result_reports_document_search_metadata() -> None:
    agent = _agent()
    await agent.initialize()
    task, _mode, _ctx = await agent._setup_execution({"id": "t1", "objective": "hello"})

    agent.document_corpus.index_document("doc:one", "Revenue was Rs 100 crore.")
    agent.document_corpus.search("revenue")
    result = agent._build_result(
        task=task,
        status="success",
        output="done",
        error=None,
        error_type=None,
        t0=0.0,
    )

    doc_meta = result.metadata["document_search"]
    assert doc_meta["documents_indexed"] == 1
    assert doc_meta["chunks_indexed"] >= 1
    assert doc_meta["document_search_count"] == 1


@pytest.mark.asyncio
async def test_agent_result_reports_phase_control_metadata() -> None:
    agent = _agent()
    await agent.initialize()
    task, _mode, _ctx = await agent._setup_execution({"id": "t1", "objective": "hello"})

    agent.phase_controller.enter("RESEARCH")
    agent.evidence_gate.required_tags = ("metric:revenue",)
    agent.evidence_gate.enforce = True
    decision = agent.evidence_gate.evaluate(agent.evidence_dossier)
    agent.phase_controller.record_evidence_gate(decision)
    result = agent._build_result(
        task=task,
        status="success",
        output="done",
        error=None,
        error_type=None,
        t0=0.0,
    )

    phase_meta = result.metadata["phase_control"]
    assert phase_meta["phase_current"] == "RESEARCH"
    assert phase_meta["phase_transitions"] == ["PLAN", "RESEARCH"]
    assert phase_meta["evidence_gate_passed"] is False
    assert phase_meta["evidence_gate_blocked"] is True
    assert phase_meta["evidence_missing_tags"] == ["metric:revenue"]


@pytest.mark.asyncio
async def test_agent_records_organize_and_synthesize_phases_for_package() -> None:
    agent = _agent()
    await agent.initialize()
    task, _mode, _ctx = await agent._setup_execution({"id": "t1", "objective": "hello"})
    agent.evidence_dossier.add_evidence(
        claim="Revenue was Rs 100 crore.",
        source_ref="obs:revenue",
        tags=("metric:revenue",),
    )

    messages = agent._build_synthesis_messages_from_context(task="hello")
    result = agent._build_result(
        task=task,
        status="success",
        output="done",
        error=None,
        error_type=None,
        t0=0.0,
    )

    assert messages is not None
    phase_meta = result.metadata["phase_control"]
    assert "ORGANIZE_EVIDENCE" in phase_meta["phase_transitions"]
    assert "SYNTHESIZE" in phase_meta["phase_transitions"]
