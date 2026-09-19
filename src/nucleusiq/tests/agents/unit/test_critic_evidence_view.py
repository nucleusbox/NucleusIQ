"""Invariant I-10 — the verifier sees what the generator saw, or knows it is seeing less.

The failure this guards against was observed live (gpt-oss 120B, 65K
window, nine invoices): the Critic was fed a curated package whose notes
section was cut at a fixed 2 500 chars, "saw" three invoices, failed a
correct nine-record answer as ungrounded, and the Refiner deleted six
records.  Three things now make that impossible:

* the package is budget-sized and reports omissions item by item;
* the Critic always receives the raw trace under a window-derived cap,
  and a notice whenever the combined view is partial;
* a FAIL reached on a partial view is downgraded to UNCERTAIN, and the
  Refiner is told not to delete content on the strength of it.
"""

from __future__ import annotations

import json

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import Critic, CritiqueResult, Verdict
from nucleusiq.agents.components.refiner import Refiner
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.diagnostics.analyzer import analyze
from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
from nucleusiq.llms.mock_llm import MockLLM

from nucleusiq.tests.conftest import make_test_prompt

DOCS = [f"office/invoice_{i:02d}.pdf" for i in range(1, 10)]


def _doc_text(i: int, repeat: int) -> str:
    sentence = (
        f"[{DOCS[i - 1]}] Invoice {i:02d} from Vendor {i:02d} dated 2026-03-{i:02d}. "
        f"Total amount {i * 1000} USD. Payment terms net 30. "
    )
    return sentence * repeat


async def _agent_after_run(window: int, *, doc_repeat: int) -> tuple[Agent, list]:
    """A real agent whose run state looks like it has read all nine docs."""
    agent = Agent(
        name="Extractor",
        role="Assistant",
        objective="Extract invoices",
        prompt=make_test_prompt(),
        llm=MockLLM(context_window=window),
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            verbose=False,
            context=ContextConfig(max_context_tokens=window),
        ),
    )
    await agent.initialize()
    await agent._setup_execution(
        {"id": "t", "objective": "Extract every invoice", "resources": DOCS}
    )
    messages: list[ChatMessage] = [ChatMessage(role="user", content="Extract")]
    for i in range(1, 10):
        text = _doc_text(i, doc_repeat)
        call_id = f"call_{i}"
        messages.append(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "name": "read_document",
                        "arguments": json.dumps({"path": DOCS[i - 1]}),
                    }
                ],
            )
        )
        messages.append(
            ChatMessage(
                role="tool", content=text, name="read_document", tool_call_id=call_id
            )
        )
        agent._activate_context_state_for_tool_result(
            tool_name="read_document",
            tool_call_id=call_id,
            tool_result=text,
            tool_args={"path": DOCS[i - 1]},
        )
    return agent, messages


def _fail_critic(mode: AutonomousMode, seen: list[str]) -> None:
    async def call_llm(agent, kwargs, **_):
        seen.append(kwargs["messages"][0]["content"])
        body = json.dumps(
            {
                "verdict": "fail",
                "score": 0.1,
                "feedback": "The evidence only contains three invoices.",
                "issues": ["records 4-9 are not supported"],
            }
        )
        return MockLLM.LLMResponse([MockLLM.Choice(MockLLM.Message(content=body))])

    mode.call_llm = call_llm  # type: ignore[method-assign]


ANSWER = json.dumps(
    {
        "documents": [
            {"path": DOCS[i - 1], "vendor": f"Vendor {i:02d}", "total_usd": i * 1000}
            for i in range(1, 10)
        ],
        "grand_total_usd": 45000,
    }
)


class TestCriticView:
    @pytest.mark.asyncio
    async def test_complete_view_keeps_the_models_fail(self) -> None:
        """Roomy window: every result whole → the Critic has full authority."""
        agent, messages = await _agent_after_run(200_000, doc_repeat=4)
        mode = AutonomousMode()
        seen: list[str] = []
        _fail_critic(mode, seen)

        critique = await mode._run_critic(agent, Critic(), "Extract", ANSWER, messages)

        assert critique.verdict == Verdict.FAIL
        assert critique.evidence_view == "complete"
        assert critique.original_verdict is None
        prompt = seen[0]
        assert "## EVIDENCE VISIBILITY" not in prompt
        # Package (map) and raw trace (territory) are both present, and the
        # coverage facts name every resource.
        assert "## Resources Processed" in prompt
        assert "9 of 9 declared resources were read" in prompt
        assert "EXECUTION TRACE" in prompt
        for doc in DOCS:
            assert doc in prompt
        views = agent._run_recorder.decisions["critic_views"]
        assert views[-1]["complete"] is True
        assert views[-1]["package_complete"] is True
        assert agent._run_recorder.counters.critic_partial_views == 0

    @pytest.mark.asyncio
    async def test_partial_view_downgrades_fail_and_says_so(self) -> None:
        """Tight window: results are capped → FAIL becomes UNCERTAIN, with notice."""
        agent, messages = await _agent_after_run(16_000, doc_repeat=60)
        mode = AutonomousMode()
        seen: list[str] = []
        _fail_critic(mode, seen)

        critique = await mode._run_critic(agent, Critic(), "Extract", ANSWER, messages)

        assert critique.verdict == Verdict.UNCERTAIN
        assert critique.original_verdict == Verdict.FAIL
        assert critique.downgraded
        assert critique.evidence_view == "partial"
        assert critique.score >= 0.4
        assert critique.feedback.startswith("[Verifier saw PARTIAL evidence")
        prompt = seen[0]
        assert "## EVIDENCE VISIBILITY" in prompt
        assert "You are seeing LESS than the generator saw" in prompt
        assert "9 of 9 declared resources were read" in prompt
        views = agent._run_recorder.decisions["critic_views"]
        assert views[-1]["complete"] is False
        assert views[-1]["raw_trace_complete"] is False
        counters = agent._run_recorder.counters
        assert counters.critic_partial_views == 1
        assert counters.critic_fail_downgraded == 1

    @pytest.mark.asyncio
    async def test_partial_view_does_not_touch_pass(self) -> None:
        agent, messages = await _agent_after_run(16_000, doc_repeat=60)
        mode = AutonomousMode()

        async def call_llm(agent, kwargs, **_):
            body = json.dumps({"verdict": "pass", "score": 0.9, "feedback": "ok"})
            return MockLLM.LLMResponse([MockLLM.Choice(MockLLM.Message(content=body))])

        mode.call_llm = call_llm  # type: ignore[method-assign]
        critique = await mode._run_critic(agent, Critic(), "Extract", ANSWER, messages)
        assert critique.verdict == Verdict.PASS
        assert critique.evidence_view == "partial"
        assert critique.original_verdict is None
        assert agent._run_recorder.counters.critic_fail_downgraded == 0

    @pytest.mark.asyncio
    async def test_per_tool_cap_accounts_for_the_package(self) -> None:
        """The raw-trace cap shrinks by the package size so both fit the window."""
        from nucleusiq.agents.modes.autonomous.critic_runner import (
            _compute_critic_per_tool_cap,
        )

        agent, messages = await _agent_after_run(65_536, doc_repeat=4)
        bare = _compute_critic_per_tool_cap(agent, messages)
        with_pkg = _compute_critic_per_tool_cap(
            agent, messages, extra_prompt_chars=40_000
        )
        assert bare is not None and with_pkg is not None
        assert with_pkg < bare
        assert bare - with_pkg == (40_000 // 4 // 9) * 4


def _report(agent: Agent):
    from nucleusiq.agents.agent_result import ResultStatus
    from nucleusiq.agents.task import Task

    task = Task(id="t", objective="Extract every invoice", resources=DOCS)
    return agent._build_run_report(task, ResultStatus.SUCCESS, "autonomous", "mock")


class TestRunReportSurface:
    @pytest.mark.asyncio
    async def test_partial_view_is_a_warning_finding(self) -> None:
        agent, messages = await _agent_after_run(16_000, doc_repeat=60)
        mode = AutonomousMode()
        _fail_critic(mode, [])
        await mode._run_critic(agent, Critic(), "Extract", ANSWER, messages)

        report = _report(agent)
        findings = analyze(report)
        codes = {f.code for f in findings}
        assert "CRITIC_PARTIAL_VIEW" in codes
        finding = next(f for f in findings if f.code == "CRITIC_PARTIAL_VIEW")
        assert finding.severity == "warning"
        assert finding.evidence["counters.critic_fail_downgraded"] == 1
        text = report.explain()
        assert "partial-evidence views 1" in text
        assert "FAIL→UNCERTAIN downgrades 1" in text

    @pytest.mark.asyncio
    async def test_complete_view_has_no_finding(self) -> None:
        agent, messages = await _agent_after_run(200_000, doc_repeat=4)
        mode = AutonomousMode()
        _fail_critic(mode, [])
        await mode._run_critic(agent, Critic(), "Extract", ANSWER, messages)
        assert "CRITIC_PARTIAL_VIEW" not in {f.code for f in analyze(_report(agent))}


class TestRefinerHonoursTheView:
    def test_partial_view_adds_the_keep_instruction(self) -> None:
        critique = CritiqueResult(
            verdict=Verdict.UNCERTAIN,
            original_verdict=Verdict.FAIL,
            evidence_view="partial",
            score=0.4,
            feedback="[Verifier saw PARTIAL evidence] only three invoices",
            issues=["records 4-9 unsupported"],
        )
        prompt = Refiner._build_revision_prompt(
            task_objective="Extract",
            candidate=ANSWER,
            critique=critique,
            tool_result_summary="## Resources Processed (harness-verified)\n- 9 of 9",
        )
        assert "## Critic Evidence Visibility" in prompt
        assert "Its own verdict was FAIL and was downgraded" in prompt
        assert "do NOT remove records" in prompt
        assert "unless the tool results below CONTRADICT them" in prompt

    def test_complete_view_leaves_the_prompt_alone(self) -> None:
        critique = CritiqueResult(
            verdict=Verdict.FAIL, evidence_view="complete", score=0.1, feedback="wrong"
        )
        prompt = Refiner._build_revision_prompt(
            task_objective="Extract",
            candidate=ANSWER,
            critique=critique,
            tool_result_summary=None,
        )
        assert "Critic Evidence Visibility" not in prompt

    @pytest.mark.asyncio
    async def test_refiner_summary_leads_with_coverage_facts(self) -> None:
        from nucleusiq.agents.modes.autonomous.refiner_runner import (
            _prepend_coverage_facts,
        )

        agent, _ = await _agent_after_run(65_536, doc_repeat=4)
        summary = _prepend_coverage_facts(agent, "[read_document] body")
        assert summary is not None
        assert summary.startswith("## Resources Processed (harness-verified)")
        assert "9 of 9 declared resources were read" in summary
        assert summary.endswith("[read_document] body")


FINDINGS = [
    {
        "id": f"batch{i}",
        "objective": f"Extract from batch {i}",
        "result": f"{DOCS[3 * (i - 1)]}: Vendor, total {(3 * i - 2) * 1000} USD",
        "status": "success",
    }
    for i in range(1, 4)
]


async def _synthesizer(*, with_tool_trace: bool) -> tuple[Agent, list[ChatMessage]]:
    """A COMPLEX parent about to verify its synthesis: the findings were
    handed over in the synthesis prompt; the parent itself may or may not
    have called tools."""
    from nucleusiq.agents.components.decomposer import Decomposer

    agent = Agent(
        name="Parent",
        role="Assistant",
        objective="Extract invoices",
        prompt=make_test_prompt(),
        llm=MockLLM(context_window=65_536),
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            verbose=False,
            context=ContextConfig(max_context_tokens=65_536),
        ),
    )
    await agent.initialize()
    await agent._setup_execution(
        {"id": "t", "objective": "Extract every invoice", "resources": DOCS}
    )
    agent._generator_inputs = {
        "label": "sub-agent findings handed to the synthesizer",
        "text": Decomposer.build_findings_section(FINDINGS, per_finding_chars=4_000),
        "truncated": [],
        "items": len(FINDINGS),
    }
    synth_prompt = Decomposer.build_synthesis_prompt(
        "Extract every invoice", FINDINGS, per_finding_chars=4_000
    )
    messages: list[ChatMessage] = [ChatMessage(role="user", content=synth_prompt)]
    if with_tool_trace:
        messages.append(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "read_document",
                        "arguments": json.dumps({"path": DOCS[0]}),
                    }
                ],
            )
        )
        messages.append(
            ChatMessage(
                role="tool",
                content=_doc_text(1, 4),
                name="read_document",
                tool_call_id="call_1",
            )
        )
    return agent, messages


class TestComplexSynthesisParity:
    """I-10 in the decompose → synthesize path.

    Observed live (Gemma 4, 65K, COMPLEX): the synthesizer made no tool
    calls, so the Critic got no trace, the curated package was empty, and
    it verified nine records against nothing — while the run report
    called the view "partial".  The generator's evidence there is the
    hand-off; the Critic must be shown the same text, and a tool-less
    generator whose inputs are all shown is a *complete* view.
    """

    def test_findings_section_is_what_the_synthesizer_reads(self) -> None:
        from nucleusiq.agents.components.decomposer import Decomposer

        section = Decomposer.build_findings_section(FINDINGS, per_finding_chars=4_000)
        prompt = Decomposer.build_synthesis_prompt(
            "Extract", FINDINGS, per_finding_chars=4_000
        )
        assert section and section in prompt
        assert section.count("### Sub-task:") == 3

    @pytest.mark.asyncio
    async def test_tool_less_synthesis_gets_the_handoff_and_is_complete(self) -> None:
        agent, messages = await _synthesizer(with_tool_trace=False)
        mode = AutonomousMode()
        seen: list[str] = []
        _fail_critic(mode, seen)

        critique = await mode._run_critic(agent, Critic(), "Extract", ANSWER, messages)

        # No trace was hidden, the hand-off is shown → the verdict stands.
        assert critique.evidence_view == "complete"
        assert critique.verdict == Verdict.FAIL
        prompt = seen[0]
        assert "## MATERIAL THE GENERATOR WAS GIVEN" in prompt
        assert "sub-agent findings handed to the synthesizer" in prompt
        assert prompt.count("### Sub-task:") == 3
        assert "## EVIDENCE VISIBILITY" not in prompt
        view = agent._run_recorder.decisions["critic_views"][-1]
        assert view["complete"] is True
        assert view["raw_trace_included"] is False
        assert view["inputs_included"] is True and view["inputs_items"] == 3
        assert agent._run_recorder.counters.critic_partial_views == 0
        assert "CRITIC_PARTIAL_VIEW" not in {f.code for f in analyze(_report(agent))}

    @pytest.mark.asyncio
    async def test_handoff_rides_along_with_a_tool_trace(self) -> None:
        agent, messages = await _synthesizer(with_tool_trace=True)
        mode = AutonomousMode()
        seen: list[str] = []
        _fail_critic(mode, seen)

        await mode._run_critic(agent, Critic(), "Extract", ANSWER, messages)

        prompt = seen[0]
        assert "## MATERIAL THE GENERATOR WAS GIVEN" in prompt
        assert "EXECUTION TRACE" in prompt
        view = agent._run_recorder.decisions["critic_views"][-1]
        assert view["raw_trace_included"] is True
        assert view["inputs_included"] is True
        assert view["complete"] is True

    def test_tool_less_generator_without_handoff_is_complete(self) -> None:
        """A pure-reasoning generator saw only the task; so does the Critic."""
        from nucleusiq.agents.modes.autonomous.critic_runner import CriticView

        view = CriticView(
            package_text="",
            package_complete=False,
            package_meta={},
            raw_trace_included=False,
            raw_trace_complete=False,
            per_tool_cap=None,
            tool_results=0,
            longest_tool_result=0,
        )
        assert view.complete is True
        assert view.notice() == ""
        assert view.inputs_section() == ""

    @pytest.mark.asyncio
    async def test_refiner_summary_leads_with_the_handoff(self) -> None:
        from nucleusiq.agents.modes.autonomous.refiner_runner import (
            _prepend_generator_inputs,
        )

        agent, _ = await _synthesizer(with_tool_trace=False)
        summary = _prepend_generator_inputs(agent, None)
        assert summary is not None
        assert summary.startswith("## Material The Generator Was Given")
        assert summary.count("### Sub-task:") == 3
        both = _prepend_generator_inputs(agent, "[read_document] body")
        assert both is not None and both.endswith("[read_document] body")
        assert (
            _prepend_generator_inputs(
                Agent(
                    name="Plain",
                    role="A",
                    objective="B",
                    prompt=make_test_prompt(),
                    llm=MockLLM(),
                ),
                "x",
            )
            == "x"
        )
