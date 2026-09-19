"""Autonomous harness hardening — PR F: coverage reconciliation (WS-4, I-8).

* ``CoverageReport`` / ``build_coverage`` — ``unprocessed = resources − touched``,
  corpus document ids count as touched, answers that *name* a resource
  acknowledge it, ``enforce`` turns a residual gap into ``blocked``;
* ``coverage_retry_message`` — one bounded retry prompt, spent once, never
  without business tools, never when the flag is off;
* SIMPLE Autonomous: the retry runs only when an attempt is left and the
  model then processes the missing resource;
* COMPLEX Autonomous: one ``coverage-followup`` child on exactly the
  unprocessed slice, appended as a finding; decision + events recorded;
* every run with ``Task.resources`` carries ``metadata["coverage"]`` and
  ``diagnostics.coverage`` (STANDARD too); ``evidence_gate_enforce`` blocks
  with ``ABSTAINED`` + ``abstention_code="coverage_incomplete"``;
* ``RunReport.redacted()`` hashes nested resource lists and tolerates the
  integer ``touched_resources`` count on children.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.agent_result import ResultStatus
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.context.coverage import (
    CoverageReport,
    ResourceTouchTracker,
    acknowledged,
    build_coverage,
    coverage_retry_message,
    has_business_tools,
)
from nucleusiq.agents.context.document_search import InMemoryDocumentCorpus
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools import BaseTool

from nucleusiq.tests.conftest import make_test_prompt

DOC_TEXT = (
    "Invoice 4471 from Acme Corp dated 2026-03-01. Total amount 12,500 USD. "
    "Payment terms net 30. Line items: consulting services March, travel "
    "expenses, and software licences for the analytics platform. "
    "Approved by finance on 2026-03-05 and archived in the vendor folder."
)


class ScriptedLLM(MockLLM):
    def __init__(self, script: list[tuple[str, Any]], **kwargs: Any):
        super().__init__(model_name="scripted", **kwargs)
        self.script = script
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    async def call(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        self.requests.append({"messages": messages, **kwargs})
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        kind, payload = self.script[idx]
        if kind == "text":
            return self.LLMResponse([self.Choice(self.Message(content=payload))])
        tool_calls = [
            {
                "id": f"call_{self.calls}_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(payload)
        ]
        return self.LLMResponse(
            [self.Choice(self.Message(content=None, tool_calls=tool_calls))]
        )


class ReadDocTool(BaseTool):
    def __init__(self, *, log: list[str] | None = None):
        super().__init__(name="read_document", description="Read", idempotent=True)
        self.log = log if log is not None else []

    async def initialize(self) -> None:
        pass

    async def execute(self, path: str) -> str:
        self.log.append(path)
        return f"[{path}]\n{DOC_TEXT}"

    def get_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }


def _agent(llm: MockLLM, config: AgentConfig, *, tools=None) -> Agent:
    return Agent(
        name="Cov",
        role="Analyst",
        objective="Analyse",
        prompt=make_test_prompt(),
        llm=llm,
        tools=tools or [],
        config=config,
    )


def _auto(**overrides: Any) -> AgentConfig:
    base = {
        "execution_mode": ExecutionMode.AUTONOMOUS,
        "max_tool_calls": 10,
        "max_retries": 3,
        "enable_decomposition": False,
    }
    base.update(overrides)
    return AgentConfig(**base)


PASS = "VERDICT: PASS\nSCORE: 0.9\nFEEDBACK: ok"
RES = ["docs/a.pdf", "docs/b.pdf"]


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


class TestCoverageReport:
    def test_build_coverage_marks_corpus_and_acknowledgement(self):
        tracker = ResourceTouchTracker(["docs/a.pdf", "docs/b.pdf", "docs/c.pdf"])
        tracker.observe(tool_name="read", tool_args={"path": "docs/a.pdf"})
        report = build_coverage(
            tracker,
            answer_text="docs/c.pdf: could not be opened (corrupt)",
            corpus_document_ids=["docs/b.pdf", ""],
            enforce=False,
        )
        assert report.touched == ["docs/a.pdf", "docs/b.pdf"]
        assert report.unprocessed == ["docs/c.pdf"]
        assert report.acknowledged == ["docs/c.pdf"]
        assert report.unaccounted == []
        assert report.complete is False and report.blocked is False
        d = report.to_dict()
        assert d["followup"] is None and d["enforce"] is False
        assert tracker.to_dict()["touched_via"]["docs/b.pdf"] == "corpus"

    def test_enforce_blocks_even_when_acknowledged(self):
        report = CoverageReport(
            resources=["a"], unprocessed=["a"], acknowledged=["a"], enforce=True
        )
        assert report.blocked is True
        assert CoverageReport(resources=["a"], enforce=True).blocked is False

    def test_acknowledged_matches_aliases(self):
        assert acknowledged("docs/Q3 Report.pdf", "I could not read q3 report.pdf")
        assert not acknowledged("docs/a.pdf", "")
        assert not acknowledged("docs/a.pdf", "nothing relevant")


class TestRetryMessage:
    class Holder:
        pass

    def _agent(self, *, resources, tools=None, followup=None, flag=True):
        a = self.Holder()
        a._resource_tracker = ResourceTouchTracker(resources)
        a._coverage_followup = followup
        a.tools = tools if tools is not None else [ReadDocTool()]
        a.config = self.Holder()
        a.config.coverage_followup = flag
        a._document_corpus = InMemoryDocumentCorpus()
        a._run_recorder = None
        return a

    def test_message_lists_pending_and_is_spent_once(self):
        a = self._agent(resources=RES)
        msg = coverage_retry_message(a, "answer without naming anything")
        assert msg is not None
        assert "- docs/a.pdf" in msg and "- docs/b.pdf" in msg
        assert "Process ONLY these resources" in msg
        assert a._coverage_followup["kind"] == "retry"
        assert a._coverage_followup["pending"] == RES
        assert coverage_retry_message(a, "still nothing") is None

    def test_none_when_nothing_to_do(self):
        assert coverage_retry_message(self._agent(resources=[]), "x") is None
        assert coverage_retry_message(self._agent(resources=RES, tools=[]), "x") is None
        assert (
            coverage_retry_message(self._agent(resources=RES, flag=False), "x") is None
        )
        named = self._agent(resources=RES)
        assert (
            coverage_retry_message(named, "a.pdf: fine. b.pdf: could not be read")
            is None
        )
        touched = self._agent(resources=RES)
        touched._document_corpus.index_document("docs/a.pdf", "x " * 100)
        touched._resource_tracker.observe(tool_name="r", tool_args={"p": "docs/b.pdf"})
        assert coverage_retry_message(touched, "answer") is None
        assert touched._resource_tracker.touched == RES

    def test_has_business_tools_ignores_context_tools(self):
        from nucleusiq.agents.context.config import ContextConfig
        from nucleusiq.agents.context.engine import ContextEngine
        from nucleusiq.agents.context.recall_tools import build_recall_tools

        a = self.Holder()
        a.tools = build_recall_tools(ContextEngine(config=ContextConfig()))
        assert has_business_tools(a) is False
        a.tools.append(ReadDocTool())
        assert has_business_tools(a) is True


# --------------------------------------------------------------------------- #
# SIMPLE Autonomous — one bounded retry                                        #
# --------------------------------------------------------------------------- #


class TestSimpleRetry:
    @pytest.mark.asyncio
    async def test_retry_processes_missing_resource_then_completes(self):
        log: list[str] = []
        llm = ScriptedLLM(
            [
                ("tools", [("read_document", {"path": "docs/a.pdf"})]),
                ("text", "Only a: 12,500 USD"),
                # coverage retry → the model reads b and answers fully
                ("tools", [("read_document", {"path": "docs/b.pdf"})]),
                ("text", "a: 12,500 USD; b: 12,500 USD"),
                ("text", PASS),
            ]
        )
        agent = _agent(llm, _auto(), tools=[ReadDocTool(log=log)])
        result = await agent.execute(Task(id="t", objective="Totals", resources=RES))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == "a: 12,500 USD; b: 12,500 USD"
        assert log == ["docs/a.pdf", "docs/b.pdf"]
        cov = result.metadata["coverage"]
        assert cov["touched"] == RES and cov["unprocessed"] == []
        assert cov["complete"] is True and cov["blocked"] is False
        assert cov["followup"]["kind"] == "retry"
        assert cov["followup"]["pending"] == ["docs/b.pdf"]
        assert result.diagnostics.coverage == cov
        kinds = [e.kind for e in result.diagnostics.timeline]
        assert "coverage_retry" in kinds and "coverage_gap" not in kinds
        # The retry prompt named exactly the missing resource.
        retry_prompts = [
            m["content"]
            for r in llm.requests
            for m in r["messages"]
            if m.get("role") == "user" and "never processed" in str(m.get("content"))
        ]
        assert retry_prompts and "- docs/b.pdf" in retry_prompts[0]
        assert "- docs/a.pdf" not in retry_prompts[0]

    @pytest.mark.asyncio
    async def test_no_retry_when_answer_names_the_resource(self):
        llm = ScriptedLLM(
            [
                ("tools", [("read_document", {"path": "docs/a.pdf"})]),
                ("text", "a: 12,500 USD. docs/b.pdf: could not be processed (missing)"),
                ("text", PASS),
            ]
        )
        agent = _agent(llm, _auto(), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Totals", resources=RES))
        assert result.status == ResultStatus.SUCCESS
        assert llm.calls == 3
        cov = result.metadata["coverage"]
        assert cov["unprocessed"] == ["docs/b.pdf"]
        assert cov["acknowledged"] == ["docs/b.pdf"] and cov["unaccounted"] == []
        assert cov["followup"] is None
        assert "coverage_gap" in [e.kind for e in result.diagnostics.timeline]

    @pytest.mark.asyncio
    async def test_no_retry_on_last_attempt_keeps_the_answer(self):
        llm = ScriptedLLM(
            [
                ("tools", [("read_document", {"path": "docs/a.pdf"})]),
                ("text", "Only a"),
                ("text", PASS),
            ]
        )
        agent = _agent(llm, _auto(max_retries=1), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Totals", resources=RES))
        assert result.status == ResultStatus.SUCCESS
        assert result.output == "Only a"
        cov = result.metadata["coverage"]
        assert cov["unprocessed"] == ["docs/b.pdf"] and cov["followup"] is None

    @pytest.mark.asyncio
    async def test_retry_is_spent_once(self):
        llm = ScriptedLLM(
            [
                ("text", "nothing read"),
                ("text", "still nothing"),  # after the single coverage retry
                ("text", PASS),
            ]
        )
        agent = _agent(llm, _auto(max_retries=3), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Totals", resources=RES))
        assert result.status == ResultStatus.SUCCESS
        assert result.output == "still nothing"
        assert llm.calls == 3
        cov = result.metadata["coverage"]
        assert cov["unprocessed"] == RES and cov["followup"]["kind"] == "retry"

    @pytest.mark.asyncio
    async def test_flag_off_disables_retry(self):
        llm = ScriptedLLM([("text", "nothing read"), ("text", PASS)])
        agent = _agent(llm, _auto(coverage_followup=False), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Totals", resources=RES))
        assert llm.calls == 2
        assert result.metadata["coverage"]["followup"] is None


# --------------------------------------------------------------------------- #
# Gate                                                                         #
# --------------------------------------------------------------------------- #


class TestEnforceGate:
    @pytest.mark.asyncio
    async def test_enforce_blocks_residual_gap(self):
        llm = ScriptedLLM(
            [
                ("text", "nothing read"),
                ("text", "still nothing"),
                ("text", PASS),
            ]
        )
        agent = _agent(
            llm, _auto(max_retries=3, evidence_gate_enforce=True), tools=[ReadDocTool()]
        )
        result = await agent.execute(Task(id="t", objective="Totals", resources=RES))
        assert result.status == ResultStatus.ABSTAINED
        assert result.abstention_code == "coverage_incomplete"
        assert "docs/a.pdf" in (result.abstention_reason or "")
        assert result.output == "still nothing"
        assert result.metadata["coverage"]["blocked"] is True

    @pytest.mark.asyncio
    async def test_enforce_passes_when_complete(self):
        llm = ScriptedLLM(
            [
                (
                    "tools",
                    [
                        ("read_document", {"path": "docs/a.pdf"}),
                        ("read_document", {"path": "docs/b.pdf"}),
                    ],
                ),
                ("text", "both"),
                ("text", PASS),
            ]
        )
        agent = _agent(llm, _auto(evidence_gate_enforce=True), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Totals", resources=RES))
        assert result.status == ResultStatus.SUCCESS
        assert result.metadata["coverage"]["complete"] is True

    @pytest.mark.asyncio
    async def test_no_resources_means_no_coverage(self):
        llm = ScriptedLLM([("text", "done"), ("text", PASS)])
        agent = _agent(llm, _auto(evidence_gate_enforce=True), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Totals"))
        assert result.status == ResultStatus.SUCCESS
        assert "coverage" not in result.metadata
        assert result.diagnostics.coverage is None


# --------------------------------------------------------------------------- #
# STANDARD — recorded, never retried                                           #
# --------------------------------------------------------------------------- #


class TestStandardRecordsOnly:
    @pytest.mark.asyncio
    async def test_standard_records_coverage(self):
        llm = ScriptedLLM(
            [("tools", [("read_document", {"path": "docs/a.pdf"})]), ("text", "a only")]
        )
        agent = _agent(
            llm,
            AgentConfig(execution_mode=ExecutionMode.STANDARD, max_tool_calls=5),
            tools=[ReadDocTool()],
        )
        result = await agent.execute(Task(id="t", objective="Totals", resources=RES))
        assert result.status == ResultStatus.SUCCESS and llm.calls == 2
        cov = result.metadata["coverage"]
        assert cov["touched"] == ["docs/a.pdf"] and cov["unprocessed"] == ["docs/b.pdf"]
        assert cov["followup"] is None
        assert result.diagnostics.coverage["unprocessed"] == ["docs/b.pdf"]


# --------------------------------------------------------------------------- #
# COMPLEX — bounded follow-up child                                            #
# --------------------------------------------------------------------------- #

SPLIT = {
    "gate1": True,
    "gate2": True,
    "gate3": True,
    "gate4": True,
    "complexity": "complex",
    "sub_tasks": [
        {"id": "s1", "objective": "Dates from a", "resources": ["docs/a.pdf"]},
        {"id": "s2", "objective": "Amounts from b", "resources": ["docs/b.pdf"]},
    ],
}


class TestComplexFollowup:
    @pytest.mark.asyncio
    async def test_followup_child_covers_the_gap(self):
        log: list[str] = []
        llm = ScriptedLLM(
            [
                ("text", json.dumps(SPLIT)),
                # s1 reads its slice; s2 answers without reading
                ("tools", [("read_document", {"path": "docs/a.pdf"})]),
                ("text", "s2: guessed"),
                ("text", "s1: 2026-03-01"),
                # coverage follow-up child reads b
                ("tools", [("read_document", {"path": "docs/b.pdf"})]),
                ("text", "docs/b.pdf: 12,500 USD"),
                # synthesis + critic
                ("text", "Final: dates + amounts"),
                ("text", PASS),
            ]
        )
        parent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_tool_calls=20,
                max_retries=2,
            ),
            tools=[ReadDocTool(log=log)],
        )
        result = await parent.execute(Task(id="t", objective="Extract", resources=RES))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == "Final: dates + amounts"
        assert sorted(log) == ["docs/a.pdf", "docs/b.pdf"]
        decision = result.diagnostics.decisions["coverage"]
        assert decision["unprocessed_before"] == ["docs/b.pdf"]
        assert decision["unprocessed_after"] == []
        assert decision["followup"]["kind"] == "child"
        findings = result.diagnostics.decisions["sub_task_findings"]
        assert [f["id"] for f in findings] == ["s1", "s2", "coverage-followup"]
        assert result.diagnostics.counters.children_spawned == 3
        kinds = [e.kind for e in result.diagnostics.timeline]
        assert "coverage_followup" in kinds
        cov = result.metadata["coverage"]
        assert cov["complete"] is True and cov["followup"]["kind"] == "child"
        # The follow-up child was capped at 2 × len(unprocessed).
        follow = [
            c for c in result.diagnostics.children if c["id"] == "coverage-followup"
        ]
        assert follow and follow[0]["max_tool_calls"] == 2
        # Synthesis saw the follow-up finding.
        synth = [
            m["content"]
            for r in llm.requests
            for m in r["messages"]
            if m.get("role") == "user" and "SUB-AGENT FINDINGS" in str(m.get("content"))
        ]
        assert synth and "docs/b.pdf: 12,500 USD" in synth[0]

    @pytest.mark.asyncio
    async def test_followup_objective_names_only_the_gap(self):
        from nucleusiq.agents.components.decomposer import Decomposer

        text = Decomposer.build_followup_objective("Extract totals", ["docs/b.pdf"])
        assert "COVERAGE FOLLOW-UP" in text
        assert "Extract totals" in text
        assert "- docs/b.pdf" in text and "docs/a.pdf" not in text
        assert "Do not redo work" in text


# --------------------------------------------------------------------------- #
# Redaction                                                                    #
# --------------------------------------------------------------------------- #


class TestRedaction:
    @pytest.mark.asyncio
    async def test_redacted_hashes_nested_resource_lists(self):
        llm = ScriptedLLM(
            [
                ("text", "nothing read"),
                ("text", "still nothing"),
                ("text", PASS),
            ]
        )
        agent = _agent(llm, _auto(max_retries=3), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Totals", resources=RES))
        red = result.diagnostics.redacted()
        assert red.coverage is not None
        assert "docs/a.pdf" not in json.dumps(red.coverage)
        assert len(red.coverage["followup"]["pending"]) == 2
        assert red.task_id != "t"
