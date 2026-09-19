"""Autonomous harness hardening — PR D: grounded decomposition.

Covers (docs/design/AUTONOMOUS_HARNESS_HARDENING.md WS-2):

* ``Task.resources`` (typed) with ``context["resources"]`` fallback;
* ``Task.context`` / ``resources`` rendered as one bounded block ahead of
  the objective — and nothing rendered when neither is set;
* the grounded 4-gate classifier prompt (resources, indexed documents,
  tool names; Gate 4 shared-source / single-record);
* the coverage contract in ``_parse_analysis``: unowned resource,
  resource-less sub-task or shared resource → SIMPLE with a reason;
* children receive attachments / context / metadata and their resources
  slice; the classification decision carries the gate vector.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.agent_result import ResultStatus
from nucleusiq.agents.attachments import Attachment
from nucleusiq.agents.components.decomposer import (
    Decomposer,
    TaskAnalysis,
    _resource_matches,
)
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM

from nucleusiq.tests.conftest import make_test_prompt

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

DOCS = ["docs/a.pdf", "docs/b.pdf", "docs/c.pdf"]


def _response(payload: dict[str, Any]) -> Any:
    llm = MockLLM()
    return llm.LLMResponse([llm.Choice(llm.Message(content=json.dumps(payload)))])


def _complex(sub_tasks: list[dict[str, Any]], **gates: bool) -> dict[str, Any]:
    body = {"gate1": True, "gate2": True, "gate3": True, "gate4": True}
    body.update(gates)
    body["complexity"] = "complex"
    body["sub_tasks"] = sub_tasks
    return body


class ScriptedLLM(MockLLM):
    def __init__(self, script: list[str]):
        super().__init__(model_name="scripted")
        self.script = script
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    async def call(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        self.requests.append({"messages": messages, **kwargs})
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.LLMResponse([self.Choice(self.Message(content=self.script[idx]))])


def _agent(llm: MockLLM, config: AgentConfig) -> Agent:
    return Agent(
        name="Grounded",
        role="Analyst",
        objective="Analyse",
        prompt=make_test_prompt(),
        llm=llm,
        config=config,
    )


# --------------------------------------------------------------------------- #
# Task.resources                                                               #
# --------------------------------------------------------------------------- #


class TestTaskResources:
    def test_typed_field_wins(self):
        t = Task(id="t", objective="x", resources=["a"], context={"resources": ["b"]})
        assert t.effective_resources() == ["a"]

    def test_context_fallback(self):
        t = Task(id="t", objective="x", context={"resources": ["b", "c", "b"]})
        assert t.effective_resources() == ["b", "c"]

    def test_none_and_junk_are_empty(self):
        assert Task(id="t", objective="x").effective_resources() == []
        assert (
            Task(id="t", objective="x", context={"resources": 42}).effective_resources()
            == []
        )
        assert (
            Task(
                id="t", objective="x", context={"resources": [None, "", "  "]}
            ).effective_resources()
            == []
        )

    def test_single_string_is_wrapped(self):
        t = Task(id="t", objective="x", context={"resources": "one.pdf"})
        assert t.effective_resources() == ["one.pdf"]

    def test_context_without_resources(self):
        t = Task(id="t", objective="x", context={"resources": ["a"], "k": "v"})
        assert t.context_without_resources() == {"k": "v"}

    def test_round_trip_keeps_resources(self):
        t = Task(id="t", objective="x", resources=["a", "b"])
        assert Task.from_dict(t.to_dict()).resources == ["a", "b"]


# --------------------------------------------------------------------------- #
# MessageBuilder — task context block                                          #
# --------------------------------------------------------------------------- #


class TestTaskContextRendering:
    def test_nothing_rendered_without_context(self):
        msgs = MessageBuilder.build(Task(id="t", objective="Do it"))
        assert [m.role for m in msgs] == ["user"]
        assert msgs[0].content == "Do it"

    def test_context_and_resources_precede_objective(self):
        task = Task(
            id="t",
            objective="Extract fields",
            context={"customer": "ACME", "limits": {"max": 3}},
            resources=DOCS,
        )
        msgs = MessageBuilder.build(task)
        assert [m.role for m in msgs] == ["user", "user"]
        block, objective = msgs[0].content, msgs[1].content
        assert objective == "Extract fields"
        assert "## Task Context" in block
        assert "- customer: ACME" in block
        assert '- limits: {"max": 3}' in block
        assert "## Resources (3)" in block
        for d in DOCS:
            assert f"- {d}" in block
        assert "resources" not in block.split("## Resources")[0].lower()

    def test_block_is_bounded_with_marker(self):
        task = Task(id="t", objective="x", context={"blob": "y" * 50_000})
        block = MessageBuilder.render_task_context(task, max_chars=2_000)
        assert len(block) <= 2_000
        assert block.endswith("[... task context truncated]")

    def test_many_resources_are_summarised(self):
        task = Task(id="t", objective="x", resources=[f"d{i}" for i in range(100)])
        block = MessageBuilder.render_task_context(task, max_resources=10)
        assert "## Resources (100)" in block
        assert "- … and 90 more" in block

    def test_dict_task_is_supported(self):
        block = MessageBuilder.render_task_context(
            {"id": "t", "objective": "x", "context": {"k": "v"}, "resources": ["r"]}
        )
        assert "- k: v" in block and "- r" in block

    @pytest.mark.asyncio
    async def test_agent_sends_context_block(self):
        llm = ScriptedLLM(["done"])
        agent = _agent(llm, AgentConfig(execution_mode=ExecutionMode.STANDARD))
        await agent.execute(
            Task(
                id="t",
                objective="Summarise",
                context={"tone": "formal"},
                resources=["memo.txt"],
            )
        )
        sent = llm.requests[0]["messages"]
        joined = "\n".join(str(m.get("content", "")) for m in sent)
        assert "## Task Context" in joined and "- tone: formal" in joined
        assert "## Resources (1)" in joined and "- memo.txt" in joined
        assert sent[-1]["content"] == "Summarise"


# --------------------------------------------------------------------------- #
# Classifier prompt                                                            #
# --------------------------------------------------------------------------- #


class TestClassifierPrompt:
    def test_ungrounded_prompt_has_four_gates_and_no_resource_rule(self):
        p = Decomposer.build_classifier_prompt("Compare A and B")
        assert "GATE 4" in p and "FOUR-GATE" in p
        assert "RESOURCES THE TASK MUST COVER" not in p
        assert '"resources"' not in p

    def test_grounded_prompt_lists_resources_docs_and_tools(self):
        p = Decomposer.build_classifier_prompt(
            "Extract",
            resources=DOCS,
            corpus_titles=["Indexed memo"],
            tool_names=["read_pdf", "lookup"],
        )
        assert "## RESOURCES THE TASK MUST COVER (3)" in p
        for d in DOCS:
            assert f"- {d}" in p
        assert "## DOCUMENTS ALREADY INDEXED" in p and "- Indexed memo" in p
        assert "## AVAILABLE TOOLS" in p and "read_pdf, lookup" in p
        assert "EVERY sub-task MUST list the resources it owns" in p
        assert '"resources": ["<exact resource ids from the list>"]' in p

    def test_resource_list_is_bounded(self):
        p = Decomposer.build_classifier_prompt(
            "x", resources=[f"r{i}" for i in range(100)]
        )
        assert "- … and 60 more" in p

    @pytest.mark.asyncio
    async def test_analyze_grounds_on_task_resources_and_tools(self):
        llm = ScriptedLLM([json.dumps({"complexity": "simple"})])
        agent = _agent(llm, AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS))
        await agent.initialize()
        d = Decomposer()
        analysis = await d.analyze(
            agent, Task(id="t", objective="Extract", resources=DOCS)
        )
        prompt = llm.requests[0]["messages"][0]["content"]
        assert "## RESOURCES THE TASK MUST COVER (3)" in prompt
        assert analysis.resources == DOCS
        assert not analysis.is_complex


# --------------------------------------------------------------------------- #
# Coverage contract                                                            #
# --------------------------------------------------------------------------- #


class TestCoverageContract:
    def test_resource_matching(self):
        assert _resource_matches("docs/a.pdf", "docs/a.pdf")
        assert _resource_matches("a.pdf", "docs/a.pdf")
        assert _resource_matches("DOCS/A.PDF", "docs/a.pdf")
        assert not _resource_matches("b.pdf", "docs/a.pdf")
        assert not _resource_matches("", "docs/a.pdf")

    def test_sound_split_is_complex(self):
        d = Decomposer()
        ta = d._parse_analysis(
            _response(
                _complex(
                    [
                        {"id": "s1", "objective": "A", "resources": ["docs/a.pdf"]},
                        {"id": "s2", "objective": "B", "resources": ["b.pdf", "c.pdf"]},
                    ]
                )
            ),
            resources=DOCS,
        )
        assert ta.is_complex
        assert ta.downgrade_reason == ""
        assert ta.gates == {"gate1": True, "gate2": True, "gate3": True, "gate4": True}
        assert ta.sub_tasks[1]["resources"] == ["b.pdf", "c.pdf"]

    def test_unowned_resource_downgrades(self):
        d = Decomposer()
        ta = d._parse_analysis(
            _response(
                _complex(
                    [
                        {"id": "s1", "objective": "A", "resources": ["docs/a.pdf"]},
                        {"id": "s2", "objective": "B", "resources": ["docs/b.pdf"]},
                    ]
                )
            ),
            resources=DOCS,
        )
        assert not ta.is_complex
        assert ta.sub_tasks == []
        assert "1 resource(s) unowned: docs/c.pdf" in ta.downgrade_reason

    def test_resourceless_sub_task_downgrades(self):
        d = Decomposer()
        ta = d._parse_analysis(
            _response(
                _complex(
                    [
                        {"id": "s1", "objective": "A", "resources": DOCS},
                        {"id": "s2", "objective": "B"},
                    ]
                )
            ),
            resources=DOCS,
        )
        assert not ta.is_complex
        assert "claims no resources" in ta.downgrade_reason

    def test_shared_source_downgrades(self):
        """The office shape: every child re-reads the same documents."""
        d = Decomposer()
        ta = d._parse_analysis(
            _response(
                _complex(
                    [
                        {"id": "s1", "objective": "Dates", "resources": DOCS},
                        {"id": "s2", "objective": "Amounts", "resources": DOCS},
                        {"id": "s3", "objective": "Parties", "resources": DOCS},
                    ]
                )
            ),
            resources=DOCS,
        )
        assert not ta.is_complex
        assert "shared by several sub-tasks" in ta.downgrade_reason
        assert "at most 1 owner(s)" in ta.downgrade_reason

    def test_owner_cap_is_configurable(self):
        d = Decomposer()
        payload = _complex(
            [
                {"id": "s1", "objective": "A", "resources": DOCS},
                {"id": "s2", "objective": "B", "resources": DOCS},
            ]
        )
        assert not d._parse_analysis(_response(payload), resources=DOCS).is_complex
        assert d._parse_analysis(
            _response(payload), resources=DOCS, max_owners=2
        ).is_complex

    def test_gate4_false_downgrades(self):
        d = Decomposer()
        ta = d._parse_analysis(
            _response(
                _complex(
                    [{"id": "s1", "objective": "A"}, {"id": "s2", "objective": "B"}],
                    gate4=False,
                )
            )
        )
        assert not ta.is_complex
        assert ta.downgrade_reason == "gates failed: gate4"

    def test_missing_gate4_is_tolerated_without_resources(self):
        d = Decomposer()
        ta = d._parse_analysis(
            _response(
                {
                    "gate1": True,
                    "gate2": True,
                    "gate3": True,
                    "complexity": "complex",
                    "sub_tasks": [
                        {"id": "s1", "objective": "A"},
                        {"id": "s2", "objective": "B"},
                    ],
                }
            )
        )
        assert ta.is_complex

    def test_no_contract_without_resources(self):
        d = Decomposer()
        ta = d._parse_analysis(
            _response(
                _complex(
                    [{"id": "s1", "objective": "A"}, {"id": "s2", "objective": "B"}]
                )
            )
        )
        assert ta.is_complex and ta.resources == []

    def test_malformed_sub_tasks_are_dropped(self):
        d = Decomposer()
        ta = d._parse_analysis(
            _response(
                _complex(["not a dict", {"id": "s1"}, {"id": "s2", "objective": "B"}])
            )
        )
        assert not ta.is_complex
        assert "only 1 well-formed" in ta.downgrade_reason

    def test_decision_summary_has_no_task_text(self):
        ta = TaskAnalysis(
            is_complex=False,
            reasoning="r",
            gates={"gate1": True, "gate2": False, "gate3": True, "gate4": True},
            downgrade_reason="gates failed: gate2",
            resources=DOCS,
        )
        d = ta.to_decision()
        assert d["gates"]["gate2"] is False
        assert d["downgrade_reason"] == "gates failed: gate2"
        assert d["resources"] == 3
        assert "docs/a.pdf" not in json.dumps(d)


# --------------------------------------------------------------------------- #
# Children receive attachments / context / resources slice                     #
# --------------------------------------------------------------------------- #


class TestChildTask:
    def test_child_gets_slice_and_parent_context(self):
        parent = Task(
            id="p",
            objective="Extract",
            context={"customer": "ACME", "resources": DOCS},
            metadata={"run": 1},
            attachments=[Attachment(type="image_url", data="https://x/y.png")],
        )
        child = Decomposer.build_child_task(
            parent, {"id": "s1", "objective": "A", "resources": ["docs/a.pdf"]}
        )
        assert child.id == "s1" and child.objective == "A"
        assert child.resources == ["docs/a.pdf"]
        assert child.context == {"customer": "ACME"}
        assert child.metadata == {"run": 1}
        assert child.attachments and child.attachments[0].data == "https://x/y.png"

    def test_child_without_slice_inherits_all_resources(self):
        parent = Task(id="p", objective="Extract", resources=DOCS)
        child = Decomposer.build_child_task(parent, {"id": "s1", "objective": "A"})
        assert child.resources == DOCS
        assert child.context is None

    def test_child_without_parent_task(self):
        child = Decomposer.build_child_task(None, {"id": "s1", "objective": "A"})
        assert child.resources is None and child.attachments is None

    @pytest.mark.asyncio
    async def test_run_sub_tasks_passes_child_task_and_reports_resources(self):
        seen: list[Task] = []

        async def fake_execute(task: Task):
            seen.append(task)
            return f"done {task.id}"

        sub_agent = MagicMock()
        sub_agent.execute = AsyncMock(side_effect=fake_execute)
        parent_task = Task(
            id="p", objective="Extract", resources=DOCS, context={"k": "v"}
        )
        with patch.object(Decomposer, "create_sub_agent", return_value=sub_agent):
            findings = await Decomposer().run_sub_tasks(
                MagicMock(),
                [
                    {"id": "s1", "objective": "A", "resources": ["docs/a.pdf"]},
                    {
                        "id": "s2",
                        "objective": "B",
                        "resources": ["docs/b.pdf", "c.pdf"],
                    },
                ],
                parent_task=parent_task,
            )
        assert [t.resources for t in seen] == [["docs/a.pdf"], ["docs/b.pdf", "c.pdf"]]
        assert all(t.context == {"k": "v"} for t in seen)
        assert findings[0]["resources"] == ["docs/a.pdf"]
        assert findings[1]["result"] == "done s2"

    @pytest.mark.asyncio
    async def test_run_sub_tasks_without_parent_task_is_unchanged(self):
        sub_agent = MagicMock()
        sub_agent.execute = AsyncMock(return_value="ok")
        with patch.object(Decomposer, "create_sub_agent", return_value=sub_agent):
            findings = await Decomposer().run_sub_tasks(
                MagicMock(), [{"id": "s1", "objective": "A"}]
            )
        assert findings == [{"id": "s1", "objective": "A", "result": "ok"}]


# --------------------------------------------------------------------------- #
# End to end: classification decision in the report                            #
# --------------------------------------------------------------------------- #


class TestClassificationDecision:
    @pytest.mark.asyncio
    async def test_shared_source_split_runs_simple_and_is_recorded(self):
        llm = ScriptedLLM(
            [
                json.dumps(
                    _complex(
                        [
                            {"id": "s1", "objective": "Dates", "resources": DOCS},
                            {"id": "s2", "objective": "Amounts", "resources": DOCS},
                        ]
                    )
                ),
                "All fields extracted",
                "VERDICT: PASS\nSCORE: 0.9\nFEEDBACK: ok",
            ]
        )
        agent = _agent(
            llm,
            AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS, max_retries=2),
        )
        result = await agent.execute(Task(id="t", objective="Extract", resources=DOCS))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == "All fields extracted"
        decision = result.diagnostics.decisions["classification"]
        assert decision["classifier_called"] is True
        assert decision["is_complex"] is False
        assert "shared by several sub-tasks" in decision["downgrade_reason"]
        assert decision["resources"] == 3
        assert decision["sub_task_resources"] == []
        kinds = [e.kind for e in result.diagnostics.timeline]
        assert "decomposition_downgraded" in kinds
        # No sub-agents were spawned: classifier + answer + critic only.
        assert llm.calls == 3
