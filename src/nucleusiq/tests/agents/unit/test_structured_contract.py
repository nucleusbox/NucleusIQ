"""PR B — the structured-output contract (WS-8).

When a user sets ``Agent.response_format`` the schema is the deliverable:

* prose that comes back is repaired once by a tools-free finalizer;
* a loop stopped by a guard/budget runs the finalizer, not prose synthesis;
* the Autonomous validation pipeline rejects non-conforming candidates
  with the validator's exact errors so the retry can fix them;
* Critic / Refiner prompts carry the schema criterion;
* ``AgentResult.parsed`` / ``structured`` expose the typed verdict while
  ``result.output`` stays the raw text (``str(result)`` unchanged);
* a final answer that still fails records ``termination_reason=
  "schema_invalid"``;
* sub-agents never inherit ``response_format``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.agent_result import ResultStatus
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import Critic, CritiqueResult, Verdict
from nucleusiq.agents.components.decomposer import Decomposer
from nucleusiq.agents.components.refiner import Refiner
from nucleusiq.agents.components.validation import ValidationPipeline
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.structured_output.contract import (
    SchemaCheck,
    StructuredOutputContract,
)
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.streaming.events import StreamEventType
from nucleusiq.tests.conftest import make_test_prompt
from nucleusiq.tools import BaseTool
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


class Extraction(BaseModel):
    title: str
    amount: float
    tags: list[str] = []


VALID_JSON = json.dumps({"title": "Invoice 42", "amount": 12.5, "tags": ["paid"]})
PROSE = "Here is a summary of the invoice: it is titled Invoice 42 and costs 12.5."


class ReadDocTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="read_doc", description="Read a doc", idempotent=True)

    async def initialize(self) -> None:
        pass

    async def execute(self, path: str) -> str:
        return f"Invoice 42 total 12.5 (paid) from {path}"

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


class ScriptedLLM(MockLLM):
    """Replays scripted responses and records every request's kwargs."""

    def __init__(self, script: list[tuple[str, Any]]):
        super().__init__(model_name="scripted")
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

    async def call_stream(
        self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ):
        from nucleusiq.streaming.events import StreamEvent

        resp = await self.call(model=model, messages=messages, **kwargs)
        msg = resp.choices[0].message
        if msg.tool_calls:
            yield StreamEvent.complete_event(
                "", metadata={"tool_calls": msg.tool_calls}
            )
            return
        yield StreamEvent.token_event(msg.content)
        yield StreamEvent.complete_event(msg.content)


def _agent(
    llm: MockLLM, config: AgentConfig, *, tools=None, response_format=Extraction
):
    return Agent(
        name="Extractor",
        role="Analyst",
        objective="Extract invoice data",
        prompt=make_test_prompt(),
        llm=llm,
        tools=tools or [],
        config=config,
        response_format=response_format,
    )


def _std(**overrides: Any) -> AgentConfig:
    base = {"execution_mode": ExecutionMode.STANDARD, "max_tool_calls": 5}
    base.update(overrides)
    return AgentConfig(**base)


def _contract(agent: Agent) -> StructuredOutputContract:
    contract = BaseExecutionMode.structured_contract(agent)
    assert contract is not None
    return contract


# --------------------------------------------------------------------------- #
# Contract — pure                                                              #
# --------------------------------------------------------------------------- #


class TestContractCheck:
    def test_valid_json_yields_typed_value(self):
        agent = _agent(ScriptedLLM([("text", VALID_JSON)]), _std())
        check = _contract(agent).check(VALID_JSON)
        assert check.valid
        assert isinstance(check.value, Extraction)
        assert check.value.amount == 12.5
        assert json.loads(check.canonical_json)["title"] == "Invoice 42"

    def test_fenced_json_is_accepted(self):
        agent = _agent(ScriptedLLM([("text", VALID_JSON)]), _std())
        check = _contract(agent).check(f"```json\n{VALID_JSON}\n```")
        assert check.valid

    def test_prose_is_rejected_with_errors(self):
        agent = _agent(ScriptedLLM([("text", VALID_JSON)]), _std())
        check = _contract(agent).check(PROSE)
        assert not check.valid
        assert check.errors
        assert check.canonical_json is None

    def test_missing_required_field_reports_field(self):
        agent = _agent(ScriptedLLM([("text", VALID_JSON)]), _std())
        check = _contract(agent).check(json.dumps({"title": "x"}))
        assert not check.valid
        assert "amount" in check.errors

    @pytest.mark.parametrize("candidate", [None, "", "   ", "Error: tool budget"])
    def test_empty_and_error_strings_are_invalid(self, candidate):
        agent = _agent(ScriptedLLM([("text", VALID_JSON)]), _std())
        assert not _contract(agent).check(candidate).valid

    def test_typed_instance_is_accepted(self):
        agent = _agent(ScriptedLLM([("text", VALID_JSON)]), _std())
        inst = Extraction(title="t", amount=1.0)
        check = _contract(agent).check(inst)
        assert check.valid and check.value is inst

    def test_prompt_fragments_mention_schema(self):
        agent = _agent(ScriptedLLM([("text", VALID_JSON)]), _std())
        c = _contract(agent)
        for text in (
            c.critic_criterion(),
            c.refiner_instruction(),
            c.finalizer_instruction(),
            c.finalizer_instruction("boom"),
            c.retry_message(SchemaCheck(valid=False, errors="bad")),
        ):
            assert "Extraction" in text
            assert "JSON" in text
        assert "boom" in c.finalizer_instruction("boom")

    def test_no_contract_without_response_format(self):
        agent = _agent(ScriptedLLM([("text", "x")]), _std(), response_format=None)
        assert BaseExecutionMode.structured_contract(agent) is None


# --------------------------------------------------------------------------- #
# Standard mode — enforcement                                                  #
# --------------------------------------------------------------------------- #


class TestStandardModeEnforcement:
    @pytest.mark.asyncio
    async def test_json_answer_passes_through_untouched(self):
        llm = ScriptedLLM(
            [("tool", [("read_doc", {"path": "a"})]), ("text", VALID_JSON)]
        )
        agent = _agent(llm, _std(), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Extract"))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == VALID_JSON
        assert str(result) == VALID_JSON
        assert isinstance(result.parsed, Extraction)
        assert result.structured["valid"] is True
        assert result.structured["finalizer_runs"] == 0
        assert result.schema_valid is True
        assert result.termination_reason == "completed"
        assert llm.calls == 2  # no synthesis, no finalizer

    @pytest.mark.asyncio
    async def test_prose_is_repaired_by_finalizer_without_tools(self):
        llm = ScriptedLLM(
            [
                ("tool", [("read_doc", {"path": "a"})]),
                ("text", PROSE),
                ("text", VALID_JSON),
            ]
        )
        agent = _agent(llm, _std(), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Extract"))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == VALID_JSON
        assert isinstance(result.parsed, Extraction)
        assert result.structured["finalizer_runs"] == 1
        assert result.termination_reason == "completed"
        assert result.metadata["raw_output"] == VALID_JSON
        # Finalizer request is tools-free so native response_format applies.
        finalizer_req = llm.requests[-1]
        assert not finalizer_req.get("tools")
        last_user = [m for m in finalizer_req["messages"] if m["role"] == "user"][-1]
        assert "Extraction" in last_user["content"]

    @pytest.mark.asyncio
    async def test_unrepairable_output_is_schema_invalid_not_error(self):
        llm = ScriptedLLM([("tool", [("read_doc", {"path": "a"})]), ("text", PROSE)])
        agent = _agent(llm, _std(), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Extract"))

        assert result.status == ResultStatus.SUCCESS  # no crash, raw text kept
        assert result.output == PROSE
        assert result.parsed is None
        assert result.structured["valid"] is False
        assert result.structured["errors"]
        assert result.schema_valid is False
        assert result.termination_reason == "schema_invalid"
        assert result.diagnostics.counters.schema_validation_failures >= 1
        assert result.diagnostics.counters.finalizer_runs == 1

    @pytest.mark.asyncio
    async def test_tool_budget_stop_runs_structured_finalizer(self):
        llm = ScriptedLLM(
            [
                ("tool", [("read_doc", {"path": "a"})]),
                ("tool", [("read_doc", {"path": "b"})]),
                ("text", VALID_JSON),
            ]
        )
        agent = _agent(llm, _std(max_tool_calls=2), tools=[ReadDocTool()])
        result = await agent.execute(Task(id="t", objective="Extract"))

        assert result.status == ResultStatus.SUCCESS
        assert result.termination_reason == "tool_budget"
        assert result.output == VALID_JSON
        assert isinstance(result.parsed, Extraction)
        assert not llm.requests[-1].get("tools")
        assert result.diagnostics.counters.finalizer_runs == 1

    @pytest.mark.asyncio
    async def test_no_tools_prose_then_json(self):
        llm = ScriptedLLM([("text", PROSE), ("text", VALID_JSON)])
        agent = _agent(llm, _std())
        result = await agent.execute(Task(id="t", objective="Extract"))
        assert result.output == VALID_JSON
        assert isinstance(result.parsed, Extraction)

    @pytest.mark.asyncio
    async def test_prose_agent_unchanged(self):
        """No response_format → no schema fields, synthesis path as before."""
        llm = ScriptedLLM([("tool", [("read_doc", {"path": "a"})]), ("text", "Done.")])
        agent = _agent(llm, _std(), tools=[ReadDocTool()], response_format=None)
        result = await agent.execute(Task(id="t", objective="Extract"))
        assert result.parsed is None
        assert result.structured is None
        assert result.schema_valid is None
        assert "raw_output" not in result.metadata


class TestStreamingEnforcement:
    @pytest.mark.asyncio
    async def test_stream_complete_event_carries_repaired_json(self):
        llm = ScriptedLLM(
            [
                ("tool", [("read_doc", {"path": "a"})]),
                ("text", PROSE),
                ("text", VALID_JSON),
            ]
        )
        agent = _agent(llm, _std(), tools=[ReadDocTool()])
        events = [
            e async for e in agent.execute_stream(Task(id="t", objective="Extract"))
        ]

        complete = [e for e in events if e.type == StreamEventType.COMPLETE]
        assert complete and complete[-1].content == VALID_JSON
        thinking = [e for e in events if e.type == StreamEventType.THINKING]
        assert any("finalizer" in (e.message or "") for e in thinking)
        assert not llm.requests[-1].get("tools")


# --------------------------------------------------------------------------- #
# Autonomous pipeline — schema validation layer                                #
# --------------------------------------------------------------------------- #


class TestValidationLayer:
    @pytest.mark.asyncio
    async def test_prose_fails_schema_layer_with_retry_details(self):
        agent = _agent(ScriptedLLM([("text", PROSE)]), _std())
        vr = await ValidationPipeline().validate(agent, PROSE, [])
        assert not vr.valid
        assert vr.layer == "schema"
        assert "Extraction" in vr.reason
        assert vr.details and "ONLY a single JSON object" in vr.details[0]

    @pytest.mark.asyncio
    async def test_json_passes_schema_layer(self):
        agent = _agent(ScriptedLLM([("text", VALID_JSON)]), _std())
        vr = await ValidationPipeline().validate(agent, VALID_JSON, [])
        assert vr.valid

    @pytest.mark.asyncio
    async def test_no_schema_layer_without_response_format(self):
        agent = _agent(ScriptedLLM([("text", PROSE)]), _std(), response_format=None)
        vr = await ValidationPipeline().validate(agent, PROSE, [])
        assert vr.valid

    @pytest.mark.asyncio
    async def test_autonomous_finalizer_repairs_before_validation(self):
        """Attempt 1 is prose → finalizer repairs → validation + Critic pass."""
        llm = ScriptedLLM(
            [
                ("text", PROSE),
                ("text", VALID_JSON),
                ("text", "VERDICT: PASS\nSCORE: 0.95\nFEEDBACK: fine"),
            ]
        )
        agent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_tool_calls=3,
                max_retries=2,
                enable_decomposition=False,
            ),
        )
        result = await agent.execute(Task(id="t", objective="Extract"))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == VALID_JSON
        assert isinstance(result.parsed, Extraction)
        assert result.diagnostics.counters.finalizer_runs == 1
        assert result.termination_reason == "completed"

    @pytest.mark.asyncio
    async def test_autonomous_schema_layer_drives_retry(self):
        """Finalizer cannot repair → schema layer FAIL → retry carries errors."""
        llm = ScriptedLLM(
            [
                ("text", PROSE),  # attempt 1 answer
                ("text", PROSE),  # attempt 1 finalizer — still prose
                ("text", VALID_JSON),  # attempt 2 answer
                ("text", "VERDICT: PASS\nSCORE: 0.95\nFEEDBACK: fine"),
            ]
        )
        agent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_tool_calls=3,
                max_retries=2,
                enable_decomposition=False,
            ),
        )
        result = await agent.execute(Task(id="t", objective="Extract"))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == VALID_JSON
        assert isinstance(result.parsed, Extraction)
        assert result.termination_reason == "completed"
        # The retry message carried the contract to the model.
        retry_prompts = [
            m["content"]
            for req in llm.requests
            for m in req["messages"]
            if m["role"] == "user"
            and "did not satisfy the required output schema"
            in str(m.get("content", ""))
        ]
        assert retry_prompts
        assert "Extraction" in retry_prompts[0]


# --------------------------------------------------------------------------- #
# Critic / Refiner prompts                                                     #
# --------------------------------------------------------------------------- #


class TestCriticRefinerPrompts:
    def test_critic_prompt_includes_contract(self):
        prompt = Critic().build_verification_prompt(
            task_objective="Extract",
            final_result=PROSE,
            generator_messages=[ChatMessage(role="user", content="Extract")],
            output_contract="**Schema Compliance (REQUIRED)** Extraction",
        )
        assert "## OUTPUT CONTRACT" in prompt
        assert "Schema Compliance" in prompt
        # Placed right before the verdict format block so it is the last
        # instruction the Verifier reads.
        assert prompt.index("## OUTPUT CONTRACT") < prompt.index('"verifier_answer"')

    def test_critic_prompt_without_contract_unchanged(self):
        prompt = Critic().build_verification_prompt(
            task_objective="Extract",
            final_result=PROSE,
            generator_messages=[ChatMessage(role="user", content="Extract")],
        )
        assert "OUTPUT CONTRACT" not in prompt

    def test_refiner_prompt_includes_contract(self):
        critique = CritiqueResult(
            verdict=Verdict.FAIL, score=0.2, feedback="prose", issues=["not JSON"]
        )
        prompt = Refiner._build_revision_prompt(
            task_objective="Extract",
            candidate=PROSE,
            critique=critique,
            tool_result_summary=None,
            output_contract="OUTPUT CONTRACT: Extraction",
        )
        assert "## Output Contract" in prompt
        assert "satisfies the Output Contract" in prompt

    def test_refiner_prompt_without_contract_unchanged(self):
        critique = CritiqueResult(verdict=Verdict.FAIL, score=0.2, feedback="x")
        prompt = Refiner._build_revision_prompt(
            task_objective="Extract",
            candidate=PROSE,
            critique=critique,
            tool_result_summary=None,
        )
        assert "Output Contract" not in prompt


# --------------------------------------------------------------------------- #
# Sub-agents                                                                   #
# --------------------------------------------------------------------------- #


class TestSubAgents:
    @pytest.mark.asyncio
    async def test_children_do_not_inherit_response_format(self):
        agent = _agent(
            ScriptedLLM([("text", VALID_JSON)]),
            AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS, max_tool_calls=3),
        )
        child = await Decomposer.create_sub_agent(agent, "s1", "Read doc a")
        assert child is not None
        assert child.response_format is None
        assert BaseExecutionMode.structured_contract(child) is None
