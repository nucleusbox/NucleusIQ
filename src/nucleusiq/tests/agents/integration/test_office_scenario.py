"""The office production failure, replayed offline (PR H).

Scenario (docs/design/AUTONOMOUS_HARNESS_HARDENING.md §1): vLLM + Gemma-class
model, 65K shared window, Autonomous mode, nine documents to extract, 19
idempotent tools registered, ``response_format`` set.  In 0.7.13 this run
looped on ``[recall_error]`` / dedup banners after emergency compaction,
lost documents through an ungrounded COMPLEX split, and answered in prose.

The model here is a *behavioural* fake rather than a fixed script: it reads
whatever documents are still unread in its context (up to three per turn),
answers classifier / Critic prompts the way a cooperative model would, and
emits schema-valid JSON when asked for it.  It deliberately has **no memory
outside the transcript** — if the harness evicted the evidence that a
document was read, this model would read it again, exactly like a real one.

Every test asserts the incident invariants:

* the run terminates with ``termination_reason == completed``;
* at most one emergency compaction, zero ``[recall_error]``;
* all nine resources are touched (coverage complete, no follow-up needed);
* the output validates against ``response_format``;
* the analyzer reports no ``LOOP_*`` / critical finding.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.agent_result import ResultStatus
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.diagnostics import RunReport
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.streaming.events import StreamEventType
from nucleusiq.tests.conftest import make_test_prompt
from nucleusiq.tools import BaseTool
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Fixtures: documents, schema, tools                                           #
# --------------------------------------------------------------------------- #

DOCS = [f"office/invoice_{i:02d}.pdf" for i in range(1, 10)]
_DOC_RE = re.compile(r"office/invoice_\d{2}\.pdf")


def _doc_total(path: str) -> int:
    return 1_000 * int(path[-6:-4])


def _doc_text(path: str, *, repeat: int = 4) -> str:
    n = path[-6:-4]
    body = (
        f"Invoice {n} from Vendor {n} dated 2026-03-{int(n):02d}. "
        f"Total amount {_doc_total(path)} USD. Payment terms net 30. "
        "Line items: consulting services, travel expenses, software licences. "
        "Approved by finance and archived in the vendor folder. "
    )
    # repeat=4 → ~1.4K chars per document: nine fit comfortably in 65K
    # tokens.  The pressure test uses a much larger repeat so the nine
    # documents cannot all stay live in a 12K window.
    return f"[{path}]\n" + body * repeat


class DocRecord(BaseModel):
    path: str
    vendor: str
    total_usd: float


class Extraction(BaseModel):
    documents: list[DocRecord]
    grand_total_usd: float


class ReadDocumentTool(BaseTool):
    def __init__(self, log: list[str], *, repeat: int = 4) -> None:
        super().__init__(
            name="read_document",
            description="Read one document by path",
            idempotent=True,
        )
        self.log = log
        self.repeat = repeat

    async def initialize(self) -> None:
        pass

    async def execute(self, path: str) -> str:
        self.log.append(path)
        if path not in DOCS:
            return f"Error: {path} not found"
        return _doc_text(path, repeat=self.repeat)

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


class NoiseTool(BaseTool):
    """One of the 18 other idempotent tools the office agent registered."""

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            description=f"{name}: office workflow helper (unused in this run)",
            idempotent=True,
        )

    async def initialize(self) -> None:
        pass

    async def execute(self, **kwargs: Any) -> str:
        return f"{self.name}: ok"

    def get_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "free text"},
                    "limit": {"type": "integer", "description": "max rows"},
                },
            },
        }


NOISE_TOOL_NAMES = [
    "list_folder",
    "get_metadata",
    "search_index",
    "lookup_vendor",
    "get_po_number",
    "check_approval",
    "currency_convert",
    "get_tax_rate",
    "list_line_items",
    "get_payment_terms",
    "find_duplicates",
    "get_department",
    "get_cost_center",
    "get_contract",
    "verify_signature",
    "get_audit_trail",
    "get_exchange_rate",
    "get_fiscal_period",
]
assert len(NOISE_TOOL_NAMES) == 18


def office_tools(log: list[str], *, doc_repeat: int = 4) -> list[BaseTool]:
    return [
        ReadDocumentTool(log, repeat=doc_repeat),
        *(NoiseTool(n) for n in NOISE_TOOL_NAMES),
    ]


# --------------------------------------------------------------------------- #
# The behavioural fake model                                                   #
# --------------------------------------------------------------------------- #

PASS = json.dumps(
    {
        "verdict": "pass",
        "score": 0.92,
        "feedback": "complete and grounded",
        "issues": [],
        "suggestions": [],
        "verifier_answer": None,
    }
)


def _split(docs: list[str], groups: int) -> list[list[str]]:
    size = -(-len(docs) // groups)
    return [docs[i : i + size] for i in range(0, len(docs), size)]


class OfficeModel(MockLLM):
    """Deterministic stand-in for a cooperative 31B model on vLLM.

    ``classify`` controls the classifier answer: ``"simple"`` or
    ``"complex"`` (three sub-tasks of three documents each — the grounded
    split Gate 4 requires).
    """

    def __init__(
        self,
        *,
        classify: str = "simple",
        reads_per_turn: int = 3,
        max_reads_per_conversation: int | None = None,
        stubborn: bool = False,
        context_window: int = 65_536,
        critic: str = "lenient",
        split_groups: int = 3,
    ) -> None:
        super().__init__(model_name="gemma-office", context_window=context_window)
        self.classify = classify
        # How many sub-tasks a COMPLEX classification proposes.  ``9`` is the
        # gpt-4.1-mini behaviour observed live: one sub-task per invoice,
        # which ``max_sub_agents`` caps at five and leaves four documents
        # for the coverage follow-up child.
        self.split_groups = split_groups
        self.reads_per_turn = reads_per_turn
        # ``"lenient"`` always passes.  ``"literal"`` is the gpt-oss 120B
        # behaviour observed live: it counts the invoice headers it can
        # see in its own prompt and FAILs any answer with more records
        # than that — "the evidence only contains three invoices".
        self.critic = critic
        self.critic_prompts: list[str] = []
        # Stubborn: only believes a document was read when its full text
        # ("[path]" header) is live in a tool message — ignores its own
        # earlier tool calls, offload markers and dedup banners.  This is
        # the model behaviour that drove the 0.7.13 infinite loop.
        self.stubborn = stubborn
        # A lazy model: fetches at most this many documents per user
        # instruction, then answers with what it has.
        self.max_reads = max_reads_per_conversation
        self.requests: list[dict[str, Any]] = []
        self.kinds: list[str] = []

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _text(msg: dict[str, Any]) -> str:
        content = msg.get("content")
        if isinstance(content, list):
            return " ".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict)
            )
        return str(content or "")

    def _already_read(self, messages: list[dict[str, Any]]) -> set[str]:
        """Documents this transcript proves were fetched: my own earlier
        tool calls, and tool results / recall catalogs naming the file."""
        seen: set[str] = set()
        for m in messages:
            text = self._text(m)
            if self.stubborn:
                if m.get("role") == "tool":
                    seen.update(
                        d for d in _DOC_RE.findall(text) if f"[{d}]\nInvoice" in text
                    )
                continue
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                seen.update(_DOC_RE.findall(str(fn.get("arguments", ""))))
            if m.get("role") == "tool":
                seen.update(_DOC_RE.findall(text))
            else:
                # Document *content* (its "[path]" header) rehydrated into a
                # Critic / Refiner / synthesis package counts as evidence too,
                # as does an evidence catalog / receipt line naming the call's
                # arguments ("args=..." / "args: ...").
                seen.update(
                    _DOC_RE.findall(" ".join(re.findall(r"\[(office/[^\]]+)\]", text)))
                )
                for line in text.splitlines():
                    if "args=" in line or line.startswith("args:"):
                        seen.update(_DOC_RE.findall(line))
        return seen

    def _in_scope(self, messages: list[dict[str, Any]]) -> list[str]:
        scope: list[str] = []
        for m in messages:
            if m.get("role") in ("system", "user"):
                for d in _DOC_RE.findall(self._text(m)):
                    if d not in scope:
                        scope.append(d)
        return scope

    def _final_answer(self, messages: list[dict[str, Any]], *, want_json: bool) -> str:
        read = self._already_read(messages)
        docs = [d for d in self._in_scope(messages) if d in read] or sorted(read)
        if want_json:
            records = [
                {"path": d, "vendor": f"Vendor {d[-6:-4]}", "total_usd": _doc_total(d)}
                for d in docs
            ]
            return json.dumps(
                {
                    "documents": records,
                    "grand_total_usd": sum(r["total_usd"] for r in records),
                }
            )
        lines = [f"{d}: Vendor {d[-6:-4]}, total {_doc_total(d)} USD" for d in docs]
        return "Findings:\n" + "\n".join(lines)

    def _tool_calls(self, paths: list[str]) -> Any:
        return self.LLMResponse(
            [
                self.Choice(
                    self.Message(
                        content=None,
                        tool_calls=[
                            {
                                "id": f"call_{len(self.requests)}_{i}",
                                "type": "function",
                                "function": {
                                    "name": "read_document",
                                    "arguments": json.dumps({"path": p}),
                                },
                            }
                            for i, p in enumerate(paths)
                        ],
                    )
                )
            ]
        )

    def _reply(self, text: str) -> Any:
        return self.LLMResponse([self.Choice(self.Message(content=text))])

    # -- the model ----------------------------------------------------------

    async def call(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        self.requests.append({"messages": messages, **kwargs})
        blob = "\n".join(self._text(m) for m in messages)
        last_user = next(
            (self._text(m) for m in reversed(messages) if m.get("role") == "user"), ""
        )

        if "You are a task classifier" in last_user:
            self.kinds.append("classify")
            scope = self._in_scope(messages)
            if self.classify == "complex" and len(scope) >= 3:
                subs = [
                    {
                        "id": f"batch{i + 1}",
                        "objective": f"Extract vendor and total from: {', '.join(g)}",
                        "resources": g,
                    }
                    for i, g in enumerate(_split(scope, self.split_groups))
                ]
                return self._reply(
                    json.dumps(
                        {
                            "gate1": True,
                            "gate2": True,
                            "gate3": True,
                            "gate4": True,
                            "complexity": "complex",
                            "sub_tasks": subs,
                        }
                    )
                )
            return self._reply(
                json.dumps(
                    {
                        "gate1": True,
                        "gate2": True,
                        "gate3": True,
                        "gate4": False,
                        "complexity": "simple",
                    }
                )
            )

        if '"verdict": "pass" | "fail" | "uncertain"' in blob:
            self.kinds.append("critic")
            self.critic_prompts.append(blob)
            if self.critic == "literal":
                answer = blob.split("## ANSWER TO VERIFY", 1)[-1]
                claimed = set(_DOC_RE.findall(answer.split("## ASSESSMENT", 1)[0]))
                evidence = blob.split("## ANSWER TO VERIFY", 1)[0]
                seen = {d for d in _DOC_RE.findall(evidence) if f"[{d}]" in evidence}
                if claimed - seen:
                    return self._reply(
                        json.dumps(
                            {
                                "verdict": "fail",
                                "score": 0.1,
                                "feedback": (
                                    f"The answer lists {len(claimed)} invoices but the "
                                    f"evidence only contains {len(seen)}."
                                ),
                                "issues": [
                                    f"{d} is not supported by the evidence"
                                    for d in sorted(claimed - seen)
                                ],
                                "suggestions": ["Remove the unsupported records."],
                            }
                        )
                    )
            return self._reply(PASS)

        tools = kwargs.get("tools") or kwargs.get("functions")
        if tools:
            read = self._already_read(messages)
            pending = [d for d in self._in_scope(messages) if d not in read]
            budget = self.reads_per_turn
            if self.max_reads is not None:
                # Lazy: bounded effort per user instruction — a new user
                # message (e.g. the coverage retry) resets the budget.
                last_user_idx = max(
                    (i for i, m in enumerate(messages) if m.get("role") == "user"),
                    default=-1,
                )
                since = self._already_read(messages[last_user_idx + 1 :])
                budget = min(budget, self.max_reads - len(since))
            if pending and budget > 0:
                self.kinds.append("read")
                return self._tool_calls(pending[:budget])

        want_json = bool(kwargs.get("response_format")) or (
            "Extraction" in blob and "JSON" in blob
        )
        self.kinds.append("final_json" if want_json else "final_text")
        return self._reply(self._final_answer(messages, want_json=want_json))

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


class ProviderParsingOfficeModel(OfficeModel):
    """An adapter that parses structured replies itself — the real
    provider contract.

    Every first-party adapter receives ``response_format=<SchemaClass>`` in
    NATIVE mode, calls the API, then ``json.loads`` the reply into the
    class.  gpt-4.1-mini returned two JSON objects back to back
    (``Extra data: line 2 column 1``) and the adapter raised
    ``SchemaParseError`` *inside* ``llm.call`` — the run ended in ERROR
    for text the output contract knows how to repair.  With
    ``response_format=(wire_format, None)`` an adapter returns the raw
    reply, which is the retry the harness now makes.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.parse_failures = 0

    async def call(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        resp = await super().call(model=model, messages=messages, **kwargs)
        fmt = kwargs.get("response_format")
        if self.kinds and self.kinds[-1] == "final_json" and isinstance(fmt, type):
            from nucleusiq.agents.structured_output.errors import SchemaParseError

            self.parse_failures += 1
            raise SchemaParseError(
                "LLM response is not valid JSON: Extra data: line 2 column 1 (char 679)"
            )
        return resp


# --------------------------------------------------------------------------- #
# Agent factory + invariants                                                   #
# --------------------------------------------------------------------------- #

OBJECTIVE = (
    "Extract vendor name and total amount (USD) from every invoice listed "
    "under resources and return one record per document plus the grand total."
)


def office_agent(
    llm: MockLLM,
    log: list[str],
    *,
    window: int = 65_536,
    response_format: Any = Extraction,
    doc_repeat: int = 4,
    **config_overrides: Any,
) -> Agent:
    cfg: dict[str, Any] = {
        "execution_mode": ExecutionMode.AUTONOMOUS,
        "max_tool_calls": 40,
        "max_retries": 3,
        "llm_max_output_tokens": 2_048,
        "context": ContextConfig(max_context_tokens=window, response_reserve=4_096),
    }
    cfg.update(config_overrides)
    return Agent(
        name="OfficeExtractor",
        role="Accounts-payable analyst",
        objective="Extract invoice data for the finance team",
        prompt=make_test_prompt(),
        llm=llm,
        tools=office_tools(log, doc_repeat=doc_repeat),
        config=AgentConfig(**cfg),
        response_format=response_format,
    )


def office_task() -> Task:
    return Task(id="office-9", objective=OBJECTIVE, resources=list(DOCS))


def assert_incident_invariants(result: Any, log: list[str]) -> RunReport:
    report = result.diagnostics
    assert isinstance(report, RunReport), "run report must always be attached"

    # 1. Terminates, and says so.
    assert result.status == ResultStatus.SUCCESS, (result.error, report.explain())
    assert result.termination_reason == "completed", report.explain()
    assert report.termination.reason.value == "completed"

    # 2. No 0.7.13 pathology.
    c = report.counters
    assert c.emergency_count <= 1, report.explain()
    assert c.recall_errors == 0, report.explain()
    assert c.no_progress_rounds == 0, report.explain()
    assert c.tool_calls_business <= 2 * len(DOCS), report.explain()

    # 3. Every document was actually read (tool traffic, not the model's claim)
    #    and the harness knows it.
    assert set(log) >= set(DOCS), sorted(set(DOCS) - set(log))
    assert max(log.count(d) for d in DOCS) <= 2, "documents were re-read repeatedly"
    cov = result.metadata["coverage"]
    assert cov["complete"] is True and cov["unprocessed"] == [], cov
    assert sorted(cov["touched"]) == sorted(DOCS)
    assert cov["followup"] is None, "coverage should not have needed a follow-up"

    # 4. The analyzer agrees the run was healthy.
    codes = [f.code for f in report.findings]
    assert not any(code.startswith("LOOP_") for code in codes), codes
    assert not any(f.severity == "critical" for f in report.findings), codes
    assert "SYNTH_VS_STRUCTURED" not in codes
    assert "COVERAGE_GAP" not in codes and "DECOMP_COVERAGE_GAP" not in codes
    return report


def assert_structured(result: Any) -> Extraction:
    assert result.structured["valid"] is True, result.structured
    parsed = result.parsed
    assert isinstance(parsed, Extraction)
    assert sorted(d.path for d in parsed.documents) == sorted(DOCS)
    assert parsed.grand_total_usd == sum(_doc_total(d) for d in DOCS)
    return parsed


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


class TestOfficeScenario:
    @pytest.mark.asyncio
    async def test_simple_path_structured(self):
        log: list[str] = []
        llm = OfficeModel(classify="simple")
        agent = office_agent(llm, log)
        result = await agent.execute(office_task())

        report = assert_incident_invariants(result, log)
        assert_structured(result)
        assert report.decisions["classification"]["is_complex"] is False
        assert report.config_resolved["context_window"] == 65_536
        assert report.config_resolved["tool_count"] == 19
        assert report.config_resolved["idempotent_tool_count"] == 19
        assert report.config_resolved["response_format_set"] is True
        # Structured runs never take the prose synthesis detour.
        assert report.counters.synthesis_runs == 0
        assert "final_text" not in llm.kinds
        # Bounded work: one classification, ≤ 4 read rounds, answer, critic.
        assert report.counters.llm_calls <= 12, llm.kinds

    @pytest.mark.asyncio
    async def test_complex_path_structured(self):
        log: list[str] = []
        llm = OfficeModel(classify="complex")
        agent = office_agent(llm, log)
        result = await agent.execute(office_task())

        report = assert_incident_invariants(result, log)
        assert_structured(result)
        cls = report.decisions["classification"]
        assert cls["is_complex"] is True and cls["sub_tasks"] == 3

        # Children inherited the parent's limits (0.7.13 gave them 8K / 15).
        business = [c for c in report.children if c["id"].startswith("batch")]
        assert len(business) == 3
        for child in business:
            assert child["window"] == 65_536, child
            assert child["max_tool_calls"] == 40, child
            assert child["status"] == "success", child
            assert sorted(child["touched_resources"]) == sorted(child["resources"])
        # Each document was owned and read by exactly one child.
        touched = [d for c in business for d in c["touched_resources"]]
        assert sorted(touched) == sorted(DOCS)
        codes = [f.code for f in report.findings]
        assert "CHILD_WINDOW_MISMATCH" not in codes
        assert "CHILD_HIDDEN_CAP" not in codes
        assert "DECOMP_SHARED_SOURCE" not in codes
        assert "HANDOFF_TRUNCATED" not in codes
        assert report.counters.children_failed == 0
        # Children's evidence was merged into the parent's stores.
        assert any(e.kind == "evidence_merged" for e in report.timeline)
        # The synthesis hand-off cap is derived from the 65K window, not the
        # historical ``[:2000]`` literal, and nothing was truncated.
        handoff = report.decisions["synthesis_handoff"]
        assert handoff["per_finding_chars"] > 2_000, handoff
        assert handoff["truncated"] == []
        # I-10 in the COMPLEX path: the synthesizer's evidence is the
        # hand-off, not a tool trace — the Critic must be shown the same
        # findings, and a tool-less synthesis is then a *complete* view,
        # not a "partial" one (Gemma 4 live: CRITIC_PARTIAL_VIEW fired on
        # a Critic that had been given nothing at all).
        view = report.decisions["critic_views"][-1]
        assert view["inputs_included"] is True and view["inputs_items"] == 3, view
        assert view["complete"] is True, view
        assert report.counters.critic_partial_views == 0
        assert "CRITIC_PARTIAL_VIEW" not in codes
        critic_prompt = llm.critic_prompts[-1]
        assert "MATERIAL THE GENERATOR WAS GIVEN" in critic_prompt
        assert critic_prompt.count("### Sub-task:") == 3
        for doc in DOCS:
            assert doc in critic_prompt.split("## ANSWER TO VERIFY", 1)[0], doc

    @pytest.mark.asyncio
    async def test_simple_path_streaming(self):
        log: list[str] = []
        llm = OfficeModel(classify="simple")
        agent = office_agent(llm, log)
        events = [e async for e in agent.execute_stream(office_task())]

        types = [e.type for e in events]
        assert StreamEventType.ERROR not in types, [
            e.content for e in events if e.type == StreamEventType.ERROR
        ]
        completes = [e for e in events if e.type == StreamEventType.COMPLETE]
        assert completes
        final_text = completes[-1].content
        parsed = Extraction.model_validate_json(final_text)
        assert sorted(d.path for d in parsed.documents) == sorted(DOCS)

        report = agent.last_run_report
        assert isinstance(report, RunReport)
        assert report.termination.reason.value == "completed", report.explain()
        assert report.counters.emergency_count <= 1
        assert report.counters.recall_errors == 0
        assert set(log) >= set(DOCS)
        assert max(log.count(d) for d in DOCS) <= 2
        # Streaming reconciles coverage into the report as well.
        assert report.coverage is not None
        assert report.coverage["unprocessed"] == []
        assert sorted(report.coverage["touched"]) == sorted(DOCS)
        assert not any(f.code.startswith("LOOP_") for f in report.findings)

    @pytest.mark.asyncio
    async def test_display_and_explain_are_pasteable(self):
        log: list[str] = []
        agent = office_agent(OfficeModel(classify="complex"), log)
        result = await agent.execute(office_task())
        report = assert_incident_invariants(result, log)

        text = result.display()
        assert "Ended  : completed" in text
        explain = report.explain(redact=True)
        assert "termination **completed**" in explain
        assert "children: 3" in explain
        for d in DOCS:
            assert d not in explain, "redacted explain must not leak resource names"
        # A pasted report can be re-analyzed offline with the same verdict.
        from nucleusiq.agents.diagnostics import load_report

        again = load_report(json.dumps(result.summary(), default=str)).analyze()
        assert [f.code for f in again.findings] == [f.code for f in report.findings]


class TestOfficeUnderPressure:
    """Same job, hostile budgets: the run must still *terminate cleanly*."""

    @pytest.mark.asyncio
    async def test_small_window_terminates_without_loop(self):
        log: list[str] = []
        llm = OfficeModel(classify="simple")
        agent = office_agent(
            llm,
            log,
            window=12_000,
            # ~14K chars (~3.5K tokens) per document: nine of them are four
            # times the working budget, so offloading / compaction must run.
            doc_repeat=40,
            llm_max_output_tokens=1_024,
            context=ContextConfig(max_context_tokens=12_000, response_reserve=1_500),
            preflight_downgrade=False,
        )
        result = await agent.execute(office_task())
        report = result.diagnostics
        assert isinstance(report, RunReport)

        # Hard invariants regardless of outcome quality.
        assert result.status == ResultStatus.SUCCESS, (result.error, report.explain())
        assert report.termination.reason.value == "completed", report.explain()
        assert report.counters.emergency_count <= 3, report.explain()
        assert report.counters.recall_errors == 0, report.explain()
        assert report.counters.tool_calls_business <= 2 * len(DOCS)
        assert report.counters.llm_calls <= 12, llm.kinds
        # Emergency compaction evicted the model's own tool calls, but the
        # evidence catalog names each ref's arguments, so an honest model
        # never re-requested a document it had already fetched.
        assert report.counters.dedup_banners == 0, report.explain()
        assert max(log.count(d) for d in DOCS) == 1, log
        assert_structured(result)
        assert result.metadata["coverage"]["complete"] is True
        assert not any(f.code.startswith("LOOP_") for f in report.findings)
        # The window really was too small: preflight said so, and the
        # context engine had to offload or compact to keep the run alive.
        assert any(f.code == "PREFLIGHT_UNFIT" for f in report.findings)
        ct = result.context_telemetry
        assert ct is not None
        assert ct.artifacts_offloaded > 0 or ct.compaction_count > 0, ct
        assert ct.peak_utilization <= 1.0
        # And the report says the window was the problem, if anything was.
        if report.counters.emergency_count >= 2:
            assert any(f.code == "EMERGENCY_THRASH" for f in report.findings)

    @pytest.mark.asyncio
    async def test_stubborn_model_cannot_loop_forever(self):
        """The 0.7.13 trigger: results are evicted, the model does not
        trust markers or banners and keeps re-requesting the same files.
        The harness must still stop — bounded, explained, no crash.

        Offloading is disabled so the adaptive Tier-1 squeeze cannot keep
        the evidence live as recallable previews (with it on, even this
        model is satisfied and the run simply completes — see
        ``test_greedy_model_keeps_unseen_evidence``).  This is the regime
        where the harness has *nothing* left but the loop guards.
        """
        log: list[str] = []
        llm = OfficeModel(classify="simple", stubborn=True)
        agent = office_agent(
            llm,
            log,
            window=12_000,
            doc_repeat=40,
            max_retries=2,
            llm_max_output_tokens=1_024,
            context=ContextConfig(
                max_context_tokens=12_000,
                response_reserve=1_500,
                enable_offloading=False,
            ),
            preflight_downgrade=False,
        )
        result = await agent.execute(office_task())
        report = result.diagnostics
        assert isinstance(report, RunReport)

        assert result.status != ResultStatus.ERROR, result.error
        assert report.termination.reason.value != "error", report.explain()
        # Bounded by the dedup / no-progress guards, not by max_tool_calls.
        assert report.counters.tool_calls_business < 40, report.explain()
        assert report.counters.llm_calls <= 30, llm.kinds
        assert report.counters.recall_errors == 0, report.explain()
        # The emergency circuit breaker (3) was never needed.
        assert report.counters.emergency_count <= 3, report.explain()
        # Every duplicate request was short-circuited — the tool itself ran
        # at most once per document.
        assert max(log.count(d) for d in DOCS) == 1, log
        assert report.counters.dedup_banners >= 3
        # The task head (objective + resource list) survived compaction:
        # every request still carried all nine resource names.
        for r in llm.requests:
            if not (r.get("tools") or r.get("functions")):
                continue  # classifier / Critic / finalizer prompts
            head = " ".join(
                str(m.get("content"))
                for m in r["messages"]
                if m.get("role") in ("system", "user")
            )
            assert all(d in head for d in DOCS), "task head evicted by compaction"
        codes = [f.code for f in report.findings]
        assert "LOOP_RECALL_DEADLOCK" not in codes, codes
        assert "LOOP_NO_PROGRESS" in codes, (codes, report.explain())
        assert "COVERAGE_GAP" in codes, codes
        # The structured finalizer still produced schema-valid output from
        # what was in context.
        assert result.structured["valid"] is True, result.structured
        assert any(
            "Do NOT re-fetch" in str(m.get("content"))
            for r in llm.requests
            for m in r["messages"]
        )

    @pytest.mark.asyncio
    async def test_critic_sees_all_evidence_when_provider_underreports_window(self):
        """Provider says 8K, the user configured 65K: the Critic's evidence
        package must be sized from the configured window.  Live regression
        (gpt-oss:120b on Ollama): sized from 8K, the Critic saw three of
        nine invoices, failed a correct answer and the Refiner deleted six
        records."""
        log: list[str] = []
        llm = OfficeModel(classify="simple", context_window=8_192)
        agent = office_agent(llm, log, window=65_536)
        result = await agent.execute(office_task())
        assert_incident_invariants(result, log)
        assert_structured(result)

        critic_requests = [
            r
            for r in llm.requests
            if any(
                '"verdict": "pass" | "fail" | "uncertain"' in str(m.get("content"))
                for m in r["messages"]
            )
        ]
        assert critic_requests, "the Critic must have run"
        for r in critic_requests:
            blob = " ".join(str(m.get("content")) for m in r["messages"])
            missing = [d for d in DOCS if f"[{d}]" not in blob]
            assert not missing, f"Critic evidence truncated; missing {missing}"

    @pytest.mark.asyncio
    async def test_literal_critic_cannot_delete_correct_records(self):
        """The live gpt-oss failure, replayed with the framework fixed.

        A Critic that counts the invoice headers in its own prompt used to
        see three (a fixed 2.5K notes budget), fail the nine-record answer
        as ungrounded, and the Refiner — fed the same blind package — cut
        it to three.  The package is now budget-sized and complete, the
        raw trace rides along, and the coverage facts say 9/9 were read:
        the same Critic sees all nine headers and passes first time."""
        log: list[str] = []
        llm = OfficeModel(classify="simple", critic="literal", context_window=8_192)
        agent = office_agent(llm, log)
        result = await agent.execute(office_task())

        report = assert_incident_invariants(result, log)
        assert_structured(result)
        assert report.counters.critic_verdicts == {"pass": 1}, report.explain()
        assert report.counters.critic_partial_views == 0, report.explain()
        assert report.counters.critic_fail_downgraded == 0
        assert "final_json" in llm.kinds and llm.kinds.count("critic") == 1
        view = report.decisions["critic_views"][-1]
        assert view["complete"] is True and view["package_complete"] is True
        pkg = report.decisions["package:critic_evidence_total"]
        assert pkg["complete"] is True and pkg["max_chars"] > 12_000, pkg

    @pytest.mark.asyncio
    async def test_literal_critic_under_pressure_is_partial_not_wrong(self):
        """Same literal Critic, 12K window, 9K-char documents.

        Here the window genuinely cannot show every result whole.  The
        invariant is not "the Critic sees everything" (impossible) but
        "the Critic knows it sees less and cannot condemn on that basis":
        the view is recorded as partial, any FAIL is downgraded, the run
        report says so, and the nine records survive."""
        log: list[str] = []
        llm = OfficeModel(classify="simple", critic="literal", reads_per_turn=3)
        agent = office_agent(
            llm,
            log,
            window=12_000,
            doc_repeat=40,
            llm_max_output_tokens=1_024,
            context=ContextConfig(max_context_tokens=12_000, response_reserve=1_500),
            preflight_downgrade=False,
        )
        result = await agent.execute(office_task())
        report = result.diagnostics
        assert isinstance(report, RunReport)

        assert result.termination_reason == "completed", report.explain()
        assert report.counters.unseen_evidence_evicted == 0, report.explain()
        parsed = result.parsed
        assert isinstance(parsed, Extraction), result.structured
        assert sorted(d.path for d in parsed.documents) == sorted(DOCS)
        assert parsed.grand_total_usd == sum(_doc_total(d) for d in DOCS)
        # The Critic saw the heads of all nine results and the 9/9 coverage
        # line, so even the literal Critic has nothing to object to …
        assert "fail" not in report.counters.critic_verdicts, report.explain()
        # … and every Critic prompt carried the harness-verified coverage.
        assert llm.critic_prompts
        assert all(
            "9 of 9 declared resources were read" in p for p in llm.critic_prompts
        )
        codes = [f.code for f in report.findings]
        assert not any(c.startswith("LOOP_") for c in codes), codes
        assert "EVIDENCE_EVICTED_UNSEEN" not in codes, codes

    @pytest.mark.asyncio
    async def test_greedy_model_keeps_unseen_evidence(self):
        """A model that fetches all nine documents in one turn returns
        ~4x the working budget before it has read any of it.

        Before the adaptive Tier-1 squeeze this went straight to emergency
        eviction: the whole round vanished before the model saw it, and a
        real Gemma-4 run then invented vendors and totals for all nine
        files.  Now the results are offloaded largest-first as recallable
        previews, no emergency runs, nothing unseen is evicted, and the
        answer is grounded."""
        log: list[str] = []
        llm = OfficeModel(classify="simple", reads_per_turn=len(DOCS))
        agent = office_agent(
            llm,
            log,
            window=12_000,
            doc_repeat=40,
            llm_max_output_tokens=1_024,
            context=ContextConfig(max_context_tokens=12_000, response_reserve=1_500),
            preflight_downgrade=False,
        )
        result = await agent.execute(office_task())
        report = result.diagnostics
        assert isinstance(report, RunReport)

        assert result.status == ResultStatus.SUCCESS, (result.error, report.explain())
        assert report.termination.reason.value == "completed", report.explain()
        assert report.counters.emergency_count == 0, report.explain()
        assert report.counters.unseen_evidence_evicted == 0, report.explain()
        assert report.counters.dedup_banners == 0
        assert [log.count(d) for d in DOCS] == [1] * len(DOCS), log
        ct = result.context_telemetry
        assert ct is not None and ct.artifacts_offloaded >= 1, ct
        assert "tool_result_compactor" in report.counters.compactions_by_strategy
        assert "emergency_compactor" not in report.counters.compactions_by_strategy
        # The squeezed receipts told the model how to proceed.
        assert any(
            "offloaded to fit the context window before you saw it"
            in str(m.get("content"))
            for r in llm.requests
            for m in r["messages"]
        )
        assert_structured(result)
        assert result.metadata["coverage"]["complete"] is True
        codes = [f.code for f in report.findings]
        assert "EVIDENCE_EVICTED_UNSEEN" not in codes, codes
        assert not any(c.startswith("LOOP_") for c in codes), codes

    @pytest.mark.asyncio
    async def test_unseen_eviction_is_never_silent(self):
        """When the harness genuinely cannot keep a round's evidence
        (offloading off), the eviction of unread results is counted and
        reported as a critical finding instead of passing as a clean run."""
        log: list[str] = []
        llm = OfficeModel(classify="simple", reads_per_turn=len(DOCS))
        agent = office_agent(
            llm,
            log,
            window=12_000,
            doc_repeat=40,
            llm_max_output_tokens=1_024,
            context=ContextConfig(
                max_context_tokens=12_000,
                response_reserve=1_500,
                enable_offloading=False,
            ),
            preflight_downgrade=False,
        )
        result = await agent.execute(office_task())
        report = result.diagnostics
        assert isinstance(report, RunReport)
        assert report.termination.reason.value != "error", report.explain()
        assert report.counters.emergency_count >= 1, report.explain()
        assert report.counters.unseen_evidence_evicted >= 1, report.explain()
        finding = next(
            f for f in report.findings if f.code == "EVIDENCE_EVICTED_UNSEEN"
        )
        assert finding.severity == "critical"
        assert finding.evidence["counters.unseen_evidence_evicted"] >= 1
        assert "unseen tool results evicted" in report.explain()
        assert "EVIDENCE_EVICTED_UNSEEN" in result.display()

    @pytest.mark.asyncio
    async def test_lazy_model_gap_is_closed_by_the_one_followup(self):
        """A model that stops after five documents gets exactly one retry
        naming the other four — and the run still ends complete."""
        log: list[str] = []
        llm = OfficeModel(classify="simple", max_reads_per_conversation=5)
        agent = office_agent(llm, log)
        result = await agent.execute(office_task())

        assert result.status == ResultStatus.SUCCESS, result.error
        assert_structured(result)
        report = result.diagnostics
        cov = result.metadata["coverage"]
        assert cov["complete"] is True, cov
        assert cov["followup"] == {
            "kind": "retry",
            "ran": True,
            "before": cov["followup"]["before"],
            "pending": cov["followup"]["pending"],
        }
        assert len(cov["followup"]["pending"]) == 4
        assert set(log) == set(DOCS)
        assert max(log.count(d) for d in DOCS) == 1, "no document was re-read"
        assert any(e.kind == "coverage_retry" for e in report.timeline)
        codes = [f.code for f in report.findings]
        assert "COVERAGE_GAP" not in codes and not any(
            c.startswith("LOOP_") for c in codes
        )

    @pytest.mark.asyncio
    async def test_lazy_model_without_followup_reports_the_gap(self):
        log: list[str] = []
        llm = OfficeModel(classify="simple", max_reads_per_conversation=5)
        agent = office_agent(llm, log, coverage_followup=False)
        result = await agent.execute(office_task())
        report = result.diagnostics
        assert isinstance(report, RunReport)

        assert result.status == ResultStatus.SUCCESS, result.error
        assert report.counters.llm_calls <= 12, llm.kinds
        cov = result.metadata["coverage"]
        assert cov["complete"] is False and cov["followup"] is None
        assert len(cov["unprocessed"]) == 4
        codes = [f.code for f in report.findings]
        assert "COVERAGE_GAP" in codes, codes
        assert not any(c.startswith("LOOP_") for c in codes), codes
        gap = next(f for f in report.findings if f.code == "COVERAGE_GAP")
        assert gap.severity == "warning"
        assert sorted(gap.evidence["coverage.unprocessed"]) == sorted(
            cov["unprocessed"]
        )
        assert gap.fix_now
        assert "COVERAGE_GAP" in result.display()

    @pytest.mark.asyncio
    async def test_enforced_gate_abstains_on_residual_gap(self):
        log: list[str] = []
        llm = OfficeModel(classify="simple", max_reads_per_conversation=5)
        agent = office_agent(
            llm, log, coverage_followup=False, evidence_gate_enforce=True
        )
        result = await agent.execute(office_task())
        assert result.status == ResultStatus.ABSTAINED
        assert result.abstention_code == "coverage_incomplete"
        assert result.output, "output is kept for inspection"
        assert "invoice_06" in (result.abstention_reason or "")
        report = result.diagnostics
        gap = next(f for f in report.findings if f.code == "COVERAGE_GAP")
        assert gap.severity == "critical"
        assert "withheld" in gap.title

    @pytest.mark.asyncio
    async def test_followup_child_survives_a_wide_tool_belt(self):
        """gpt-4.1-mini live: nine one-document sub-tasks, capped at five
        children, four documents left for the coverage follow-up child.
        That child inherited the parent's 19 business tools with an 8-call
        budget and died in preflight ("has 19 tools but STANDARD mode allows
        max 8") before its first LLM call — the run ended with
        ``CHILDREN_FAILED`` and an empty finding that did not even say why."""
        log: list[str] = []
        llm = OfficeModel(classify="complex", split_groups=9)
        agent = office_agent(llm, log)
        result = await agent.execute(office_task())

        assert result.status == ResultStatus.SUCCESS, result.error
        assert result.termination_reason == "completed"
        assert_structured(result)
        report = result.diagnostics
        assert report.decisions["classification"]["sub_tasks"] == 9
        assert report.counters.children_spawned == 6, report.children
        assert report.counters.children_failed == 0, report.explain()

        followup = next(c for c in report.children if c["id"] == "coverage-followup")
        assert followup["status"] == "success", followup
        assert followup["termination_reason"] == "completed"
        assert followup["max_tool_calls"] >= 19, followup
        assert len(followup["resources"]) == 4
        assert sorted(followup["touched_resources"]) == sorted(followup["resources"])
        assert "error" not in followup

        cov = result.metadata["coverage"]
        assert cov["complete"] is True, cov
        assert cov["followup"]["kind"] == "child"
        assert cov["followup"]["status"] == "success"
        assert cov["followup"]["after"] == []
        assert set(log) == set(DOCS)
        codes = [f.code for f in report.findings]
        assert "CHILDREN_FAILED" not in codes and "COVERAGE_GAP" not in codes, codes

    @pytest.mark.asyncio
    async def test_provider_parse_failure_is_recovered_not_fatal(self):
        """An adapter that cannot parse the structured reply raises inside
        ``llm.call``.  The harness retries once with the wire format kept
        and the parse disabled, and the output contract validates the raw
        text — the run completes with a valid ``Extraction``."""
        log: list[str] = []
        llm = ProviderParsingOfficeModel(classify="simple")
        agent = office_agent(llm, log)
        result = await agent.execute(office_task())

        report = assert_incident_invariants(result, log)
        assert_structured(result)
        assert llm.parse_failures >= 1
        fallbacks = [
            e for e in report.timeline if e.kind == "structured_parse_fallback"
        ]
        assert len(fallbacks) == llm.parse_failures
        assert "SchemaParseError" in fallbacks[0].detail
        retried = [
            r for r in llm.requests if isinstance(r.get("response_format"), tuple)
        ]
        assert retried, "the retry must reach the adapter"
        for r in retried:
            wire, schema_type = r["response_format"]
            assert isinstance(wire, dict) and schema_type is None
        codes = [f.code for f in report.findings]
        assert not any(c.startswith("LOOP_") for c in codes), codes
