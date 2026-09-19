"""Autonomous harness hardening — PR E: shared evidence between parent and children.

Covers (docs/design/AUTONOMOUS_HARNESS_HARDENING.md WS-3 / I-5 / I-6):

* layered store / corpus / dossier — reads fall through to the parent,
  writes stay local, the parent's telemetry is untouched;
* ``merge_child_evidence`` — child-local entries land in the parent once,
  idempotently, with a ``MergeReport``;
* ``ParentEvidenceView`` — system hint only when there is something to find;
* ``ResourceTouchTracker`` — which declared resources tool traffic touched;
* end to end: a child recalls / searches what the parent already has, its
  own reads are merged back, the finding carries ``touched_resources`` /
  ``refs`` / ``status`` and the run report records every child;
* gather-first is opt-in (default off) and bounded.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.components.decomposer import Decomposer, SubTaskFinding
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.coverage import ResourceTouchTracker, resource_aliases
from nucleusiq.agents.context.document_search import InMemoryDocumentCorpus
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier
from nucleusiq.agents.context.shared_evidence import (
    LayeredContentStore,
    LayeredDocumentCorpus,
    LayeredEvidenceDossier,
    MergeReport,
    ParentEvidenceView,
    merge_child_evidence,
    merge_corpus,
    merge_dossier,
    merge_store,
)
from nucleusiq.agents.context.store import ContentStore
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools import BaseTool

from nucleusiq.tests.conftest import make_test_prompt

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

DOC_TEXT = (
    "Invoice 4471 from Acme Corp dated 2026-03-01. Total amount 12,500 USD. "
    "Payment terms net 30. Line items: consulting services March, travel "
    "expenses, and software licences for the analytics platform. "
    "Approved by finance on 2026-03-05 and archived in the vendor folder."
)


class ScriptedLLM(MockLLM):
    """Replays ``("text", str)`` or ``("tools", [(name, args), ...])`` entries."""

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
    """Idempotent read tool; long enough output to be indexed into the corpus."""

    def __init__(self, *, log: list[str] | None = None, idempotent: bool = True):
        super().__init__(
            name="read_document", description="Read a document", idempotent=idempotent
        )
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


def _agent(llm: MockLLM, config: AgentConfig, *, tools=None, name="Parent") -> Agent:
    return Agent(
        name=name,
        role="Analyst",
        objective="Analyse",
        prompt=make_test_prompt(),
        llm=llm,
        tools=tools or [],
        config=config,
    )


def _std(**overrides: Any) -> AgentConfig:
    base = {"execution_mode": ExecutionMode.STANDARD, "max_tool_calls": 8}
    base.update(overrides)
    return AgentConfig(**base)


def _tool_messages(request: dict[str, Any]) -> list[str]:
    return [
        str(m.get("content", ""))
        for m in request["messages"]
        if m.get("role") == "tool"
    ]


# --------------------------------------------------------------------------- #
# LayeredContentStore                                                          #
# --------------------------------------------------------------------------- #


class TestLayeredContentStore:
    def test_reads_fall_through_and_writes_stay_local(self):
        parent = ContentStore()
        parent.store("lookup:p1", "parent body", 10, tool_name="lookup")
        child = LayeredContentStore(parent)

        assert child.retrieve("lookup:p1") == "parent body"
        assert child.preview("lookup:p1") == "parent body"
        assert child.metadata("lookup:p1").tool_name == "lookup"
        assert child.contains("lookup:p1") and "lookup:p1" in child
        assert child.local_keys() == []

        child.store("lookup:c1", "child body", 5)
        assert child.local_keys() == ["lookup:c1"]
        assert parent.retrieve("lookup:c1") is None
        assert parent.size == 1
        assert child.size == 2
        assert set(child.keys()) == {"lookup:c1", "lookup:p1"}
        assert set(iter(child)) == {"lookup:c1", "lookup:p1"}

    def test_local_shadows_parent_and_missing_is_none(self):
        parent = ContentStore()
        parent.store("k", "old", 1)
        child = LayeredContentStore(parent)
        child.store("k", "new", 1)
        assert child.retrieve("k") == "new"
        assert parent.retrieve("k") == "old"
        assert child.retrieve("missing") is None
        assert child.metadata("missing") is None
        assert not child.contains("missing")

    def test_no_parent_behaves_like_plain_store(self):
        child = LayeredContentStore(None)
        assert child.retrieve("x") is None
        child.store("x", "v", 1)
        assert child.keys() == ["x"] and child.size == 1


# --------------------------------------------------------------------------- #
# LayeredDocumentCorpus                                                        #
# --------------------------------------------------------------------------- #


class TestLayeredDocumentCorpus:
    def test_search_merges_both_layers_without_touching_parent_counters(self):
        parent = InMemoryDocumentCorpus()
        parent.index_document("doc-a", "alpha beta gamma " * 20, title="A")
        before = parent.stats()
        child = LayeredDocumentCorpus(parent)
        child.index_document("doc-b", "alpha delta epsilon " * 20, title="B")

        hits = child.search("alpha", limit=10)
        assert {h.document_id for h in hits} == {"doc-a", "doc-b"}
        assert parent.stats() == before
        assert child.stats().document_chunks_returned == len(hits)

        only_parent = child.search("gamma")
        assert {h.document_id for h in only_parent} == {"doc-a"}
        assert child.search("") == []

    def test_lookups_fall_through_and_local_shadows(self):
        parent = InMemoryDocumentCorpus()
        parent.index_document("doc-a", "alpha " * 50, title="A-parent")
        child = LayeredDocumentCorpus(parent)

        parent_chunk = parent.chunks_for_document("doc-a")[0]
        assert child.get_chunk(parent_chunk.id) is parent_chunk
        assert child.chunks_for_document("doc-a") == parent.chunks_for_document("doc-a")
        assert [d.id for d in child.list_documents()] == ["doc-a"]
        assert child.local_documents() == []

        child.index_document("doc-a", "omega " * 50, title="A-child")
        titles = {d.id: d.title for d in child.list_documents()}
        assert titles == {"doc-a": "A-child"}
        assert {d.id for d in parent.list_documents()} == {"doc-a"}
        assert parent.list_documents()[0].title == "A-parent"
        # A shadowed id is not searched in the parent again.
        assert {h.document_id for h in child.search("alpha")} == set()

    def test_filters_apply_to_parent_hits(self):
        parent = InMemoryDocumentCorpus()
        parent.index_document("a", "alpha " * 30, metadata={"kind": "pdf"})
        parent.index_document("b", "alpha " * 30, metadata={"kind": "csv"})
        child = LayeredDocumentCorpus(parent)
        hits = child.search("alpha", filters={"kind": "pdf"})
        assert {h.document_id for h in hits} == {"a"}


# --------------------------------------------------------------------------- #
# LayeredEvidenceDossier                                                       #
# --------------------------------------------------------------------------- #


class TestLayeredEvidenceDossier:
    def test_union_reads_local_writes(self):
        parent = InMemoryEvidenceDossier()
        parent.add_evidence(
            claim="Revenue grew 10%", source_ref="doc-a", tags=("revenue",)
        )
        child = LayeredEvidenceDossier(parent)
        child.add_evidence(claim="Margin fell", source_ref="doc-b", tags=("margin",))
        child.add_gap(
            question="No cash-flow data", reason="not in corpus", tags=("cashflow",)
        )

        assert len(child.list()) == 3
        assert len(child.local_items()) == 2
        assert len(parent.list()) == 1
        assert [i.claim for i in child.list(tags=("revenue",))] == ["Revenue grew 10%"]
        assert [i.claim for i in child.query("margin fell")] == ["Margin fell"]
        assert [i.claim for i in child.query("revenue")] == ["Revenue grew 10%"]
        assert child.query("") == []

        summary = child.summarize(max_chars=10_000)
        assert "Revenue grew 10%" in summary and "Margin fell" in summary
        assert child.summarize(max_chars=0) == ""
        assert child.summarize(max_chars=40).endswith("...")
        assert "gap" not in child.summarize(max_chars=10_000, include_gaps=False)

        cov = child.coverage(("revenue", "margin", "cashflow", "other"))
        assert cov.present_tags == ("revenue", "margin")
        assert cov.missing_tags == ("cashflow", "other")
        assert cov.gap_tags == ("cashflow",)

    def test_status_filter_and_no_parent(self):
        child = LayeredEvidenceDossier(None)
        child.add_evidence(claim="x", source_ref="r", status="partial")
        assert [i.claim for i in child.list(status="partial")] == ["x"]
        assert child.list(status="supported") == []


# --------------------------------------------------------------------------- #
# Merge                                                                        #
# --------------------------------------------------------------------------- #


class TestMerge:
    def test_store_merge_copies_local_only_and_is_idempotent(self):
        parent = ContentStore()
        parent.store("p", "parent", 3, tool_name="lookup")
        child = LayeredContentStore(parent)
        child.store("c", "child", 2, trusted=False, tool_name="read")

        report = MergeReport(child="s1")
        merge_store(parent, child, report)
        assert report.store_entries == 1
        assert parent.retrieve("c") == "child"
        meta = parent.metadata("c")
        assert (
            meta.trusted is False
            and meta.tool_name == "read"
            and meta.original_tokens == 2
        )

        again = MergeReport()
        merge_store(parent, child, again)
        assert again.store_entries == 0 and again.skipped_existing == 1

    def test_plain_child_store_merges_all_keys(self):
        parent, child = ContentStore(), ContentStore()
        child.store("a", "1", 1)
        child.store("b", "2", 1)
        report = MergeReport()
        merge_store(parent, child, report)
        assert report.store_entries == 2 and parent.size == 2

    def test_corpus_merge_keeps_chunks_and_skips_existing(self):
        parent = InMemoryDocumentCorpus()
        parent.index_document("shared", "alpha " * 40)
        child = LayeredDocumentCorpus(parent)
        child.index_document("new", "beta gamma " * 40, title="New")
        child.index_document("shared", "replaced " * 40)

        report = MergeReport()
        merge_corpus(parent, child, report)
        assert report.documents == 1 and report.document_ids == ["new"]
        assert report.skipped_existing == 1
        assert parent.chunks_for_document("new") == child.chunks_for_document("new")
        assert {h.document_id for h in parent.search("gamma")} == {"new"}
        assert parent.chunks_for_document("shared")[0].text.startswith("alpha")

    def test_dossier_merge_and_full_parent(self):
        parent = InMemoryEvidenceDossier(max_items=2)
        parent.add_evidence(claim="one", source_ref="r")
        child = LayeredEvidenceDossier(parent)
        child.add_evidence(claim="two", source_ref="r")
        child.add_evidence(claim="three", source_ref="r")

        report = MergeReport()
        merge_dossier(parent, child, report)
        assert report.evidence_items == 1
        assert [i.claim for i in parent.list()] == ["one", "two"]

        again = MergeReport()
        merge_dossier(parent, child, again)
        assert again.evidence_items == 0 and again.skipped_existing == 1

    def test_merge_child_evidence_end_to_end_objects(self):
        class Holder:
            pass

        parent, child = Holder(), Holder()
        parent._context_engine = Holder()
        parent._context_engine.store = ContentStore()
        parent._document_corpus = InMemoryDocumentCorpus()
        parent._evidence_dossier = InMemoryEvidenceDossier()
        child._context_engine = Holder()
        child._context_engine.store = LayeredContentStore(parent._context_engine.store)
        child._document_corpus = LayeredDocumentCorpus(parent._document_corpus)
        child._evidence_dossier = LayeredEvidenceDossier(parent._evidence_dossier)

        child._context_engine.store.store("k", "v", 1)
        child._document_corpus.index_document("d", "text " * 50)
        child._evidence_dossier.add_evidence(claim="c", source_ref="d")

        report = merge_child_evidence(parent, child, label="s1")
        assert report.to_dict() == {
            "child": "s1",
            "store_entries": 1,
            "documents": 1,
            "evidence_items": 1,
            "skipped_existing": 0,
        }
        assert parent._context_engine.store.retrieve("k") == "v"
        assert [d.id for d in parent._document_corpus.list_documents()] == ["d"]
        assert len(parent._evidence_dossier.list()) == 1

    def test_merge_child_evidence_tolerates_missing_stores(self):
        class Holder:
            pass

        report = merge_child_evidence(Holder(), Holder(), label="x")
        assert report.store_entries == report.documents == report.evidence_items == 0


# --------------------------------------------------------------------------- #
# ParentEvidenceView                                                           #
# --------------------------------------------------------------------------- #


class TestParentEvidenceView:
    def test_empty_view_has_no_hint(self):
        view = ParentEvidenceView(
            store=ContentStore(),
            corpus=InMemoryDocumentCorpus(),
            dossier=InMemoryEvidenceDossier(),
        )
        assert view.has_content is False
        assert view.system_hint() == ""
        assert ParentEvidenceView().system_hint() == ""

    def test_hint_names_the_tools_and_prefetched_resources(self):
        corpus = InMemoryDocumentCorpus()
        corpus.index_document("a", "alpha " * 30)
        store = ContentStore()
        store.store("k", "v", 1)
        view = ParentEvidenceView(
            store=store,
            corpus=corpus,
            prefetched_resources=tuple(f"docs/{i}.pdf" for i in range(15)),
        )
        hint = view.system_hint()
        assert hint.startswith("Shared evidence: ")
        assert "1 document(s) are already indexed" in hint
        assert "search_document_corpus" in hint
        assert "recall_tool_result" in hint
        assert "docs/0.pdf" in hint and "(+3 more)" in hint

    def test_from_agent_picks_up_agent_stores(self):
        class Holder:
            pass

        agent = Holder()
        agent._context_engine = Holder()
        agent._context_engine.store = ContentStore()
        agent._document_corpus = InMemoryDocumentCorpus()
        agent._evidence_dossier = InMemoryEvidenceDossier()
        view = ParentEvidenceView.from_agent(agent, prefetched=["a", " ", "b"])
        assert view.store is agent._context_engine.store
        assert view.corpus is agent._document_corpus
        assert view.dossier is agent._evidence_dossier
        assert view.prefetched_resources == ("a", "b")

        bare = ParentEvidenceView.from_agent(Holder())
        assert bare.store is None and bare.corpus is None and bare.dossier is None


# --------------------------------------------------------------------------- #
# ResourceTouchTracker                                                         #
# --------------------------------------------------------------------------- #


class TestResourceTouchTracker:
    def test_aliases(self):
        assert resource_aliases("Docs/Q3 Report.PDF") == {
            "docs/q3 report.pdf",
            "q3 report.pdf",
        }
        assert resource_aliases("  ") == set()
        assert resource_aliases("ab") == {"ab"}

    def test_observe_args_and_result_head(self):
        t = ResourceTouchTracker(
            ["docs/a.pdf", "docs/b.pdf", "https://x.io/c", "docs/a.pdf"]
        )
        assert t.resources == ["docs/a.pdf", "docs/b.pdf", "https://x.io/c"]
        assert t.observe(tool_name="read", tool_args={"path": "C:\\share\\A.PDF"}) == [
            "docs/a.pdf"
        ]
        assert t.observe(
            tool_name="fetch", tool_result="Fetched https://x.io/c ok"
        ) == ["https://x.io/c"]
        assert t.observe(tool_name="fetch", tool_args={"path": "docs/a.pdf"}) == []
        assert t.touched == ["docs/a.pdf", "https://x.io/c"]
        assert t.unprocessed == ["docs/b.pdf"]
        assert t.to_dict()["touched_via"] == {
            "docs/a.pdf": "read",
            "https://x.io/c": "fetch",
        }

    def test_result_scan_is_bounded_to_head(self):
        t = ResourceTouchTracker(["needle.txt"])
        far = "x" * 10_000 + " needle.txt"
        assert t.observe(tool_name="read", tool_result=far) == []
        assert t.observe(tool_name="read", tool_result={"body": "see needle.txt"}) == [
            "needle.txt"
        ]

    def test_mark_and_no_resources(self):
        t = ResourceTouchTracker([])
        assert t.observe(tool_name="x", tool_args={"a": "b"}) == []
        t2 = ResourceTouchTracker(["docs/a.pdf", "docs/b.pdf"])
        t2.mark(["a.pdf"], via="child:s1")
        assert t2.touched == ["docs/a.pdf"]
        assert t2.to_dict()["touched_via"]["docs/a.pdf"] == "child:s1"


# --------------------------------------------------------------------------- #
# SubTaskFinding                                                               #
# --------------------------------------------------------------------------- #


class TestSubTaskFinding:
    def test_to_dict_is_backward_compatible(self):
        assert SubTaskFinding(id="s1", objective="A", result="ok").to_dict() == {
            "id": "s1",
            "objective": "A",
            "result": "ok",
        }
        full = SubTaskFinding(
            id="s1",
            objective="A",
            result="Error: boom",
            status="error",
            resources=["a"],
            touched_resources=["a"],
            refs=["read:1"],
            merged={"documents": 1},
        )
        assert full.failed is True
        d = full.to_dict()
        assert (
            d["status"] == "error"
            and d["refs"] == ["read:1"]
            and d["merged"] == {"documents": 1}
        )

    def test_synthesis_prompt_renders_handoff_metadata(self):
        prompt = Decomposer.build_synthesis_prompt(
            "Extract",
            [
                {
                    "id": "s1",
                    "objective": "Dates",
                    "result": "2026-03-01",
                    "status": "partial",
                    "touched_resources": ["docs/a.pdf"],
                    "refs": [f"read_document:c{i}" for i in range(10)],
                },
                {
                    "id": "s2",
                    "objective": "Amounts",
                    "result": "12,500",
                    "status": "success",
                },
            ],
        )
        assert "_status: partial_" in prompt
        assert "resources covered: docs/a.pdf" in prompt
        assert "evidence refs (recall_tool_result): read_document:c0" in prompt
        assert "(+2 more)" in prompt
        assert "_status: success_" not in prompt


# --------------------------------------------------------------------------- #
# Engine wiring                                                                #
# --------------------------------------------------------------------------- #


class TestEngineStore:
    def test_engine_accepts_injected_store(self):
        parent = ContentStore()
        parent.store("k", "v", 1)
        layered = LayeredContentStore(parent)
        engine = ContextEngine(config=ContextConfig(), store=layered)
        assert engine.store is layered
        assert engine.store.retrieve("k") == "v"
        assert isinstance(ContextEngine(config=ContextConfig()).store, ContentStore)


# --------------------------------------------------------------------------- #
# End to end                                                                   #
# --------------------------------------------------------------------------- #


class TestChildSharesParentEvidence:
    @pytest.mark.asyncio
    async def test_child_recalls_and_searches_parent_evidence_and_merges_back(self):
        log: list[str] = []
        llm = ScriptedLLM(
            [
                ("text", "parent warm-up"),
                # child: read its own resource, recall a parent offload, search
                # a parent-indexed document, answer.
                ("tools", [("read_document", {"path": "docs/b.pdf"})]),
                ("tools", [("recall_tool_result", {"ref": "read_document:p1"})]),
                ("tools", [("search_document_corpus", {"query": "acme invoice"})]),
                ("text", "child answer"),
            ]
        )
        parent = _agent(llm, _std(), tools=[ReadDocTool(log=log)])
        await parent.execute(Task(id="warm", objective="warm up"))
        # Seed the parent's evidence as if it had read docs/a.pdf itself.
        parent._context_engine.store.store(
            "read_document:p1",
            "PARENT OFFLOAD: " + DOC_TEXT,
            80,
            tool_name="read_document",
        )
        parent._document_corpus.index_document(
            "docs/a.pdf", DOC_TEXT, title="docs/a.pdf"
        )
        parent_store_size = parent._context_engine.store.size
        parent._run_recorder.record_decision("marker", True)

        parent_task = Task(
            id="t", objective="Extract invoices", resources=["docs/a.pdf", "docs/b.pdf"]
        )
        # ``_setup_execution`` does this for the real parent run.
        parent._resource_tracker = ResourceTouchTracker(
            parent_task.effective_resources()
        )
        findings = await Decomposer().run_sub_tasks(
            parent,
            [{"id": "s1", "objective": "Read docs/b.pdf", "resources": ["docs/b.pdf"]}],
            parent_task=parent_task,
        )

        assert len(findings) == 1
        f = findings[0]
        assert f["result"] == "child answer"
        assert f["status"] == "success"
        assert f["resources"] == ["docs/b.pdf"]
        assert f["touched_resources"] == ["docs/b.pdf"]
        # The child's own read was indexed locally and merged into the parent.
        assert f["merged"]["documents"] >= 1
        parent_docs = {d.id for d in parent._document_corpus.list_documents()}
        assert "docs/a.pdf" in parent_docs and len(parent_docs) >= 2
        assert parent._context_engine.store.size >= parent_store_size
        assert parent._resource_tracker.touched == ["docs/b.pdf"]
        assert parent._resource_tracker.unprocessed == ["docs/a.pdf"]

        # The child saw the parent's offload and the parent's indexed document
        # (the fullest request is the one carrying every tool result; the
        # very last one is the tools-free synthesis pass).
        fullest = max(llm.requests, key=lambda r: len(r["messages"]))
        tool_msgs = _tool_messages(fullest)
        assert any("PARENT OFFLOAD" in m for m in tool_msgs)
        assert any("docs/a.pdf" in m and "Acme" in m for m in tool_msgs)
        system = fullest["messages"][0]["content"]
        assert "Shared evidence:" in system and "search_document_corpus" in system
        # Only the child read a document — and only its own slice.
        assert log == ["docs/b.pdf"]

        # The parent's run report records the child with its resolved limits.
        children = parent._run_recorder.children
        assert len(children) == 1
        child_info = children[0]
        assert child_info["id"] == "s1" and child_info["status"] == "success"
        assert child_info["max_tool_calls"] == 8
        assert child_info["touched_resources"] == ["docs/b.pdf"]
        assert child_info["resources"] == ["docs/b.pdf"]
        assert parent._run_recorder.counters.children_spawned == 1
        assert parent._run_recorder.counters.children_failed == 0

    @pytest.mark.asyncio
    async def test_failed_child_is_recorded_as_failed(self):
        class Boom(ScriptedLLM):
            async def call(self, **kwargs: Any):
                if self.calls >= 1:
                    self.calls += 1
                    raise RuntimeError("provider down")
                return await super().call(**kwargs)

        llm = Boom([("text", "parent warm-up")])
        parent = _agent(llm, _std(), tools=[ReadDocTool()])
        await parent.execute(Task(id="warm", objective="warm up"))
        findings = await Decomposer().run_sub_tasks(
            parent,
            [{"id": "s1", "objective": "A"}],
            parent_task=Task(id="t", objective="x"),
        )
        assert findings[0]["status"] in {"error", "failed"}
        assert parent._run_recorder.counters.children_failed == 1


# --------------------------------------------------------------------------- #
# Gather-first                                                                 #
# --------------------------------------------------------------------------- #


class TestGatherFirst:
    def test_default_off(self):
        assert AgentConfig().decomposition_gather_first is False

    def test_gather_objective_and_tools(self):
        text = Decomposer.build_gather_objective([f"docs/{i}.pdf" for i in range(45)])
        assert "GATHER PHASE" in text and "Do NOT analyse" in text
        assert "docs/0.pdf" in text and "and 5 more" in text

        class Holder:
            tools = [ReadDocTool(), ReadDocTool(idempotent=False)]

        assert [t.idempotent for t in Decomposer.gather_tools(Holder())] == [True]
        assert Decomposer.gather_tools(object()) == []

    @pytest.mark.asyncio
    async def test_gather_phase_skips_without_resources_or_idempotent_tools(self):
        d = Decomposer()
        assert (await d.run_gather_phase(object(), None))[
            "skipped"
        ] == "no resources declared"

        class Holder:
            tools = [ReadDocTool(idempotent=False)]

        out = await d.run_gather_phase(
            Holder(), Task(id="t", objective="x", resources=["a"])
        )
        assert out["ran"] is False and out["skipped"] == "no idempotent tools available"
        assert out["unprocessed"] == ["a"]

    @pytest.mark.asyncio
    async def test_gather_phase_reads_resources_into_parent_and_is_capped(self):
        log: list[str] = []
        llm = ScriptedLLM(
            [
                ("text", "parent warm-up"),
                ("tools", [("read_document", {"path": "docs/a.pdf"})]),
                ("tools", [("read_document", {"path": "docs/b.pdf"})]),
                ("text", "docs/a.pdf: read\ndocs/b.pdf: read"),
            ]
        )
        parent = _agent(
            llm,
            _std(max_tool_calls=20, decomposition_gather_first=True),
            tools=[ReadDocTool(log=log)],
        )
        await parent.execute(Task(id="warm", objective="warm up"))
        task = Task(id="t", objective="Extract", resources=["docs/a.pdf", "docs/b.pdf"])
        parent._resource_tracker = ResourceTouchTracker(task.effective_resources())

        summary = await Decomposer().run_gather_phase(parent, task)

        assert summary["ran"] is True
        assert summary["max_tool_calls"] == 4
        assert summary["touched"] == ["docs/a.pdf", "docs/b.pdf"]
        assert summary["unprocessed"] == []
        assert summary["merged"]["documents"] >= 2
        assert log == ["docs/a.pdf", "docs/b.pdf"]
        docs = {d.id for d in parent._document_corpus.list_documents()}
        assert len(docs) >= 2
        # Children spawned afterwards are told what is already fetched.
        view = Decomposer._parent_view(parent)
        hint = view.system_hint()
        assert "already fetched: docs/a.pdf, docs/b.pdf" in hint
        children = parent._run_recorder.children
        assert children and children[0]["id"] == "gather"
        assert children[0]["max_tool_calls"] == 4

    @pytest.mark.asyncio
    async def test_complex_run_with_gather_first_records_decision(self):
        """Full Autonomous COMPLEX run: gather child, two analysis children,
        synthesis, critic — with the gather decision in the report."""
        split = {
            "gate1": True,
            "gate2": True,
            "gate3": True,
            "gate4": True,
            "complexity": "complex",
            "sub_tasks": [
                {"id": "s1", "objective": "Dates from a", "resources": ["docs/a.pdf"]},
                {
                    "id": "s2",
                    "objective": "Amounts from b",
                    "resources": ["docs/b.pdf"],
                },
            ],
        }
        llm = ScriptedLLM(
            [
                ("text", json.dumps(split)),
                # gather child
                ("tools", [("read_document", {"path": "docs/a.pdf"})]),
                ("tools", [("read_document", {"path": "docs/b.pdf"})]),
                ("text", "docs/a.pdf: read\ndocs/b.pdf: read"),
                # analysis children run concurrently; both answer from shared
                # evidence without new reads.
                ("text", "s-answer"),
                ("text", "s-answer"),
                # synthesis + critic
                ("text", "Final: 2026-03-01, 12,500 USD"),
                ("text", "VERDICT: PASS\nSCORE: 0.9\nFEEDBACK: ok"),
            ]
        )
        log: list[str] = []
        parent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_tool_calls=20,
                max_retries=2,
                decomposition_gather_first=True,
            ),
            tools=[ReadDocTool(log=log)],
        )
        result = await parent.execute(
            Task(id="t", objective="Extract", resources=["docs/a.pdf", "docs/b.pdf"])
        )

        assert result.output == "Final: 2026-03-01, 12,500 USD"
        gather = result.diagnostics.decisions["gather_first"]
        assert gather["ran"] is True and gather["touched"] == [
            "docs/a.pdf",
            "docs/b.pdf",
        ]
        kinds = [e.kind for e in result.diagnostics.timeline]
        assert "gather_first_completed" in kinds
        findings = result.diagnostics.decisions["sub_task_findings"]
        assert [f["id"] for f in findings] == ["s1", "s2"]
        assert result.diagnostics.counters.children_spawned == 3
        assert log == ["docs/a.pdf", "docs/b.pdf"]

    @pytest.mark.asyncio
    async def test_complex_run_without_flag_does_not_gather(self):
        split = {
            "gate1": True,
            "gate2": True,
            "gate3": True,
            "gate4": True,
            "complexity": "complex",
            "sub_tasks": [
                {"id": "s1", "objective": "A", "resources": ["docs/a.pdf"]},
                {"id": "s2", "objective": "B", "resources": ["docs/b.pdf"]},
            ],
        }
        llm = ScriptedLLM(
            [
                ("text", json.dumps(split)),
                ("text", "s-answer"),
                ("text", "s-answer"),
                ("text", "Final"),
                ("text", "VERDICT: PASS\nSCORE: 0.9\nFEEDBACK: ok"),
            ]
        )
        parent = _agent(
            llm,
            AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_tool_calls=20,
                max_retries=2,
                coverage_followup=False,
            ),
            tools=[ReadDocTool()],
        )
        result = await parent.execute(
            Task(id="t", objective="Extract", resources=["docs/a.pdf", "docs/b.pdf"])
        )
        assert result.output == "Final"
        assert "gather_first" not in result.diagnostics.decisions
        assert result.diagnostics.counters.children_spawned == 2
        assert llm.calls == 5
        # Coverage is still recorded even though no follow-up was allowed.
        cov = result.metadata["coverage"]
        assert cov["unprocessed"] == ["docs/a.pdf", "docs/b.pdf"]
        assert cov["followup"] is None and cov["blocked"] is False
        assert result.diagnostics.decisions["coverage"]["followup"] is None
