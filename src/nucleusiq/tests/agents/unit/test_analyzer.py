"""Analyzer rules over ``RunReport`` (WS-7 §7.2, PR G).

Every rule is exercised twice: once with a report that must trigger it
and once with a healthy report that must not.  A final end-to-end test
runs a real Agent and checks the findings land on
``AgentResult.diagnostics`` and in ``display()``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.agents.diagnostics import (
    Finding,
    RunRecorder,
    RunReport,
    TerminationReason,
    analyze,
    attach_findings,
    load_report,
    recommendations,
    registered_rules,
)
from nucleusiq.agents.diagnostics.__main__ import main as cli_main
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM

from nucleusiq.tests.conftest import make_test_prompt

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _agent(config: AgentConfig) -> Agent:
    return Agent(
        name="AnalyzerAgent",
        role="Analyst",
        objective="Answer",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        config=config,
    )


def make_report(
    *,
    termination: TerminationReason = TerminationReason.COMPLETED,
    counters: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    decisions: dict[str, Any] | None = None,
    children: list[dict[str, Any]] | None = None,
    coverage: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
    status: str = "success",
) -> RunReport:
    rec = RunRecorder()
    for key, value in (counters or {}).items():
        setattr(rec.counters, key, value)
    for key, value in (decisions or {}).items():
        rec.record_decision(key, value)
    for child in children or []:
        rec.record_child(child, failed=child.get("status", "success") != "success")
    rec.coverage = coverage
    rec.preflight = preflight
    rec.set_termination(termination, "test")
    base_cfg = {
        "context_window": 65_536,
        "response_reserve": 4_096,
        "llm_max_output_tokens": 2_048,
        "max_tool_calls": 40,
        "max_context_tool_calls": 20,
        "response_format_set": False,
    }
    base_cfg.update(config or {})
    return rec.build(
        framework_version="test",
        provider="openai_compatible",
        model="gemma",
        mode="autonomous",
        agent_name="a",
        task_id="t",
        status=status,
        config_resolved=base_cfg,
        default_reason=TerminationReason.COMPLETED,
    )


def codes(report: RunReport) -> list[str]:
    return [f.code for f in analyze(report)]


HEALTHY = make_report()


# --------------------------------------------------------------------------- #
# Baseline                                                                     #
# --------------------------------------------------------------------------- #


class TestBaseline:
    def test_healthy_report_has_no_findings(self):
        assert codes(HEALTHY) == []

    def test_every_rule_is_registered_once(self):
        names = registered_rules()
        assert len(names) == len(set(names))
        assert "loop_recall_deadlock" in names
        assert "decomp_coverage_gap" in names

    def test_analyze_never_raises_on_partial_report(self):
        # A report with garbage in every free-form slot.
        report = make_report(
            decisions={
                "classification": "not-a-dict",
                "coverage": 42,
                "synthesis_handoff": ["x"],
                "gather_first": None,
                "structured_output": "nope",
            },
            children=[{"id": None, "window": "abc", "max_tool_calls": "?"}],
            coverage={"unprocessed": "not-a-list"},
            preflight={"fitness": 7},
            config={"context_window": "big", "max_tool_calls": None},
        )
        assert isinstance(analyze(report), list)

    def test_findings_sorted_critical_first(self):
        report = make_report(
            termination=TerminationReason.CONTEXT_OVERFLOW,
            counters={"empty_responses": 3},
            preflight={"fitness": "unfit", "working_tokens": 1000},
        )
        found = analyze(report)
        sev = [f.severity for f in found]
        assert sev == sorted(sev, key={"critical": 0, "warning": 1, "info": 2}.get)
        assert found[0].code == "CONTEXT_OVERFLOW_400"

    def test_attach_findings_and_recommendations(self):
        report = make_report(termination=TerminationReason.TOOL_BUDGET)
        out = attach_findings(report)
        assert [f.code for f in out.findings] == ["TOOL_BUDGET_EXHAUSTED"]
        assert out.recommendations and out.recommendations[0].startswith(
            "[TOOL_BUDGET_EXHAUSTED]"
        )
        assert report.findings == ()  # original untouched (frozen model)
        assert out.analyze().findings == out.findings

    def test_recommendations_dedupe(self):
        f = Finding(code="A", severity="info", title="a", fix_now="do x")
        g = Finding(code="B", severity="info", title="b", fix_now="do x")
        assert recommendations([f, g]) == ["[A] do x"]

    def test_every_finding_cites_evidence(self):
        report = make_report(
            termination=TerminationReason.NO_PROGRESS,
            counters={
                "recall_errors": 5,
                "dedup_banners": 5,
                "emergency_count": 3,
                "children_failed": 1,
                "children_spawned": 2,
                "synthesis_runs": 1,
                "empty_responses": 2,
            },
            config={
                "response_format_set": True,
                "llm_max_output_tokens": 8_000,
                "window_is_fallback": True,
            },
            decisions={
                "classification": {"is_complex": True, "downgrade_reason": "gate4"},
                "synthesis_handoff": {"per_finding_chars": 100, "truncated": ["s1"]},
                "structured_output": {"valid": False, "errors": "x"},
                "gather_first": {"ran": False, "skipped": "no tools"},
            },
            children=[
                {
                    "id": "s1",
                    "status": "error",
                    "window": 8_192,
                    "max_tool_calls": 15,
                    "touched_resources": ["a", "b"],
                },
                {
                    "id": "s2",
                    "status": "success",
                    "window": 8_192,
                    "max_tool_calls": 15,
                    "touched_resources": ["a", "b"],
                },
            ],
            coverage={"unprocessed": ["c"], "resources": ["a", "b", "c"]},
            preflight={"fitness": "marginal", "working_tokens": 20_000},
        )
        found = analyze(report)
        assert len(found) >= 12
        for f in found:
            assert f.evidence, f.code
            assert f.title and f.cause and f.fix_now, f.code


class TestOffice0713Fixture:
    """The 0.7.13 office run, as its report would have looked (design §11)."""

    REPORT = make_report(
        termination=TerminationReason.TOOL_BUDGET,
        status="error",
        counters={
            "recall_errors": 41,
            "dedup_banners": 63,
            "emergency_count": 3,
            "tool_calls_business": 300,
            "tool_calls_context": 118,
            "rounds": 121,
            "children_spawned": 3,
        },
        config={
            "context_window": 65_536,
            "max_tool_calls": 300,
            "response_format_set": True,
            "tool_count": 19,
            "idempotent_tool_count": 19,
        },
        decisions={
            "classification": {
                "is_complex": True,
                "sub_tasks": 3,
                "classifier_called": True,
            },
            "synthesis_handoff": {
                "per_finding_chars": 2_000,
                "truncated": ["s1", "s2"],
            },
        },
        children=[
            {
                "id": "s1",
                "status": "success",
                "window": 8_192,
                "max_tool_calls": 15,
                "touched_resources": ["d1.pdf", "d2.pdf", "d3.pdf", "d4.pdf"],
            },
            {
                "id": "s2",
                "status": "success",
                "window": 8_192,
                "max_tool_calls": 15,
                "touched_resources": ["d1.pdf", "d2.pdf", "d3.pdf", "d5.pdf"],
            },
            {
                "id": "s3",
                "status": "success",
                "window": 8_192,
                "max_tool_calls": 15,
                "touched_resources": ["d2.pdf", "d3.pdf"],
            },
        ],
        coverage={
            "resources": [f"d{i}.pdf" for i in range(1, 10)],
            "touched": ["d1.pdf", "d2.pdf", "d3.pdf", "d4.pdf", "d5.pdf"],
            "unprocessed": ["d6.pdf", "d7.pdf", "d8.pdf", "d9.pdf"],
        },
    )

    def test_emits_the_incident_findings_with_evidence(self):
        found = {f.code: f for f in analyze(self.REPORT)}
        for code in (
            "LOOP_RECALL_DEADLOCK",
            "CHILD_WINDOW_MISMATCH",
            "CHILD_HIDDEN_CAP",
            "DECOMP_SHARED_SOURCE",
            "DECOMP_COVERAGE_GAP",
            "HANDOFF_TRUNCATED",
            "EMERGENCY_THRASH",
            "TOOL_BUDGET_EXHAUSTED",
        ):
            assert code in found, (code, sorted(found))

        deadlock = found["LOOP_RECALL_DEADLOCK"]
        assert deadlock.severity == "critical"
        assert deadlock.evidence["counters.recall_errors"] == 41
        assert deadlock.evidence["counters.dedup_banners"] == 63

        mismatch = found["CHILD_WINDOW_MISMATCH"]
        assert mismatch.evidence["config_resolved.context_window"] == 65_536
        assert {c["window"] for c in mismatch.evidence["children[].window"]} == {8_192}

        cap = found["CHILD_HIDDEN_CAP"]
        assert cap.severity == "warning" and "hidden plugin" in cap.title
        assert len(cap.evidence["children[].max_tool_calls"]) == 3

        shared = found["DECOMP_SHARED_SOURCE"]
        pairs = {
            tuple(o["children"])
            for o in shared.evidence["children[].touched_resources"]
        }
        assert ("s1", "s2") in pairs and ("s1", "s3") in pairs and ("s2", "s3") in pairs

        gap = found["DECOMP_COVERAGE_GAP"]
        assert gap.evidence["coverage.unprocessed"] == [
            "d6.pdf",
            "d7.pdf",
            "d8.pdf",
            "d9.pdf",
        ]

        # Critical first, and the deadlock is the headline.
        assert analyze(self.REPORT)[0].code == "LOOP_RECALL_DEADLOCK"
        # The report explains itself and survives redaction end to end.
        text = attach_findings(self.REPORT).explain(redact=True)
        assert "LOOP_RECALL_DEADLOCK" in text and "d6.pdf" not in text


# --------------------------------------------------------------------------- #
# Loop / budget rules                                                          #
# --------------------------------------------------------------------------- #


class TestLoopRules:
    def test_recall_deadlock(self):
        report = make_report(
            counters={"recall_errors": 3, "dedup_banners": 3, "emergency_count": 1}
        )
        found = analyze(report)
        assert found[0].code == "LOOP_RECALL_DEADLOCK"
        assert found[0].severity == "critical"
        assert found[0].evidence["counters.recall_errors"] == 3

    def test_recall_deadlock_needs_all_three_signals(self):
        assert "LOOP_RECALL_DEADLOCK" not in codes(
            make_report(counters={"recall_errors": 9, "dedup_banners": 9})
        )

    def test_no_progress(self):
        assert codes(make_report(termination=TerminationReason.NO_PROGRESS)) == [
            "LOOP_NO_PROGRESS"
        ]
        assert "LOOP_NO_PROGRESS" in codes(
            make_report(counters={"no_progress_rounds": 3})
        )

    def test_emergency_thrash(self):
        assert "EMERGENCY_THRASH" in codes(make_report(counters={"emergency_count": 2}))
        assert "EMERGENCY_THRASH" not in codes(
            make_report(counters={"emergency_count": 1})
        )

    def test_evidence_evicted_unseen_is_critical(self):
        report = make_report(
            counters={"unseen_evidence_evicted": 9, "emergency_count": 1},
            config={"context_window": 12_000},
        )
        found = analyze(report)
        assert found[0].code == "EVIDENCE_EVICTED_UNSEEN"
        assert found[0].severity == "critical"
        assert found[0].evidence["counters.unseen_evidence_evicted"] == 9
        assert "not grounded" in found[0].title
        assert "unseen tool results evicted 9" in attach_findings(report).explain()
        assert "EVIDENCE_EVICTED_UNSEEN" not in codes(
            make_report(counters={"emergency_count": 1})
        )

    def test_window_fallback_from_config_or_preflight(self):
        assert "WINDOW_FALLBACK" in codes(
            make_report(config={"window_is_fallback": True})
        )
        assert "WINDOW_FALLBACK" in codes(
            make_report(preflight={"window_is_fallback": True, "fitness": "ok"})
        )
        assert "WINDOW_FALLBACK" not in codes(
            make_report(config={"context_window": 8_192, "window_is_fallback": False})
        )

    @pytest.mark.parametrize(
        "reason, code",
        [
            (TerminationReason.CONTEXT_OVERFLOW, "CONTEXT_OVERFLOW_400"),
            (TerminationReason.TOOL_BUDGET, "TOOL_BUDGET_EXHAUSTED"),
            (TerminationReason.CONTEXT_TOOL_BUDGET, "CONTEXT_TOOL_BUDGET_EXHAUSTED"),
            (TerminationReason.DEADLINE, "DEADLINE"),
            (TerminationReason.LLM_TIMEOUT, "LLM_TIMEOUT"),
            (TerminationReason.CRITIC_ABSTAIN, "CRITIC_ABSTAIN"),
            (TerminationReason.SCHEMA_INVALID, "SCHEMA_NOT_SATISFIED"),
            (TerminationReason.EMPTY_RESPONSE, "EMPTY_RESPONSES"),
        ],
    )
    def test_termination_driven_rules(self, reason, code):
        found = analyze(make_report(termination=reason))
        assert [f.code for f in found] == [code]
        assert found[0].evidence["termination.reason"] == reason.value

    def test_reserve_lt_max_tokens(self):
        assert "RESERVE_LT_MAX_TOKENS" in codes(
            make_report(
                config={"response_reserve": 1_000, "llm_max_output_tokens": 4_000}
            )
        )
        assert "RESERVE_LT_MAX_TOKENS" not in codes(
            make_report(
                config={"response_reserve": 4_000, "llm_max_output_tokens": 4_000}
            )
        )
        # Unknown reserve → cannot judge → silent.
        assert "RESERVE_LT_MAX_TOKENS" not in codes(
            make_report(
                config={"response_reserve": None, "llm_max_output_tokens": 4_000}
            )
        )

    def test_preflight_unfit_and_marginal(self):
        unfit = analyze(
            make_report(
                preflight={
                    "fitness": "unfit",
                    "action": "downgraded_to_standard",
                    "working_tokens": 9_000,
                }
            )
        )
        assert unfit[0].code == "PREFLIGHT_UNFIT"
        assert "downgraded_to_standard" in unfit[0].title
        marginal = analyze(make_report(preflight={"fitness": "marginal"}))
        assert marginal[0].code == "PREFLIGHT_MARGINAL"
        assert marginal[0].severity == "info"
        # Also picked up from decisions when recorder.preflight was not set.
        assert "PREFLIGHT_UNFIT" in codes(
            make_report(decisions={"preflight": {"fitness": "unfit"}})
        )


# --------------------------------------------------------------------------- #
# Decomposition rules                                                          #
# --------------------------------------------------------------------------- #


class TestDecompositionRules:
    def test_child_window_mismatch_ignores_aux_children(self):
        report = make_report(
            children=[
                {"id": "s1", "window": 8_192},
                {"id": "gather", "window": 8_192},
            ]
        )
        found = [f for f in analyze(report) if f.code == "CHILD_WINDOW_MISMATCH"]
        assert len(found) == 1
        assert found[0].evidence["children[].window"] == [{"id": "s1", "window": 8_192}]
        assert "CHILD_WINDOW_MISMATCH" not in codes(
            make_report(children=[{"id": "s1", "window": 65_536}])
        )

    def test_child_hidden_cap_legacy_vs_tightened(self):
        legacy = analyze(make_report(children=[{"id": "s1", "max_tool_calls": 15}]))
        legacy = [f for f in legacy if f.code == "CHILD_HIDDEN_CAP"][0]
        assert legacy.severity == "warning"
        assert "hidden plugin" in legacy.title

        tight = analyze(make_report(children=[{"id": "s1", "max_tool_calls": 20}]))
        tight = [f for f in tight if f.code == "CHILD_HIDDEN_CAP"][0]
        assert tight.severity == "info"

        assert "CHILD_HIDDEN_CAP" not in codes(
            make_report(children=[{"id": "s1", "max_tool_calls": 40}])
        )
        assert "CHILD_HIDDEN_CAP" not in codes(
            make_report(children=[{"id": "coverage-followup", "max_tool_calls": 2}])
        )

    def test_decomp_shared_source(self):
        report = make_report(
            decisions={"classification": {"is_complex": True}},
            children=[
                {"id": "s1", "touched_resources": ["a.pdf", "b.pdf"]},
                {"id": "s2", "touched_resources": ["a.pdf", "b.pdf", "c.pdf"]},
                {"id": "s3", "touched_resources": ["d.pdf"]},
            ],
        )
        found = [f for f in analyze(report) if f.code == "DECOMP_SHARED_SOURCE"]
        assert len(found) == 1
        overlaps = found[0].evidence["children[].touched_resources"]
        assert overlaps == [{"children": ["s1", "s2"], "shared": 2, "ratio": 1.0}]

    def test_decomp_shared_source_disjoint_is_silent(self):
        report = make_report(
            children=[
                {"id": "s1", "touched_resources": ["a.pdf"]},
                {"id": "s2", "touched_resources": ["b.pdf"]},
            ]
        )
        assert "DECOMP_SHARED_SOURCE" not in codes(report)

    def test_coverage_gap_simple_vs_complex_and_blocked(self):
        simple = analyze(
            make_report(coverage={"unprocessed": ["c"], "resources": ["c"]})
        )
        assert simple[0].code == "COVERAGE_GAP"
        assert simple[0].severity == "warning"

        complex_blocked = analyze(
            make_report(
                decisions={"classification": {"is_complex": True}},
                coverage={"unprocessed": ["c"], "resources": ["c"], "blocked": True},
                status="abstained",
            )
        )
        assert complex_blocked[0].code == "DECOMP_COVERAGE_GAP"
        assert complex_blocked[0].severity == "critical"
        assert "withheld" in complex_blocked[0].title

        assert "COVERAGE_GAP" not in codes(
            make_report(
                coverage={"unprocessed": [], "resources": ["c"], "touched": ["c"]}
            )
        )

    def test_handoff_truncated(self):
        report = make_report(
            decisions={
                "synthesis_handoff": {
                    "per_finding_chars": 2_000,
                    "findings": 2,
                    "truncated": ["s2"],
                }
            },
            children=[{"id": "s1", "refs": 2}, {"id": "s2", "refs": 5}],
        )
        found = [f for f in analyze(report) if f.code == "HANDOFF_TRUNCATED"][0]
        assert "1 sub-task result(s)" in found.title
        assert found.evidence["children[].refs"] == {"s1": 2, "s2": 5}
        assert "HANDOFF_TRUNCATED" not in codes(
            make_report(
                decisions={
                    "synthesis_handoff": {"per_finding_chars": 2_000, "truncated": []}
                }
            )
        )

    def test_decomp_downgraded_and_children_failed(self):
        report = make_report(
            decisions={
                "classification": {
                    "is_complex": False,
                    "downgrade_reason": "Gate 4: sub-tasks share sources",
                }
            },
            children=[
                {"id": "s1", "status": "error", "termination_reason": "tool_budget"}
            ],
        )
        found = {f.code: f for f in analyze(report)}
        assert found["DECOMP_DOWNGRADED"].severity == "info"
        assert found["CHILDREN_FAILED"].evidence["children[].termination_reason"] == [
            {"id": "s1", "termination_reason": "tool_budget"}
        ]

    def test_gather_skipped(self):
        assert "GATHER_SKIPPED" in codes(
            make_report(
                decisions={
                    "gather_first": {"ran": False, "skipped": "no idempotent tools"}
                }
            )
        )
        assert "GATHER_SKIPPED" not in codes(
            make_report(decisions={"gather_first": {"ran": True, "touched": ["a"]}})
        )


# --------------------------------------------------------------------------- #
# Structured output rules                                                      #
# --------------------------------------------------------------------------- #


class TestStructuredRules:
    def test_synth_vs_structured(self):
        assert "SYNTH_VS_STRUCTURED" in codes(
            make_report(
                config={"response_format_set": True}, counters={"synthesis_runs": 1}
            )
        )
        assert "SYNTH_VS_STRUCTURED" not in codes(
            make_report(
                config={"response_format_set": False}, counters={"synthesis_runs": 1}
            )
        )
        assert "SYNTH_VS_STRUCTURED" not in codes(
            make_report(
                config={"response_format_set": True}, counters={"finalizer_runs": 1}
            )
        )

    def test_schema_not_satisfied_from_decision(self):
        found = analyze(
            make_report(
                decisions={"structured_output": {"valid": False, "errors": "missing x"}}
            )
        )
        assert [f.code for f in found] == ["SCHEMA_NOT_SATISFIED"]
        assert "SCHEMA_NOT_SATISFIED" not in codes(
            make_report(decisions={"structured_output": {"valid": True}})
        )


# --------------------------------------------------------------------------- #
# Rendering / IO                                                               #
# --------------------------------------------------------------------------- #


class TestRenderingAndIO:
    def test_explain_renders_findings_children_and_redacts(self):
        report = attach_findings(
            make_report(
                termination=TerminationReason.TOOL_BUDGET,
                children=[
                    {
                        "id": "s1",
                        "status": "success",
                        "termination_reason": "completed",
                        "tool_calls_business": 3,
                        "tool_calls_context": 0,
                        "max_tool_calls": 40,
                        "window": 65_536,
                        "resources": ["/secret/a.pdf"],
                        "touched_resources": ["/secret/a.pdf"],
                    }
                ],
                coverage={"resources": ["/secret/a.pdf"], "touched": ["/secret/a.pdf"]},
            )
        )
        text = report.explain()
        assert "`TOOL_BUDGET_EXHAUSTED`" in text
        assert "- cause:" in text and "- now:" in text and "- evidence:" in text
        assert "children: 1" in text and "touched 1/1" in text
        assert "/secret/a.pdf" in text

        red = report.explain(redact=True)
        assert "/secret/a.pdf" not in red
        assert "`TOOL_BUDGET_EXHAUSTED`" in red
        assert "**Findings**: none" in HEALTHY.explain()

    def test_redacted_hashes_child_resources_too(self):
        report = make_report(
            children=[
                {"id": "s1", "resources": ["/secret/a.pdf"], "touched_resources": []}
            ]
        )
        assert "/secret/a.pdf" not in json.dumps(report.redacted().to_dict())

    def test_load_report_from_dict_str_path_and_result_summary(self, tmp_path):
        report = make_report(termination=TerminationReason.DEADLINE)
        data = report.to_dict()
        assert load_report(data).termination.reason == TerminationReason.DEADLINE
        assert (
            load_report(json.dumps(data)).termination.reason
            == TerminationReason.DEADLINE
        )
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"status": "success", "diagnostics": data}), "utf-8")
        loaded = load_report(str(path))
        assert loaded.termination.reason == TerminationReason.DEADLINE
        assert loaded.findings == ()  # analysis is not persisted, it is recomputed

    def test_cli_text_json_and_exit_codes(self, tmp_path, capsys):
        path = tmp_path / "r.json"

        path.write_text(json.dumps(HEALTHY.to_dict()), "utf-8")
        assert cli_main([str(path)]) == 0
        assert "**Findings**: none" in capsys.readouterr().out

        warn = make_report(termination=TerminationReason.TOOL_BUDGET)
        path.write_text(json.dumps(warn.to_dict()), "utf-8")
        assert cli_main([str(path), "--json"]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["findings"][0]["code"] == "TOOL_BUDGET_EXHAUSTED"
        assert payload["recommendations"]

        crit = make_report(
            counters={"recall_errors": 3, "dedup_banners": 3, "emergency_count": 1},
            coverage={"unprocessed": ["/secret/x"], "resources": ["/secret/x"]},
        )
        path.write_text(json.dumps(crit.to_dict()), "utf-8")
        assert cli_main([str(path), "--redact"]) == 2
        out = capsys.readouterr().out
        assert "LOOP_RECALL_DEADLOCK" in out
        assert "/secret/x" not in out


# --------------------------------------------------------------------------- #
# End-to-end: findings land on AgentResult                                     #
# --------------------------------------------------------------------------- #


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_findings_attached_to_result_and_display(self):
        agent = _agent(AgentConfig(execution_mode="standard", max_tool_calls=3))
        await agent.initialize()
        result = await agent.execute(Task(id="t", objective="say done"))

        report = result.diagnostics
        assert isinstance(report, RunReport)
        assert isinstance(report.findings, tuple)
        assert report.termination.reason == TerminationReason.COMPLETED
        # Healthy scripted run → no loop/budget findings.
        assert not any(f.code.startswith("LOOP_") for f in report.findings)
        assert "Ended  : completed" in result.display()
        # display() only mentions findings when there are any.
        if report.findings:
            assert "Diag   :" in result.display()

    @pytest.mark.asyncio
    async def test_findings_reflect_recorded_signals(self, monkeypatch):
        agent = _agent(AgentConfig(execution_mode="standard"))
        await agent.initialize()

        # Inject signals the way the loops do, via the live recorder, right
        # after execute() creates it.
        original = Agent._setup_execution

        async def setup_and_poison(self, *args, **kwargs):
            out = await original(self, *args, **kwargs)
            rec = self._run_recorder
            for _ in range(3):
                rec.record_no_progress_round()
                rec.record_empty_response()
            return out

        monkeypatch.setattr(Agent, "_setup_execution", setup_and_poison)
        result = await agent.execute(Task(id="t", objective="x"))
        assert result.diagnostics is not None, result.error
        codes_found = [f.code for f in result.diagnostics.findings]
        assert codes_found == ["LOOP_NO_PROGRESS", "EMPTY_RESPONSES"]
        assert result.diagnostics.recommendations
        assert "LOOP_NO_PROGRESS" in result.display()
        assert "LOOP_NO_PROGRESS" in result.diagnostics.explain()
