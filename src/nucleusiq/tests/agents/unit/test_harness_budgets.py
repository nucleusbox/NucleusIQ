"""Autonomous harness hardening — PR C: budgets, wall clock, overflow, preflight.

Covers (docs/design/AUTONOMOUS_HARNESS_HARDENING.md WS-1 / WS-6):

* ``BudgetResolver`` — every hand-off cap derives from the window;
* ``ContextConfig.resolve_response_reserve`` — 128K keeps 8192, 8K shrinks,
  explicit wins, large replies grow the reserve;
* ``ContextEngine.budgets`` / ``resolved_response_reserve`` /
  ``force_emergency``;
* wall clock (``max_execution_time``) — graceful stop at loop boundaries
  with a synthesized answer, SimpleRunner returns the best candidate,
  children inherit the remaining seconds;
* ``llm_call_timeout`` / ``step_timeout`` — enforced only when explicit;
* ``ContextLengthError`` recovery — emergency compaction, then
  ``max_output_tokens`` shrink, then a precise ``context_overflow``;
* preflight fitness — Autonomous downgrades to STANDARD below the 16K
  working-token floor unless ``preflight_downgrade=False``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.agent_result import ResultStatus
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.decomposer import Decomposer
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.context.budgets import (
    FALLBACK_WINDOW,
    BudgetResolver,
    budgets_for,
)
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.modes.loop_guards import (
    deadline_exceeded,
    explicit_timeout,
    remaining_seconds,
)
from nucleusiq.agents.task import Task
from nucleusiq.llms.errors import ContextLengthError
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools import BaseTool

from nucleusiq.tests.conftest import make_test_prompt

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


class ScriptedLLM(MockLLM):
    """Replays scripted responses; ``("raise", exc)`` raises, ``("sleep", s)`` stalls."""

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
        if kind == "raise":
            raise payload
        if kind == "sleep":
            await asyncio.sleep(payload)
            return self.LLMResponse([self.Choice(self.Message(content="late"))])
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


class HookTool(BaseTool):
    """Idempotent tool that runs ``hook()`` and returns a fixed payload."""

    def __init__(self, hook=None, *, delay: float = 0.0, payload: str = "42"):
        super().__init__(name="lookup", description="Look up", idempotent=True)
        self._hook = hook
        self._delay = delay
        self._payload = payload

    async def initialize(self) -> None:
        pass

    async def execute(self, key: str) -> str:
        if self._hook is not None:
            self._hook()
        if self._delay:
            await asyncio.sleep(self._delay)
        return f"{self._payload} for {key}"

    def get_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        }


def _agent(llm: MockLLM, config: AgentConfig, *, tools=None) -> Agent:
    return Agent(
        name="Harness",
        role="Analyst",
        objective="Answer",
        prompt=make_test_prompt(),
        llm=llm,
        tools=tools or [],
        config=config,
    )


def _std(**overrides: Any) -> AgentConfig:
    base = {"execution_mode": ExecutionMode.STANDARD, "max_tool_calls": 5}
    base.update(overrides)
    return AgentConfig(**base)


def _auto(**overrides: Any) -> AgentConfig:
    base = {
        "execution_mode": ExecutionMode.AUTONOMOUS,
        "max_tool_calls": 5,
        "max_retries": 3,
        "enable_decomposition": False,
    }
    base.update(overrides)
    return AgentConfig(**base)


def _resolver(window: int, *, reserve: int | None = None, **kw: Any) -> BudgetResolver:
    cfg = ContextConfig(max_context_tokens=window)
    return BudgetResolver.from_config(
        cfg,
        window=window,
        working_budget=ContextConfig.resolve_optimal_budget(cfg, window),
        response_reserve=reserve
        if reserve is not None
        else ContextConfig.resolve_response_reserve(cfg, window, 2048),
        **kw,
    )


# --------------------------------------------------------------------------- #
# BudgetResolver — pure math                                                   #
# --------------------------------------------------------------------------- #


class TestBudgetResolver:
    def test_prompt_budget_and_usable_tokens(self):
        r = BudgetResolver.from_config(
            None,
            window=65_536,
            working_budget=45_000,
            response_reserve=4_096,
            system_tokens=1_500,
            tool_schema_tokens=6_000,
        )
        assert r.prompt_budget == 45_000 - 4_096
        assert r.usable_tokens == 65_536 - 4_096 - 1_500 - 6_000

    def test_prompt_budget_never_exceeds_window(self):
        r = BudgetResolver.from_config(
            None, window=8_000, working_budget=100_000, response_reserve=1_000
        )
        assert r.prompt_budget == 7_000

    def test_ceiling_dominates_on_large_window(self):
        r = _resolver(128_000)
        assert r.handoff_chars("synthesis_finding") == 40_000
        assert r.handoff_chars("refiner_candidate") == 32_000
        assert r.handoff_chars("refiner_tool_summary") == 16_000
        assert r.handoff_chars("validation_result") == 20_000
        assert r.handoff_chars("gap_summary") == 4_000

    def test_caps_shrink_proportionally_on_small_window(self):
        big = _resolver(128_000)
        mid = _resolver(65_536)
        small = _resolver(8_000)
        for role in ("synthesis_finding", "refiner_candidate", "validation_result"):
            assert big.handoff_chars(role) >= mid.handoff_chars(role)
            assert mid.handoff_chars(role) > small.handoff_chars(role)

    def test_floor_holds_on_tiny_window(self):
        r = _resolver(4_000)
        assert r.handoff_chars("refiner_candidate") == 2_000
        assert r.handoff_chars("synthesis_finding") == 1_000
        assert r.handoff_chars("validation_task") == 500

    def test_items_share_the_role_budget(self):
        r = _resolver(65_536)
        one = r.handoff_chars("synthesis_finding", items=1)
        three = r.handoff_chars("synthesis_finding", items=3)
        nine = r.handoff_chars("synthesis_finding", items=9)
        assert one >= three >= nine
        # Three children on a 65K model each hand over far more than the
        # legacy 2 000 chars — the office failure mode.
        assert three > 2_000 * 4

    def test_clamp_is_clamp_down_only(self):
        r = _resolver(128_000)
        assert r.clamp("critic_claimed_answer", 10_000) == 10_000
        assert r.clamp("critic_claimed_answer", 1_000_000) == 50_000
        small = _resolver(8_000)
        assert small.clamp("critic_claimed_answer", 10_000) < 10_000

    def test_unknown_role_raises(self):
        with pytest.raises(KeyError):
            _resolver(128_000).handoff_chars("nope")

    def test_with_fixed_costs_is_immutable_copy(self):
        r = _resolver(128_000)
        r2 = r.with_fixed_costs(system_tokens=900, tool_schema_tokens=3_000)
        assert r.system_tokens == 0
        assert r2.system_tokens == 900 and r2.tool_schema_tokens == 3_000
        assert r2.usable_tokens == r.usable_tokens - 3_900

    def test_to_dict_is_json_serialisable(self):
        d = _resolver(65_536).to_dict()
        json.dumps(d)
        assert set(d) >= {
            "window",
            "prompt_budget",
            "usable_tokens",
            "response_reserve",
        }


class TestBudgetsFor:
    def test_engine_backed_when_available(self):
        engine = ContextEngine(config=ContextConfig(), max_tokens=64_000)
        agent = MagicMock()
        agent._context_engine = engine
        assert budgets_for(agent) == engine.budgets
        assert budgets_for(agent).window == 64_000

    def test_bare_double_uses_declared_or_fallback_window(self):
        agent = MagicMock()
        agent._context_engine = None
        agent.config = AgentConfig()
        agent.llm = None
        r = budgets_for(agent)
        assert r.window == FALLBACK_WINDOW
        assert r.handoff_chars("refiner_candidate") == 32_000

    def test_bare_double_respects_config_window(self):
        agent = MagicMock()
        agent._context_engine = None
        agent.config = AgentConfig(context=ContextConfig(max_context_tokens=8_000))
        r = budgets_for(agent)
        assert r.window == 8_000
        assert r.response_reserve < 8_000


# --------------------------------------------------------------------------- #
# response_reserve resolution                                                  #
# --------------------------------------------------------------------------- #


class TestResolveResponseReserve:
    def test_128k_default_is_unchanged(self):
        assert (
            ContextConfig.resolve_response_reserve(ContextConfig(), 128_000, 2048)
            == 8192
        )

    def test_65k_keeps_default_when_it_fits(self):
        assert (
            ContextConfig.resolve_response_reserve(ContextConfig(), 65_536, 4096)
            == 8192
        )

    def test_8k_derives_a_real_budget(self):
        reserve = ContextConfig.resolve_response_reserve(ContextConfig(), 8_000, 2048)
        assert reserve == 2560
        assert reserve < 8_000 // 2

    def test_explicit_wins(self):
        cfg = ContextConfig(response_reserve=3_000)
        assert ContextConfig.resolve_response_reserve(cfg, 8_000, 2048) == 3_000
        assert ContextConfig.resolve_response_reserve(cfg, 128_000, 2048) == 3_000

    def test_large_reply_grows_reserve(self):
        reserve = ContextConfig.resolve_response_reserve(
            ContextConfig(), 128_000, 16_000
        )
        assert reserve >= 16_000 + 512

    def test_never_more_than_half_the_window(self):
        reserve = ContextConfig.resolve_response_reserve(ContextConfig(), 3_000, 2_900)
        assert reserve == 1_500


class TestEngineBudgets:
    def test_engine_exposes_budgets_and_reserve(self):
        engine = ContextEngine(
            config=ContextConfig(max_context_tokens=8_000),
            max_tokens=8_000,
            max_output_tokens=1_024,
        )
        assert engine.resolved_response_reserve == 1_536
        assert engine.budgets.window == 8_000
        assert engine.budgets.response_reserve == 1_536

    def test_set_fixed_costs_reduces_usable(self):
        engine = ContextEngine(config=ContextConfig(), max_tokens=128_000)
        before = engine.budgets.usable_tokens
        engine.set_fixed_costs(system_tokens=1_000, tool_schema_tokens=5_000)
        assert engine.budgets.usable_tokens == before - 6_000

    @pytest.mark.asyncio
    async def test_force_emergency_shrinks_transcript(self):
        engine = ContextEngine(
            config=ContextConfig(max_context_tokens=16_000),
            max_tokens=16_000,
        )
        messages = [ChatMessage(role="system", content="sys")]
        for i in range(12):
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[{"id": f"c{i}", "name": "lookup", "arguments": "{}"}],
                )
            )
            messages.append(
                ChatMessage(role="tool", tool_call_id=f"c{i}", content="x " * 3_000)
            )
        before = engine.token_counter.count_messages(messages)
        reduced = await engine.force_emergency(messages)
        after = engine.token_counter.count_messages(reduced)
        assert after < before
        assert engine.emergency_count >= 1


# --------------------------------------------------------------------------- #
# Loop guards — wall clock helpers                                             #
# --------------------------------------------------------------------------- #


class TestWallClockHelpers:
    def test_remaining_seconds_none_without_deadline(self):
        agent = MagicMock()
        agent._run_deadline = None
        assert remaining_seconds(agent) is None
        assert deadline_exceeded(agent) is False

    def test_deadline_in_past_is_exceeded(self):
        agent = MagicMock()
        agent._run_deadline = time.monotonic() - 1
        assert deadline_exceeded(agent)
        assert remaining_seconds(agent) <= 0

    def test_explicit_timeout_only_when_set(self):
        assert explicit_timeout(AgentConfig(), "llm_call_timeout") is None
        assert explicit_timeout(AgentConfig(), "step_timeout") is None
        assert (
            explicit_timeout(AgentConfig(llm_call_timeout=7), "llm_call_timeout") == 7
        )
        assert explicit_timeout(AgentConfig(step_timeout=3), "step_timeout") == 3
        assert explicit_timeout(None, "step_timeout") is None


# --------------------------------------------------------------------------- #
# Wall clock in the loops                                                      #
# --------------------------------------------------------------------------- #


class TestWallClock:
    @pytest.mark.asyncio
    async def test_standard_loop_stops_gracefully_with_synthesis(self):
        holder: dict[str, Agent] = {}

        def expire() -> None:
            holder["agent"]._run_deadline = time.monotonic() - 1

        llm = ScriptedLLM(
            [
                ("tool", [("lookup", {"key": "a"})]),
                ("text", "Synthesized: 42"),
            ]
        )
        agent = _agent(llm, _std(max_execution_time=600), tools=[HookTool(expire)])
        holder["agent"] = agent
        result = await agent.execute(Task(id="t", objective="Find 42"))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == "Synthesized: 42"
        assert result.termination_reason == "deadline"
        # One tool round, then the tools-free synthesis call — no second tool round.
        assert llm.calls == 2
        assert not llm.requests[-1].get("tools")
        assert "max_execution_time=600s" in result.diagnostics.termination.message

    @pytest.mark.asyncio
    async def test_unlimited_when_zero(self):
        llm = ScriptedLLM([("text", "done")])
        agent = _agent(llm, _std(max_execution_time=0))
        result = await agent.execute(Task(id="t", objective="x"))
        assert result.status == ResultStatus.SUCCESS
        assert agent._run_deadline is None

    @pytest.mark.asyncio
    async def test_simple_runner_returns_best_candidate_on_deadline(self):
        """Critic FAILs attempt 1; the clock runs out → best candidate, not abstain."""
        holder: dict[str, Agent] = {}

        class ExpiringLLM(ScriptedLLM):
            async def call(self, **kwargs: Any):
                resp = await super().call(**kwargs)
                if self.calls == 2:  # right after the Critic verdict
                    holder["agent"]._run_deadline = time.monotonic() - 1
                return resp

        llm = ExpiringLLM(
            [
                ("text", "Answer v1"),
                ("text", "VERDICT: FAIL\nSCORE: 0.2\nFEEDBACK: incomplete"),
                ("text", "Answer v2 (should not be requested)"),
            ]
        )
        agent = _agent(llm, _auto(max_execution_time=600))
        holder["agent"] = agent
        result = await agent.execute(Task(id="t", objective="Answer"))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == "Answer v1"
        assert result.termination_reason == "deadline"
        assert llm.calls == 2

    @pytest.mark.asyncio
    async def test_children_inherit_remaining_deadline_and_timeouts(self):
        agent = _agent(
            ScriptedLLM([("text", "x")]),
            _auto(max_execution_time=600, llm_call_timeout=45, step_timeout=20),
        )
        await agent.initialize()
        agent._run_deadline = time.monotonic() + 100
        child = await Decomposer.create_sub_agent(agent, "s1", "Sub task")
        assert child is not None
        assert 1 <= child.config.max_execution_time <= 100
        assert child.config.llm_call_timeout == 45
        assert child.config.step_timeout == 20
        assert "llm_call_timeout" in child.config.model_fields_set

    @pytest.mark.asyncio
    async def test_children_do_not_inherit_unset_timeouts(self):
        agent = _agent(ScriptedLLM([("text", "x")]), _auto())
        await agent.initialize()
        child = await Decomposer.create_sub_agent(agent, "s1", "Sub task")
        assert child is not None
        assert "llm_call_timeout" not in child.config.model_fields_set
        assert "step_timeout" not in child.config.model_fields_set


# --------------------------------------------------------------------------- #
# Opt-in timeouts                                                              #
# --------------------------------------------------------------------------- #


class TestTimeouts:
    @pytest.mark.asyncio
    async def test_llm_call_timeout_enforced_when_explicit(self):
        llm = ScriptedLLM([("sleep", 5)])
        agent = _agent(llm, _std(llm_call_timeout=1))
        t0 = time.monotonic()
        result = await agent.execute(Task(id="t", objective="x"))
        assert time.monotonic() - t0 < 4
        assert result.status == ResultStatus.ERROR
        assert "llm_call_timeout=1s" in (result.error or "")
        assert result.termination_reason == "llm_timeout"

    @pytest.mark.asyncio
    async def test_llm_call_timeout_not_enforced_by_default(self):
        llm = ScriptedLLM([("sleep", 0.05)])
        agent = _agent(llm, _std())
        result = await agent.execute(Task(id="t", objective="x"))
        assert result.status == ResultStatus.SUCCESS
        assert result.output == "late"

    @pytest.mark.asyncio
    async def test_step_timeout_turns_slow_tool_into_evidence(self):
        llm = ScriptedLLM(
            [
                ("tool", [("lookup", {"key": "a"})]),
                ("text", "Proceeded without the tool"),
            ]
        )
        agent = _agent(llm, _std(step_timeout=1), tools=[HookTool(delay=5)])
        t0 = time.monotonic()
        result = await agent.execute(Task(id="t", objective="x"))
        assert time.monotonic() - t0 < 4
        assert result.status == ResultStatus.SUCCESS
        assert result.output == "Proceeded without the tool"
        tool_msgs = [m for m in llm.requests[-1]["messages"] if m.get("role") == "tool"]
        assert (
            tool_msgs
            and "timed out after 1s (step_timeout)" in tool_msgs[-1]["content"]
        )
        assert result.diagnostics.counters.tool_errors >= 1


# --------------------------------------------------------------------------- #
# ContextLengthError recovery                                                  #
# --------------------------------------------------------------------------- #


class TestOverflowRecovery:
    @pytest.mark.asyncio
    async def test_shrinks_max_output_tokens_and_retries(self):
        llm = ScriptedLLM(
            [
                ("raise", ContextLengthError("prompt too long")),
                ("text", "fits now"),
            ],
            context_window=3_000,
        )
        agent = _agent(
            llm,
            _std(
                llm_max_output_tokens=2_900,
                context=ContextConfig(max_context_tokens=3_000),
            ),
        )
        result = await agent.execute(Task(id="t", objective="x"))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == "fits now"
        assert llm.calls == 2
        assert (
            llm.requests[1]["max_output_tokens"] < llm.requests[0]["max_output_tokens"]
        )
        kinds = [e.kind for e in result.diagnostics.timeline]
        assert "context_overflow_recovery" in kinds

    @pytest.mark.asyncio
    async def test_unrecoverable_overflow_is_precise_and_sticky(self):
        llm = ScriptedLLM([("raise", ContextLengthError("prompt too long"))])
        agent = _agent(llm, _std())
        result = await agent.execute(Task(id="t", objective="x"))

        assert result.status == ResultStatus.ERROR
        assert "prompt too long" in (result.error or "")
        assert result.termination_reason == "context_overflow"
        assert "window=128000" in result.diagnostics.termination.message

    @pytest.mark.asyncio
    async def test_emergency_compaction_path(self):
        """A long tool transcript is compacted before the retry."""
        big = "evidence " * 2_000
        llm = ScriptedLLM(
            [
                ("tool", [("lookup", {"key": "a"})]),
                ("tool", [("lookup", {"key": "b"})]),
                ("tool", [("lookup", {"key": "c"})]),
                ("raise", ContextLengthError("prompt too long")),
                ("text", "recovered"),
            ],
            context_window=40_000,
        )
        agent = _agent(
            llm,
            _std(
                max_tool_calls=10,
                context=ContextConfig(max_context_tokens=40_000),
            ),
            tools=[HookTool(payload=big)],
        )
        result = await agent.execute(Task(id="t", objective="x"))

        assert result.status == ResultStatus.SUCCESS
        assert result.output == "recovered"
        assert result.diagnostics.counters.emergency_count >= 1
        kinds = [e.kind for e in result.diagnostics.timeline]
        assert "context_overflow_recovery" in kinds


# --------------------------------------------------------------------------- #
# Preflight fitness                                                            #
# --------------------------------------------------------------------------- #


class TestPreflight:
    @pytest.mark.asyncio
    async def test_autonomous_downgrades_on_small_window(self):
        llm = ScriptedLLM([("text", "done")], context_window=8_000)
        agent = _agent(llm, _auto(context=ContextConfig(max_context_tokens=8_000)))
        result = await agent.execute(Task(id="t", objective="x"))

        assert result.status == ResultStatus.SUCCESS
        assert result.mode == "standard"
        pf = result.diagnostics.decisions["preflight"]
        assert pf["mode_requested"] == "autonomous"
        assert pf["fitness"] == "unfit"
        assert pf["action"] == "downgraded_to_standard"
        assert pf["working_tokens"] < 16_000
        assert result.diagnostics.config_resolved["preflight"]["action"] == (
            "downgraded_to_standard"
        )
        kinds = [e.kind for e in result.diagnostics.timeline]
        assert "preflight_downgraded" in kinds
        # No Critic call happened — one LLM call total.
        assert llm.calls == 1

    @pytest.mark.asyncio
    async def test_autonomous_forced_when_downgrade_disabled(self):
        llm = ScriptedLLM(
            [
                ("text", "done"),
                ("text", "VERDICT: PASS\nSCORE: 0.9\nFEEDBACK: ok"),
            ],
            context_window=8_000,
        )
        agent = _agent(
            llm,
            _auto(
                context=ContextConfig(max_context_tokens=8_000),
                preflight_downgrade=False,
            ),
        )
        result = await agent.execute(Task(id="t", objective="x"))
        assert result.status == ResultStatus.SUCCESS
        assert result.mode == "autonomous"
        pf = result.diagnostics.decisions["preflight"]
        assert pf["fitness"] == "unfit" and pf["action"] == "forced"

    @pytest.mark.asyncio
    async def test_autonomous_marginal_between_floors(self):
        llm = ScriptedLLM(
            [
                ("text", "done"),
                ("text", "VERDICT: PASS\nSCORE: 0.9\nFEEDBACK: ok"),
            ],
            context_window=32_000,
        )
        agent = _agent(llm, _auto(context=ContextConfig(max_context_tokens=32_000)))
        result = await agent.execute(Task(id="t", objective="x"))
        assert result.mode == "autonomous"
        pf = result.diagnostics.decisions["preflight"]
        assert pf["fitness"] == "marginal" and pf["action"] == "none"

    @pytest.mark.asyncio
    async def test_large_window_is_fit(self):
        llm = ScriptedLLM(
            [
                ("text", "done"),
                ("text", "VERDICT: PASS\nSCORE: 0.9\nFEEDBACK: ok"),
            ]
        )
        agent = _agent(llm, _auto())
        result = await agent.execute(Task(id="t", objective="x"))
        pf = result.diagnostics.decisions["preflight"]
        assert pf["fitness"] == "ok"
        assert pf["window_is_fallback"] is False
        assert pf["tool_schema_tokens"] >= 0

    @pytest.mark.asyncio
    async def test_tool_schemas_count_against_working_budget(self):
        tools = [HookTool(payload=str(i)) for i in range(1)]
        llm = ScriptedLLM([("text", "done")])
        agent = _agent(llm, _std(), tools=tools)
        result = await agent.execute(Task(id="t", objective="x"))
        pf = result.diagnostics.decisions["preflight"]
        assert pf["tool_schema_tokens"] > 0
        assert pf["working_tokens"] == (
            pf["window"]
            - pf["response_reserve"]
            - pf["system_tokens"]
            - pf["tool_schema_tokens"]
        )

    @pytest.mark.asyncio
    async def test_undeclared_window_is_flagged(self):
        from nucleusiq.llms.base_llm import BaseLLM

        class BareLLM(ScriptedLLM):
            # Provider that never overrode the BaseLLM constant.
            get_context_window = BaseLLM.get_context_window

        llm = BareLLM([("text", "done")])
        agent = _agent(llm, _std())
        result = await agent.execute(Task(id="t", objective="x"))
        assert result.diagnostics.config_resolved["window_is_fallback"] is True
        assert result.diagnostics.decisions["preflight"]["window_is_fallback"] is True

    @pytest.mark.asyncio
    async def test_report_carries_budgets(self):
        llm = ScriptedLLM([("text", "done")])
        agent = _agent(llm, _std())
        result = await agent.execute(Task(id="t", objective="x"))
        cfg = result.diagnostics.config_resolved
        assert cfg["response_reserve"] == 8192
        assert cfg["budgets"]["window"] == 128_000
        assert cfg["max_execution_time"] == 3600
        assert "llm_call_timeout" not in cfg
