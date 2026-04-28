"""F6 — Critic legacy single-call must route through
``BaseExecutionMode.call_llm``.

Before F6, ``Critic._run_single_call`` invoked ``agent.llm.call`` directly,
bypassing the framework's shared LLM entry point.  That meant:

* ``ContextEngine.prepare`` / ``post_response`` never fired on the
  Critic's slice of the conversation (so context masking/compaction was
  silently disabled for the Critic).
* The ``UsageTracker`` never saw Critic calls with
  ``CallPurpose.CRITIC`` — Critic traffic was counted as untagged
  "other" in run summaries.
* The ``DefaultExecutionTracer`` never recorded Critic ``LLMCallRecord``
  entries, so observability dashboards were missing Critic rounds.
* The plugin pipeline (``before_model`` / ``wrap_model_call`` /
  ``after_model``) never ran for Critic calls, so guardrail / PII /
  retry plugins did not apply.

These tests pin down the fix: every Critic LLM call — whether through
the new ``CriticRunner`` (``build_verification_prompt`` path) or the
legacy ``review_step`` / ``review_final`` API — now goes through
``BaseExecutionMode.call_llm`` and therefore lights up the same
observability / plugin / context-management hooks as the Generator and
Refiner.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq.agents.components.critic import Critic, Verdict
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.agents.plan import PlanStep
from nucleusiq.agents.usage.usage_tracker import CallPurpose

# ------------------------------------------------------------------ #
# Lightweight fakes (no MagicMock so we can assert on calls cleanly)  #
# ------------------------------------------------------------------ #


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.model = "test-model"
        self.usage = None


class _RecordingContextEngine:
    """Records prepare/post_response calls; returns messages unchanged."""

    def __init__(self) -> None:
        self.prepare_calls: list[list[Any]] = []
        self.post_response_calls: list[list[Any]] = []
        self.store = None

    async def prepare(self, messages: list[Any]) -> list[Any]:
        self.prepare_calls.append(list(messages))
        return messages

    def post_response(self, messages: list[Any]) -> list[Any]:
        self.post_response_calls.append(list(messages))
        return messages


class _RecordingUsageTracker:
    """Minimal stand-in that captures (purpose, response) pairs."""

    def __init__(self) -> None:
        self.recorded: list[tuple[CallPurpose, Any]] = []

    def record_from_response(self, purpose: CallPurpose, response: Any) -> None:
        self.recorded.append((purpose, response))


class _RecordingTracer:
    """Captures LLM call records; mimics the subset ``call_llm`` uses."""

    def __init__(self) -> None:
        self.llm_calls: list[Any] = []
        self.tool_calls: list[Any] = []

    def record_llm_call(self, record: Any) -> None:
        self.llm_calls.append(record)


def _make_agent_with_hooks(
    llm_content: str,
    *,
    with_engine: bool = True,
    with_tracker: bool = True,
    with_tracer: bool = True,
) -> MagicMock:
    """Build an agent mock that actually wires the observability hooks.

    The standard ``test_critic_refiner`` helper sets those hooks to
    ``None`` for unit-test simplicity.  F6 specifically needs to verify
    they fire, so we wire real recorders here.
    """
    agent = MagicMock()
    agent.name = "critic-test-agent"
    agent._logger = MagicMock()
    agent.config = AgentConfig()
    agent.tools = []
    agent._current_llm_overrides = {}

    llm = MagicMock()
    llm.model_name = "test-model"
    llm.call = AsyncMock(return_value=_FakeLLMResponse(llm_content))
    llm.convert_tool_specs = MagicMock(return_value=[])
    agent.llm = llm

    agent._executor = MagicMock()
    agent._plugin_manager = None  # F6 contract tested separately.
    agent._context_engine = _RecordingContextEngine() if with_engine else None
    agent._usage_tracker = _RecordingUsageTracker() if with_tracker else None
    agent._tracer = _RecordingTracer() if with_tracer else None
    return agent


def _pass_response_json() -> str:
    return json.dumps(
        {
            "verdict": "pass",
            "score": 0.9,
            "feedback": "Looks good",
            "issues": [],
            "suggestions": [],
        }
    )


def _make_step() -> PlanStep:
    return PlanStep(step=1, action="execute", details="step")


# ------------------------------------------------------------------ #
# The actual F6 contracts                                             #
# ------------------------------------------------------------------ #


class TestCriticRecordsUsageAsCriticPurpose:
    """Usage tracker must see Critic traffic tagged ``CallPurpose.CRITIC``."""

    @pytest.mark.asyncio
    async def test_review_step_tags_usage_as_critic(self) -> None:
        agent = _make_agent_with_hooks(_pass_response_json())
        critic = Critic()

        result = await critic.review_step(agent, "Add 2 + 3", _make_step(), "5", {})

        assert result.verdict == Verdict.PASS
        recorded = agent._usage_tracker.recorded
        assert len(recorded) == 1, "Critic must record exactly one LLM call"
        purpose, response = recorded[0]
        assert purpose is CallPurpose.CRITIC, (
            "Legacy Critic path must tag usage as CallPurpose.CRITIC so "
            "Critic tokens don't leak into untagged 'other' traffic."
        )
        assert response is not None

    @pytest.mark.asyncio
    async def test_review_final_tags_usage_as_critic(self) -> None:
        agent = _make_agent_with_hooks(_pass_response_json())
        critic = Critic()

        await critic.review_final(
            agent,
            task_objective="Summarise the meeting notes",
            messages=[{"role": "assistant", "content": "Summary here."}],
            final_result="Summary here.",
        )

        purposes = [p for p, _ in agent._usage_tracker.recorded]
        assert purposes == [CallPurpose.CRITIC]


class TestCriticHitsContextEngine:
    """``ContextEngine.prepare`` and ``post_response`` must fire."""

    @pytest.mark.asyncio
    async def test_engine_prepare_and_post_response_fire(self) -> None:
        agent = _make_agent_with_hooks(_pass_response_json())
        critic = Critic()

        await critic.review_step(agent, "Task", _make_step(), "result", {})

        engine: _RecordingContextEngine = agent._context_engine
        assert len(engine.prepare_calls) == 1, (
            "ContextEngine.prepare must run before the Critic LLM call "
            "so masked/compacted context applies to verification too."
        )
        assert len(engine.post_response_calls) == 1, (
            "ContextEngine.post_response must run after so observation "
            "masking fires on the Critic's conversation slice."
        )

    @pytest.mark.asyncio
    async def test_no_engine_configured_does_not_crash(self) -> None:
        agent = _make_agent_with_hooks(_pass_response_json(), with_engine=False)
        critic = Critic()

        result = await critic.review_step(agent, "Task", _make_step(), "result", {})
        assert result.verdict == Verdict.PASS


class TestCriticRecordsOnTracer:
    """Tracer must see one LLM record per Critic call."""

    @pytest.mark.asyncio
    async def test_tracer_records_llm_call(self) -> None:
        agent = _make_agent_with_hooks(_pass_response_json())
        critic = Critic()

        await critic.review_step(agent, "Task", _make_step(), "result", {})

        tracer: _RecordingTracer = agent._tracer
        assert len(tracer.llm_calls) == 1, (
            "Critic call must appear in tracer.llm_calls so observability "
            "dashboards can count Critic rounds."
        )
        record = tracer.llm_calls[0]
        purpose = getattr(record, "purpose", None) or (
            record.get("purpose") if isinstance(record, dict) else None
        )
        assert purpose == "critic", (
            f"Critic LLM record should have purpose='critic', got {purpose!r}"
        )


class TestCriticStillCallsUnderlyingLLM:
    """The routing change must not skip the actual ``agent.llm.call``."""

    @pytest.mark.asyncio
    async def test_llm_call_invoked_exactly_once(self) -> None:
        agent = _make_agent_with_hooks(_pass_response_json())
        critic = Critic()

        await critic.review_step(agent, "Task", _make_step(), "result", {})

        assert agent.llm.call.await_count == 1, (
            "BaseExecutionMode.call_llm must still reach agent.llm.call — "
            "F6 adds hooks around it, it does not replace it."
        )

    @pytest.mark.asyncio
    async def test_llm_call_kwargs_preserved(self) -> None:
        agent = _make_agent_with_hooks(_pass_response_json())
        agent._current_llm_overrides = {"temperature": 0.1}
        critic = Critic()

        await critic.review_step(agent, "Task", _make_step(), "result", {})

        call_args = agent.llm.call.await_args
        assert call_args is not None
        kwargs = call_args.kwargs
        assert kwargs.get("model") == "test-model"
        assert kwargs.get("max_output_tokens") == AgentConfig().llm_max_output_tokens
        assert kwargs.get("temperature") == 0.1, (
            "agent._current_llm_overrides must still propagate to the LLM."
        )
        assert "messages" in kwargs


class TestCriticErrorHandlingUnchanged:
    """Non-fatal Critic errors must still produce UNCERTAIN, not raise."""

    @pytest.mark.asyncio
    async def test_llm_failure_returns_uncertain(self) -> None:
        agent = _make_agent_with_hooks(_pass_response_json())
        agent.llm.call = AsyncMock(side_effect=RuntimeError("boom"))
        critic = Critic()

        result = await critic.review_step(agent, "Task", _make_step(), "result", {})
        assert result.verdict == Verdict.UNCERTAIN
        assert "boom" in result.feedback
