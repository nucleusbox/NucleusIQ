"""
Comprehensive tests for the Critic and Refiner components.

Tests the Generate → Verify → Revise architecture:
- Critic (Verifier): CritiqueResult model, prompt construction, LLM parsing
- Refiner (Reviser): directed revision, tool re-execution, LLM revision
- AutonomousMode: full loop integration
"""

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import (
    Critic,
    CritiqueResult,
    Verdict,
    _truncate,
)
from nucleusiq.agents.config.agent_config import AgentConfig, AgentState
from nucleusiq.agents.plan import Plan, PlanStep
from nucleusiq.agents.task import Task

# ================================================================== #
# Test helpers                                                        #
# ================================================================== #


def _make_task(objective: str = "Test task") -> Task:
    return Task(id="test-1", objective=objective)


def _make_step(
    step_num: int = 1,
    action: str = "execute",
    details: str = "Test step",
    args: Dict[str, Any] | None = None,
) -> PlanStep:
    return PlanStep(
        step=step_num,
        action=action,
        details=details,
        args=args,
    )


def _make_plan(steps: List[PlanStep] | None = None) -> Plan:
    task = _make_task()
    if steps is None:
        steps = [_make_step(1, "execute", "Step 1")]
    return Plan(steps=steps, task=task)


class FakeFunction:
    """Simple mock for tool call function info (avoids MagicMock .name issue)."""

    def __init__(self, name: str, arguments: str = "{}"):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    """Simple mock for a tool call object."""

    def __init__(self, function: FakeFunction):
        self.function = function


class FakeMessage:
    """Simple mock for LLM message."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeLLMResponse:
    """Mimics MockLLM.LLMResponse structure."""

    def __init__(self, content: str | None = None, tool_calls=None):
        msg = FakeMessage(content=content, tool_calls=tool_calls)
        self.choices = [MagicMock(message=msg)]


def _make_agent(
    llm_response: Any = None,
    tools: List | None = None,
    config: AgentConfig | None = None,
    has_llm: bool = True,
) -> MagicMock:
    """Create a mock Agent with configurable LLM response."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent._logger = MagicMock()
    agent.config = config or AgentConfig()
    agent.tools = tools or []
    agent._current_llm_overrides = {}

    if has_llm:
        llm = MagicMock()
        llm.model_name = "test-model"
        if llm_response is not None:
            llm.call = AsyncMock(return_value=llm_response)
        else:
            llm.call = AsyncMock(
                return_value=FakeLLMResponse(content="default response")
            )
        llm.convert_tool_specs = MagicMock(return_value=[])
        agent.llm = llm
    else:
        agent.llm = None

    agent._executor = MagicMock()
    agent._executor.execute = AsyncMock(return_value="tool result")

    # F6 — Critic's legacy single-call path now routes through
    # BaseExecutionMode.call_llm, which introspects these subsystems.
    # Match the real Agent defaults (None) so MagicMock auto-attrs
    # don't accidentally enable the plugin / tracer / engine branches.
    agent._plugin_manager = None
    agent._context_engine = None
    agent._tracer = None
    agent._usage_tracker = None

    return agent


# ================================================================== #
# CritiqueResult Model Tests                                          #
# ================================================================== #


class TestCritiqueResult:
    def test_default_values(self):
        cr = CritiqueResult(verdict=Verdict.PASS)
        assert cr.verdict == Verdict.PASS
        assert cr.score == 0.5
        assert cr.feedback == ""
        assert cr.issues == []
        assert cr.suggestions == []

    def test_full_construction(self):
        cr = CritiqueResult(
            verdict=Verdict.FAIL,
            score=0.2,
            feedback="Result is incorrect",
            issues=["Wrong calculation", "Missing context"],
            suggestions=["Recalculate", "Add context"],
        )
        assert cr.verdict == Verdict.FAIL
        assert cr.score == 0.2
        assert len(cr.issues) == 2
        assert len(cr.suggestions) == 2

    def test_score_bounds(self):
        cr_low = CritiqueResult(verdict=Verdict.PASS, score=0.0)
        assert cr_low.score == 0.0
        cr_high = CritiqueResult(verdict=Verdict.PASS, score=1.0)
        assert cr_high.score == 1.0

    def test_score_out_of_bounds_raises(self):
        with pytest.raises(Exception):
            CritiqueResult(verdict=Verdict.PASS, score=1.5)
        with pytest.raises(Exception):
            CritiqueResult(verdict=Verdict.PASS, score=-0.1)


class TestVerdict:
    def test_enum_values(self):
        assert Verdict.PASS.value == "pass"
        assert Verdict.FAIL.value == "fail"
        assert Verdict.UNCERTAIN.value == "uncertain"

    def test_from_string(self):
        assert Verdict("pass") == Verdict.PASS
        assert Verdict("fail") == Verdict.FAIL
        assert Verdict("uncertain") == Verdict.UNCERTAIN


# ================================================================== #
# Critic Tests                                                         #
# ================================================================== #


class TestCriticReviewStep:
    @pytest.mark.asyncio
    async def test_pass_verdict_from_json(self):
        response = FakeLLMResponse(
            content=json.dumps(
                {
                    "verdict": "pass",
                    "score": 0.95,
                    "feedback": "Result is correct and complete",
                    "issues": [],
                    "suggestions": [],
                }
            )
        )
        agent = _make_agent(llm_response=response)
        critic = Critic()

        result = await critic.review_step(
            agent,
            "Add 2 + 3",
            _make_step(),
            "5",
            {},
        )
        assert result.verdict == Verdict.PASS
        assert result.score == 0.95

    @pytest.mark.asyncio
    async def test_fail_verdict_from_json(self):
        response = FakeLLMResponse(
            content=json.dumps(
                {
                    "verdict": "fail",
                    "score": 0.2,
                    "feedback": "Calculation is wrong",
                    "issues": ["2 + 3 should be 5, got 6"],
                    "suggestions": ["Recalculate the sum"],
                }
            )
        )
        agent = _make_agent(llm_response=response)
        critic = Critic()

        result = await critic.review_step(
            agent,
            "Add 2 + 3",
            _make_step(),
            "6",
            {},
        )
        assert result.verdict == Verdict.FAIL
        assert result.score == 0.2
        assert len(result.issues) == 1
        assert len(result.suggestions) == 1

    @pytest.mark.asyncio
    async def test_uncertain_verdict(self):
        response = FakeLLMResponse(
            content=json.dumps(
                {
                    "verdict": "uncertain",
                    "score": 0.5,
                    "feedback": "Cannot verify this",
                }
            )
        )
        agent = _make_agent(llm_response=response)
        critic = Critic()

        result = await critic.review_step(
            agent,
            "Complex task",
            _make_step(),
            "some result",
            {},
        )
        assert result.verdict == Verdict.UNCERTAIN

    @pytest.mark.asyncio
    async def test_no_llm_returns_auto_pass(self):
        agent = _make_agent(has_llm=False)
        critic = Critic()

        result = await critic.review_step(
            agent,
            "task",
            _make_step(),
            "result",
            {},
        )
        assert result.verdict == Verdict.PASS
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_llm_error_returns_uncertain(self):
        agent = _make_agent()
        agent.llm.call = AsyncMock(side_effect=Exception("LLM API error"))
        critic = Critic()

        result = await critic.review_step(
            agent,
            "task",
            _make_step(),
            "result",
            {},
        )
        assert result.verdict == Verdict.UNCERTAIN
        assert "failed" in result.feedback.lower()

    @pytest.mark.asyncio
    async def test_empty_response_returns_uncertain(self):
        response = FakeLLMResponse(content=None)
        agent = _make_agent(llm_response=response)
        critic = Critic()

        result = await critic.review_step(
            agent,
            "task",
            _make_step(),
            "result",
            {},
        )
        assert result.verdict == Verdict.UNCERTAIN

    @pytest.mark.asyncio
    async def test_context_included_in_prompt(self):
        response = FakeLLMResponse(
            content=json.dumps(
                {
                    "verdict": "pass",
                    "score": 0.9,
                    "feedback": "OK",
                }
            )
        )
        agent = _make_agent(llm_response=response)
        critic = Critic()

        context = {"step_1": "previous result", "step_1_action": "tool1"}
        await critic.review_step(
            agent,
            "task",
            _make_step(2),
            "result",
            context,
        )
        call_args = agent.llm.call.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "previous result" in prompt
        assert "step_1_action" not in prompt


class TestCriticReviewFinal:
    @pytest.mark.asyncio
    async def test_final_pass_with_messages(self):
        """Critic receives conversation messages and returns PASS."""
        response = FakeLLMResponse(
            content=json.dumps(
                {
                    "verdict": "pass",
                    "score": 0.9,
                    "feedback": "Result satisfies the task",
                }
            )
        )
        agent = _make_agent(llm_response=response)
        critic = Critic()

        messages = [
            ChatMessage(role="user", content="Calculate 2+3"),
            ChatMessage(role="assistant", content="5"),
        ]
        result = await critic.review_final(
            agent,
            "Calculate 2+3",
            final_result="5",
            messages=messages,
        )
        assert result.verdict == Verdict.PASS
        call_args = agent.llm.call.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "What the Agent Did" in prompt

    @pytest.mark.asyncio
    async def test_final_pass_legacy_plan(self):
        """Critic falls back to plan-based prompt when no messages."""
        response = FakeLLMResponse(
            content=json.dumps(
                {
                    "verdict": "pass",
                    "score": 0.9,
                    "feedback": "Result satisfies the task",
                }
            )
        )
        agent = _make_agent(llm_response=response)
        critic = Critic()

        plan = _make_plan()
        result = await critic.review_final(
            agent,
            "Calculate 2+3",
            plan=plan,
            results=["5"],
            final_result="5",
        )
        assert result.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_final_no_llm(self):
        agent = _make_agent(has_llm=False)
        critic = Critic()

        result = await critic.review_final(
            agent,
            "task",
            final_result="res",
        )
        assert result.verdict == Verdict.PASS


class TestCriticParsing:
    def test_parse_json_in_markdown_fence(self):
        critic = Critic()
        content = '```json\n{"verdict": "pass", "score": 0.8, "feedback": "OK"}\n```'
        response = FakeLLMResponse(content=content)
        result = critic._parse_response(response)
        assert result.verdict == Verdict.PASS

    def test_parse_plain_json(self):
        critic = Critic()
        response = FakeLLMResponse(content='{"verdict": "fail", "score": 0.3}')
        result = critic._parse_response(response)
        assert result.verdict == Verdict.FAIL

    def test_parse_non_json_with_pass_keyword(self):
        critic = Critic()
        response = FakeLLMResponse(
            content="The result looks good. I would pass this step."
        )
        result = critic._parse_response(response)
        assert result.verdict == Verdict.PASS

    def test_parse_non_json_with_fail_keyword(self):
        critic = Critic()
        response = FakeLLMResponse(
            content="This step has failed. The result is incorrect."
        )
        result = critic._parse_response(response)
        assert result.verdict == Verdict.FAIL

    def test_parse_non_json_ambiguous(self):
        critic = Critic()
        response = FakeLLMResponse(content="I'm not sure what to make of this result.")
        result = critic._parse_response(response)
        assert result.verdict == Verdict.UNCERTAIN

    def test_parse_malformed_json(self):
        critic = Critic()
        response = FakeLLMResponse(content='{"verdict": "pass", broken}')
        result = critic._parse_response(response)
        assert result.verdict in (Verdict.PASS, Verdict.UNCERTAIN)


class TestCriticPromptConstruction:
    def test_step_review_prompt_structure(self):
        step = _make_step(1, "add", "Add two numbers")
        prompt = Critic()._build_step_review_prompt(
            "Calculate 2+3",
            step,
            "5",
            {},
        )
        assert "Original Task" in prompt
        assert "Calculate 2+3" in prompt
        assert "Step 1: add" in prompt
        assert "5" in prompt
        assert "verdict" in prompt

    def test_step_review_prompt_with_context(self):
        step = _make_step(2, "multiply", "Multiply result")
        ctx = {"step_1": "5", "step_1_action": "add"}
        prompt = Critic()._build_step_review_prompt(
            "task",
            step,
            "25",
            ctx,
        )
        assert "step_1" in prompt
        assert "5" in prompt

    def test_final_review_prompt_structure(self):
        plan = _make_plan(
            [
                _make_step(1, "add", "Add numbers"),
                _make_step(2, "format", "Format result"),
            ]
        )
        prompt = Critic()._build_final_review_prompt(
            "Calculate and format",
            plan,
            ["5", "Result: 5"],
            "Result: 5",
        )
        assert "Original Task" in prompt
        assert "Calculate and format" in prompt
        assert "Step 1: add" in prompt
        assert "Step 2: format" in prompt


# ================================================================== #
# Refiner tests — the new, public Refiner.revise() surface            #
# ================================================================== #
# NOTE: Tests for the Reviser role (``Refiner.revise``) live in the   #
# dedicated module ``test_refiner_revise.py``. Pre-F1 tests targeting #
# ``refine_step`` / ``refine_final`` / ``build_revision_message`` and #
# the extract-helpers have been removed along with that legacy API.   #
# ================================================================== #


# ================================================================== #
# AutonomousMode Integration Tests                                     #
# ================================================================== #


class TestAutonomousModeFull:
    """Integration tests for the new Autonomous Mode architecture.

    The autonomous mode is now a thin orchestrator over Standard mode
    with validation pipeline and structured retry.
    """

    @staticmethod
    def _make_auto_agent(
        llm_response=None,
        tools=None,
        config=None,
        has_llm=True,
    ):
        """Create a mock agent with proper string attributes."""
        agent = _make_agent(
            llm_response=llm_response,
            tools=tools,
            config=config,
            has_llm=has_llm,
        )
        agent.prompt = None
        agent.role = "Calculator"
        agent.objective = "Solve tasks"
        agent.memory = None
        agent._resolve_response_format = MagicMock(return_value=None)
        agent._get_structured_output_kwargs = MagicMock(return_value={})
        agent._plugin_manager = None
        agent._execution_progress = None
        return agent

    @pytest.mark.asyncio
    async def test_simple_task_returns_result(self):
        """Standard execution produces result, validation passes → done.

        Uses a PASS Critic stub so the loop exits on attempt 1 rather
        than falling through to F2 abstention.  The ``mode.run`` happy
        path is what this test cares about, not the Critic's own logic.
        """
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        async def mock_call(**kwargs):
            return FakeLLMResponse(content="42")

        agent = self._make_auto_agent()
        agent.llm.call = AsyncMock(side_effect=mock_call)

        mode = AutonomousMode()
        task = _make_task("What is 6*7?")

        with patch.object(
            AutonomousMode,
            "_run_critic",
            new=AsyncMock(return_value=CritiqueResult(verdict=Verdict.PASS, score=1.0)),
        ):
            result = await mode.run(agent, task)

        assert "42" in str(result)
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_result_returned_after_max_retries(self):
        """After max retries with the Critic still refusing, the mode
        now raises ``AbstentionSignal`` (F2) instead of silently
        returning the last candidate.  The signal still carries the
        best candidate for inspection."""
        from nucleusiq.agents.agent_result import AbstentionSignal
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        async def mock_call(**kwargs):
            return FakeLLMResponse(content="some result")

        agent = self._make_auto_agent()
        agent.llm.call = AsyncMock(side_effect=mock_call)
        agent.config.max_retries = 2

        mode = AutonomousMode()
        task = _make_task("Hard task")

        with patch.object(
            AutonomousMode,
            "_run_critic",
            new=AsyncMock(
                return_value=CritiqueResult(
                    verdict=Verdict.FAIL,
                    score=0.1,
                    feedback="nope",
                )
            ),
        ):
            with pytest.raises(AbstentionSignal) as excinfo:
                await mode.run(agent, task)

        assert excinfo.value.best_candidate is not None
        assert excinfo.value.critique.verdict == Verdict.FAIL
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_no_llm_delegates_to_standard(self):
        """No LLM → delegates to StandardMode directly."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
        from nucleusiq.agents.modes.standard_mode import StandardMode

        agent = _make_agent(has_llm=False)

        async def fake_std_run(a, t):
            return "Echo: task"

        mode = AutonomousMode()
        task = _make_task("task")

        with patch.object(StandardMode, "run", side_effect=fake_std_run):
            result = await mode.run(agent, task)

        assert result == "Echo: task"

    @pytest.mark.asyncio
    async def test_error_result_returns_immediately(self):
        """If tool loop returns error, AutonomousMode returns it."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        async def mock_call(**kwargs):
            return FakeLLMResponse(content=None)

        agent = self._make_auto_agent()
        agent.llm.call = AsyncMock(side_effect=mock_call)

        mode = AutonomousMode()
        task = _make_task("task")
        result = await mode.run(agent, task)

        assert "Error" in str(result)

    def test_is_error_helper(self):
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        assert mode._is_error("Error: Something went wrong") is True
        assert mode._is_error("   Error: bad") is True
        assert mode._is_error("Success") is False
        assert mode._is_error(42) is False

    @pytest.mark.asyncio
    async def test_progress_tracking(self):
        """Autonomous mode tracks execution progress."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        async def mock_call(**kwargs):
            return FakeLLMResponse(content="result")

        agent = self._make_auto_agent()
        agent.llm.call = AsyncMock(side_effect=mock_call)

        mode = AutonomousMode()
        task = _make_task("Compute something")

        with patch.object(
            AutonomousMode,
            "_run_critic",
            new=AsyncMock(return_value=CritiqueResult(verdict=Verdict.PASS, score=1.0)),
        ):
            await mode.run(agent, task)

        progress = agent._execution_progress
        assert progress is not None
        assert len(progress.steps) >= 1


class TestAutonomousModeConfig:
    def test_config_has_critique_rounds(self):
        config = AgentConfig()
        assert config.critique_rounds == 3

    def test_config_custom_critique_rounds(self):
        config = AgentConfig(critique_rounds=5)
        assert config.critique_rounds == 5


# ================================================================== #
# Utility Tests                                                        #
# ================================================================== #


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self):
        text = "a" * 200
        result = _truncate(text, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_exact_length_unchanged(self):
        text = "a" * 50
        assert _truncate(text, 50) == text
