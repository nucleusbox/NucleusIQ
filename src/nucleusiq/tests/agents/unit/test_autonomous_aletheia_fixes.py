"""
Tests for the Autonomous Mode verification pipeline.

Validates the three core concerns:

1. **Critic context limits** — ``CriticLimits`` controls how much of the
   Generator's output the Verifier sees.  STANDARD and REASONING presets
   are provided; the orchestrator selects the right one automatically.

2. **Error retry instead of bailout** — ``_is_error`` no longer kills the
   retry loop; errors are treated as failed attempts that get retried.

3. **Re-synthesis over re-exploration** — The Refiner message discourages
   re-calling tools and encourages re-synthesizing from existing data.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import (
    REASONING_LIMITS,
    STANDARD_LIMITS,
    Critic,
    CriticLimits,
    CritiqueResult,
    Verdict,
    _truncate,
)
from nucleusiq.agents.components.refiner import Refiner
from nucleusiq.agents.config.agent_config import (
    AgentConfig,
    AgentState,
    ExecutionMode,
)
from nucleusiq.agents.task import Task

# ================================================================== #
# Helpers                                                              #
# ================================================================== #


def _make_mock_agent(
    model_name: str = "gpt-5.1",
    config: AgentConfig | None = None,
    llm_response_content: str | None = "answer 42",
):
    """Build a mock Agent with controllable LLM responses."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.role = "Analyst"
    agent.tools = []
    agent._logger = MagicMock()
    agent._last_messages = []
    agent._execution_progress = None
    agent._plugin_manager = None
    agent.state = AgentState.EXECUTING
    agent.memory = None

    if config is None:
        config = AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS)
    agent.config = config

    llm = MagicMock()
    llm.model_name = model_name
    llm.is_reasoning_model = False
    llm.call = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=llm_response_content))],
        )
    )
    llm.convert_tool_specs = MagicMock(return_value=[])
    agent.llm = llm

    agent._executor = MagicMock()
    agent._executor.execute = AsyncMock(return_value="tool result")
    agent._usage_tracker = None
    agent._tracer = None
    return agent


# ================================================================== #
# 1. Critic Truncation Limits (Aletheia "full context" principle)      #
# ================================================================== #


class TestCriticTruncationLimits:
    """Verify the Critic gives the Verifier generous context."""

    def test_claimed_answer_uses_standard_limit(self):
        """Answer section respects STANDARD_LIMITS.claimed_answer."""
        critic = Critic()
        long_answer = "A" * 25_000

        prompt = critic.build_verification_prompt(
            task_objective="Write a report",
            final_result=long_answer,
            generator_messages=None,
            allow_tool_instructions=False,
        )

        assert "ANSWER TO VERIFY" in prompt
        answer_section = prompt.split("ANSWER TO VERIFY\n")[1].split("\n\n##")[0]
        assert len(answer_section) >= STANDARD_LIMITS.claimed_answer - 3

    def test_claimed_answer_tool_path_uses_standard_limit(self):
        """Tool verification path also respects STANDARD_LIMITS.claimed_answer."""
        critic = Critic()
        long_answer = "B" * 25_000

        gen_msgs = [
            MagicMock(
                role="assistant",
                content=None,
                tool_calls=[
                    MagicMock(name="read_file", arguments='{"path": "x"}'),
                ],
            ),
            MagicMock(role="tool", content="result data", tool_calls=None),
        ]

        prompt = critic.build_verification_prompt(
            task_objective="Analyze",
            final_result=long_answer,
            generator_messages=gen_msgs,
            allow_tool_instructions=True,
        )

        answer_section = prompt.split("CLAIMED ANSWER\n")[1].split("\n\n##")[0]
        assert len(answer_section) >= STANDARD_LIMITS.claimed_answer - 3

    def test_tool_result_limit(self):
        """Tool results in trace respect STANDARD_LIMITS.tool_result."""
        critic = Critic()
        long_tool_result = "D" * (STANDARD_LIMITS.tool_result + 500)

        gen_msgs = [
            {"role": "tool", "content": long_tool_result},
        ]

        trace = critic._extract_reasoning_trace(gen_msgs)

        result_line = [line for line in trace.split("\n") if "[Tool Result]" in line][0]
        content_part = result_line.replace("[Tool Result] ", "")
        assert len(content_part) >= STANDARD_LIMITS.tool_result - 3

    def test_assistant_content_limit(self):
        """Assistant content in trace respects STANDARD_LIMITS.assistant_content."""
        critic = Critic()
        long_content = "E" * (STANDARD_LIMITS.assistant_content + 500)

        gen_msgs = [
            {"role": "assistant", "content": long_content, "tool_calls": None},
        ]

        trace = critic._extract_reasoning_trace(gen_msgs)

        assistant_line = [line for line in trace.split("\n") if "[Assistant]" in line][
            0
        ]
        content_part = assistant_line.replace("[Assistant] ", "")
        assert len(content_part) >= STANDARD_LIMITS.assistant_content - 3

    def test_tool_call_args_limit(self):
        """Tool call arguments in trace respect STANDARD_LIMITS.tool_args."""
        critic = Critic()
        long_args = "F" * (STANDARD_LIMITS.tool_args + 200)

        gen_msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "my_tool", "arguments": long_args}},
                ],
            },
        ]

        trace = critic._extract_reasoning_trace(gen_msgs)
        tool_line = [line for line in trace.split("\n") if "[Tool Call]" in line][0]
        assert len(tool_line) > STANDARD_LIMITS.tool_args

    def test_trace_line_cap(self):
        """Trace respects STANDARD_LIMITS.trace_lines."""
        critic = Critic()
        n_msgs = STANDARD_LIMITS.trace_lines + 50
        gen_msgs = []
        for i in range(n_msgs):
            gen_msgs.append(
                {"role": "assistant", "content": f"Step {i}", "tool_calls": None}
            )

        trace = critic._extract_reasoning_trace(gen_msgs, STANDARD_LIMITS)
        lines = trace.split("\n")

        assert "middle steps omitted" in trace
        assert len(lines) == STANDARD_LIMITS.trace_lines

    def test_trace_under_cap_not_truncated(self):
        """Traces within the limit should NOT be truncated."""
        critic = Critic()
        n_msgs = STANDARD_LIMITS.trace_lines - 20
        gen_msgs = []
        for i in range(n_msgs):
            gen_msgs.append(
                {"role": "assistant", "content": f"Step {i}", "tool_calls": None}
            )

        trace = critic._extract_reasoning_trace(gen_msgs)
        assert "middle steps omitted" not in trace

    def test_legacy_conversation_review_uses_instance_limits(self):
        """_build_conversation_review_prompt uses limits from the instance."""
        critic = Critic()
        gen_msgs = [
            MagicMock(
                role="assistant",
                content="X" * (STANDARD_LIMITS.assistant_content + 500),
                tool_calls=None,
            ),
            MagicMock(
                role="tool",
                content="Y" * (STANDARD_LIMITS.tool_result + 500),
                tool_calls=None,
            ),
        ]
        prompt = critic._build_conversation_review_prompt(
            "task",
            "Z" * (STANDARD_LIMITS.claimed_answer + 5000),
            gen_msgs,
        )

        assert "Agent's Final Answer" in prompt
        answer_section = prompt.split("Agent's Final Answer\n")[1].split("\n\n##")[0]
        assert len(answer_section) >= STANDARD_LIMITS.claimed_answer - 3


# ================================================================== #
# 2. Error Retry (no more _is_error bailout)                          #
# ================================================================== #


class TestErrorRetryNotBailout:
    """Verify that _is_error no longer causes immediate return."""

    @pytest.mark.asyncio
    async def test_run_simple_retries_on_error(self):
        """_run_simple should retry when Generator returns an error string."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = _make_mock_agent(
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=3,
            ),
        )

        call_count = [0]

        async def mock_tool_call_loop(ag, task, msgs, specs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return "Error: LLM did not respond."
            return "Final valid answer"

        from nucleusiq.agents.modes.standard_mode import StandardMode

        std_mode = StandardMode()
        std_mode._ensure_executor = MagicMock()
        std_mode._get_tool_specs = MagicMock(return_value=[])
        std_mode.build_messages = MagicMock(
            return_value=[ChatMessage(role="user", content="Test")]
        )
        std_mode._tool_call_loop = mock_tool_call_loop

        import nucleusiq.agents.modes.autonomous_mode as auto_mod

        original_std = auto_mod.StandardMode

        class PatchedStd:
            def __init__(self):
                self._ensure_executor = std_mode._ensure_executor
                self._get_tool_specs = std_mode._get_tool_specs
                self.build_messages = std_mode.build_messages
                self._tool_call_loop = std_mode._tool_call_loop

        auto_mod.StandardMode = PatchedStd

        async def mock_validate(self_vp, ag, result, msgs):
            from nucleusiq.agents.components.validation import ValidationResult

            return ValidationResult(valid=True, layer="none", reason="ok")

        from nucleusiq.agents.components.validation import ValidationPipeline

        original_validate = ValidationPipeline.validate
        ValidationPipeline.validate = mock_validate

        critique_pass = json.dumps(
            {"verdict": "pass", "score": 0.95, "feedback": "Good"}
        )

        async def mock_call_llm(ag, kwargs, **kw):
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content=critique_pass))]
            )

        mode.call_llm = mock_call_llm

        try:
            result = await mode._run_simple(agent, Task(id="t1", objective="Test"))

            assert call_count[0] == 3
            assert result == "Final valid answer"
            assert "Error" not in str(result)
        finally:
            auto_mod.StandardMode = original_std
            ValidationPipeline.validate = original_validate

    @pytest.mark.asyncio
    async def test_run_simple_returns_error_after_exhausting_retries(self):
        """If all retries produce errors, the last error is returned."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = _make_mock_agent(
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=2,
            ),
        )

        async def always_error(ag, task, msgs, specs):
            return "Error: LLM did not respond."

        import nucleusiq.agents.modes.autonomous_mode as auto_mod

        original_std = auto_mod.StandardMode

        class PatchedStd:
            def __init__(self):
                self._ensure_executor = MagicMock()
                self._get_tool_specs = MagicMock(return_value=[])
                self.build_messages = MagicMock(
                    return_value=[ChatMessage(role="user", content="Test")]
                )
                self._tool_call_loop = always_error

        auto_mod.StandardMode = PatchedStd

        try:
            result = await mode._run_simple(agent, Task(id="t1", objective="Test"))
            assert result == "Error: LLM did not respond."
        finally:
            auto_mod.StandardMode = original_std

    def test_is_error_still_detects_errors(self):
        """_is_error helper still works for detection (used by retry logic)."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        assert AutonomousMode._is_error("Error: LLM did not respond.") is True
        assert AutonomousMode._is_error("Error: Connection timeout") is True
        assert AutonomousMode._is_error("Valid result text") is False
        assert AutonomousMode._is_error(None) is False
        assert AutonomousMode._is_error("") is False


# ================================================================== #
# 3. Refiner — Re-synthesize, not re-explore                          #
# ================================================================== #


class TestRefinerReSynthesis:
    """Reviser prompt discourages blind re-tooling and re-exploration.

    These tests target the F1 ``Refiner._build_revision_prompt`` static
    method, which is the sole source of truth for how the Reviser role
    is primed.  (The legacy ``build_revision_message`` helper was deleted
    in F1 as part of the ISP cleanup.)
    """

    @staticmethod
    def _build(critique: CritiqueResult, *, candidate: str = "old candidate") -> str:
        return Refiner._build_revision_prompt(
            task_objective="Test objective",
            candidate=candidate,
            critique=critique,
            tool_result_summary=None,
        )

    def test_revision_prompt_discourages_retooling(self):
        critique = CritiqueResult(
            verdict=Verdict.FAIL,
            score=0.2,
            feedback="Missing revenue data",
            issues=["Revenue figure is wrong"],
            suggestions=["Use the tool result from step 3"],
        )

        msg = self._build(critique)

        assert "Do NOT call tools" in msg
        assert "MISSING data" in msg

    def test_revision_prompt_allows_tools_for_missing_data(self):
        """Should mention that tools are OK when data is MISSING."""
        critique = CritiqueResult(
            verdict=Verdict.FAIL,
            score=0.15,
            feedback="Incomplete",
            issues=["Missing PE ratio analysis"],
        )

        msg = self._build(critique)

        assert "MISSING data" in msg

    def test_revision_prompt_does_not_encourage_reexploration(self):
        """Old message had 'If you need to re-call a tool, do so' — removed."""
        critique = CritiqueResult(
            verdict=Verdict.FAIL,
            score=0.3,
            feedback="Wrong calculation",
            issues=["Sum is incorrect"],
        )

        msg = self._build(critique)

        assert (
            "If you need to re-call a tool with different arguments, do so" not in msg
        )

    def test_revision_prompt_includes_issues_and_suggestions(self):
        critique = CritiqueResult(
            verdict=Verdict.FAIL,
            score=0.2,
            feedback="Errors found",
            issues=["Issue A", "Issue B"],
            suggestions=["Fix A", "Fix B"],
        )

        msg = self._build(critique)

        assert "Issue A" in msg
        assert "Issue B" in msg
        assert "Fix A" in msg
        assert "Fix B" in msg

    def test_revision_prompt_asks_for_complete_answer(self):
        critique = CritiqueResult(
            verdict=Verdict.FAIL,
            score=0.25,
            feedback="Incomplete",
            issues=["Missing section"],
        )

        msg = self._build(critique)

        assert "complete, self-contained final answer" in msg


# ================================================================== #
# 4. CriticLimits configuration and auto-selection                     #
# ================================================================== #


class TestCriticLimitsConfig:
    """Verify CriticLimits presets and auto-selection logic."""

    def test_standard_limits_values(self):
        assert STANDARD_LIMITS.claimed_answer == 20_000
        assert STANDARD_LIMITS.tool_result == 3_000
        assert STANDARD_LIMITS.assistant_content == 2_000
        assert STANDARD_LIMITS.tool_args == 600
        assert STANDARD_LIMITS.trace_lines == 120
        assert STANDARD_LIMITS.evidence_total == 20_000
        assert STANDARD_LIMITS.reasoning_total == 8_000

    def test_reasoning_limits_larger_than_standard(self):
        assert REASONING_LIMITS.claimed_answer > STANDARD_LIMITS.claimed_answer
        assert REASONING_LIMITS.tool_result > STANDARD_LIMITS.tool_result
        assert REASONING_LIMITS.assistant_content > STANDARD_LIMITS.assistant_content
        assert REASONING_LIMITS.trace_lines > STANDARD_LIMITS.trace_lines
        assert REASONING_LIMITS.evidence_total > STANDARD_LIMITS.evidence_total
        assert REASONING_LIMITS.reasoning_total > STANDARD_LIMITS.reasoning_total

    def test_reasoning_limits_values(self):
        assert REASONING_LIMITS.claimed_answer == 50_000
        assert REASONING_LIMITS.tool_result == 5_000
        assert REASONING_LIMITS.assistant_content == 4_000
        assert REASONING_LIMITS.tool_args == 1_000
        assert REASONING_LIMITS.trace_lines == 200
        assert REASONING_LIMITS.evidence_total == 40_000
        assert REASONING_LIMITS.reasoning_total == 16_000

    def test_critic_defaults_to_standard(self):
        critic = Critic()
        assert critic._limits is STANDARD_LIMITS

    def test_critic_accepts_custom_limits(self):
        custom = CriticLimits(
            claimed_answer=5_000,
            tool_result=500,
            assistant_content=600,
            tool_args=200,
            trace_lines=40,
            evidence_total=3_000,
            reasoning_total=1_500,
        )
        critic = Critic(limits=custom)
        assert critic._limits is custom
        assert critic._limits.claimed_answer == 5_000

    def test_reasoning_model_gets_reasoning_limits(self):
        """AutonomousMode._select_critic_limits picks REASONING for reasoning LLMs."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        agent = _make_mock_agent()
        agent.llm.is_reasoning_model = True
        assert AutonomousMode._select_critic_limits(agent) is REASONING_LIMITS

    def test_standard_model_gets_standard_limits(self):
        """AutonomousMode._select_critic_limits picks STANDARD for regular LLMs."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        agent = _make_mock_agent()
        agent.llm.is_reasoning_model = False
        assert AutonomousMode._select_critic_limits(agent) is STANDARD_LIMITS

    def test_reasoning_limits_expand_claimed_answer(self):
        """With REASONING limits the Verifier sees 50k chars of the answer."""
        critic = Critic(limits=REASONING_LIMITS)
        long_answer = "R" * 55_000

        prompt = critic.build_verification_prompt(
            task_objective="Deep analysis",
            final_result=long_answer,
            generator_messages=None,
            allow_tool_instructions=False,
        )

        answer_section = prompt.split("ANSWER TO VERIFY\n")[1].split("\n\n##")[0]
        assert len(answer_section) >= REASONING_LIMITS.claimed_answer - 3

    def test_reasoning_limits_expand_trace_lines(self):
        """With REASONING limits, trace cap is 200 lines."""
        critic = Critic(limits=REASONING_LIMITS)
        gen_msgs = []
        for i in range(250):
            gen_msgs.append(
                {"role": "assistant", "content": f"Step {i}", "tool_calls": None}
            )

        trace = critic._extract_reasoning_trace(gen_msgs, REASONING_LIMITS)
        lines = trace.split("\n")

        assert "middle steps omitted" in trace
        assert len(lines) == REASONING_LIMITS.trace_lines


# ================================================================== #
# 5. Integration: Error → Retry → Critic Pass                         #
# ================================================================== #


class TestErrorRetryCriticIntegration:
    """End-to-end: error on attempt 1 → retry → success → Critic pass."""

    @pytest.mark.asyncio
    async def test_error_then_success_then_critic_pass(self):
        """Simulates: attempt 1 = error, attempt 2 = valid, Critic = PASS."""
        import nucleusiq.agents.modes.autonomous_mode as auto_mod
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = _make_mock_agent(
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=3,
            ),
        )

        call_count = [0]

        async def mock_loop(ag, task, msgs, specs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Error: LLM did not respond."
            return "Complete TCS equity analysis report..."

        original_std = auto_mod.StandardMode

        class PatchedStd:
            def __init__(self):
                self._ensure_executor = MagicMock()
                self._get_tool_specs = MagicMock(return_value=[])
                self.build_messages = MagicMock(
                    return_value=[ChatMessage(role="user", content="Analyze TCS")]
                )
                self._tool_call_loop = mock_loop

        auto_mod.StandardMode = PatchedStd

        async def mock_validate(self_vp, ag, result, msgs):
            from nucleusiq.agents.components.validation import ValidationResult

            return ValidationResult(valid=True, layer="none", reason="ok")

        from nucleusiq.agents.components.validation import ValidationPipeline

        original_validate = ValidationPipeline.validate
        ValidationPipeline.validate = mock_validate

        async def mock_call_llm(ag, kwargs, **kw):
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps(
                                {
                                    "verdict": "pass",
                                    "score": 0.92,
                                    "feedback": "Thorough analysis",
                                }
                            )
                        )
                    )
                ]
            )

        mode.call_llm = mock_call_llm

        try:
            result = await mode._run_simple(
                agent, Task(id="t1", objective="Analyze TCS")
            )
            assert call_count[0] == 2
            assert "TCS equity analysis" in result
        finally:
            auto_mod.StandardMode = original_std
            ValidationPipeline.validate = original_validate


# ================================================================== #
# 6. _truncate helper                                                  #
# ================================================================== #


class TestTruncateHelper:
    """Verify the _truncate utility works correctly."""

    def test_short_text_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_exact_length_unchanged(self):
        assert _truncate("abc", 3) == "abc"

    def test_long_text_truncated_with_ellipsis(self):
        result = _truncate("A" * 100, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_empty_text(self):
        assert _truncate("", 100) == ""
