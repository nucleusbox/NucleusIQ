"""
Tests for the adaptive verification system.

Covers:
- Critic.build_verification_prompt() strategy dispatch
- Tool verification prompt generation
- Reasoning verification prompt generation
- _extract_reasoning_trace() with and without tools
"""

import json

from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.components.critic import (
    Critic,
    CritiqueResult,
    Verdict,
)

# ================================================================== #
# Fixtures / helpers                                                   #
# ================================================================== #


def _tool_messages() -> list:
    """Simulated Generator conversation that used tools."""
    return [
        ChatMessage(role="system", content="You are an agent."),
        ChatMessage(role="user", content="Calculate 2+3"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCallRequest(name="add", arguments='{"a": 2, "b": 3}'),
            ],
        ),
        ChatMessage(role="tool", content="5"),
        ChatMessage(role="assistant", content="The answer is 5."),
    ]


def _reasoning_messages() -> list:
    """Simulated Generator conversation with no tools."""
    return [
        ChatMessage(role="system", content="You are an agent."),
        ChatMessage(role="user", content="Explain quantum entanglement"),
        ChatMessage(
            role="assistant",
            content=(
                "Quantum entanglement is a phenomenon where particles "
                "become correlated so that the state of one instantly "
                "influences the state of the other, regardless of distance."
            ),
        ),
    ]


# ================================================================== #
# Strategy Dispatch                                                    #
# ================================================================== #


class TestBuildVerificationPromptDispatch:
    def test_dispatches_to_tool_strategy_when_tools_used(self):
        critic = Critic()
        prompt = critic.build_verification_prompt(
            task_objective="Calculate 2+3",
            final_result="5",
            generator_messages=_tool_messages(),
        )
        assert "RE-DERIVE THE FINAL ANSWER" in prompt
        assert "EXECUTION TRACE" in prompt

    def test_dispatches_to_reasoning_strategy_when_no_tools(self):
        critic = Critic()
        prompt = critic.build_verification_prompt(
            task_objective="Explain X",
            final_result="Some explanation",
            generator_messages=_reasoning_messages(),
        )
        assert "Logical Consistency" in prompt
        assert "Completeness" in prompt

    def test_dispatches_to_reasoning_when_no_messages(self):
        critic = Critic()
        prompt = critic.build_verification_prompt(
            task_objective="Simple task",
            final_result="42",
            generator_messages=None,
        )
        assert "Logical Consistency" in prompt

    def test_dispatches_to_reasoning_when_empty_messages(self):
        critic = Critic()
        prompt = critic.build_verification_prompt(
            task_objective="Simple task",
            final_result="42",
            generator_messages=[],
        )
        assert "Logical Consistency" in prompt


# ================================================================== #
# Tool Verification Strategy                                           #
# ================================================================== #


class TestToolVerification:
    def test_includes_execution_trace(self):
        prompt = Critic._build_tool_verification(
            "Calculate WACC",
            "0.12",
            "[Tool Call] calculate_wacc({...})\n[Tool Result] 0.12",
        )
        assert "EXECUTION TRACE" in prompt
        assert "calculate_wacc" in prompt

    def test_instructs_final_answer_re_derivation(self):
        prompt = Critic._build_tool_verification(
            "Task",
            "Result",
            "[Tool Call] foo()\n[Tool Result] bar",
        )
        assert "MUST call tools" in prompt
        assert "RE-DERIVE THE FINAL ANSWER" in prompt

    def test_includes_verdict_format(self):
        prompt = Critic._build_tool_verification(
            "Task",
            "Result",
            "[Tool Call] x()\n[Tool Result] y",
        )
        assert '"verdict"' in prompt
        assert '"score"' in prompt

    def test_original_task_included(self):
        prompt = Critic._build_tool_verification(
            "Calculate ROI for Project Alpha",
            "15%",
            "trace",
        )
        assert "Calculate ROI for Project Alpha" in prompt

    def test_claimed_answer_included(self):
        prompt = Critic._build_tool_verification(
            "Task",
            "The answer is 42",
            "trace",
        )
        assert "The answer is 42" in prompt


# ================================================================== #
# Reasoning Verification Strategy                                      #
# ================================================================== #


class TestReasoningVerification:
    def test_no_tool_requirement(self):
        prompt = Critic._build_reasoning_verification(
            "Explain gravity",
            "Gravity is a force...",
            "Some reasoning",
        )
        assert "Call 1-3 tools" not in prompt

    def test_checks_four_dimensions(self):
        prompt = Critic._build_reasoning_verification(
            "Task",
            "Answer",
            "Trace",
        )
        assert "Logical Consistency" in prompt
        assert "Completeness" in prompt
        assert "Factual Plausibility" in prompt
        assert "Task Alignment" in prompt

    def test_includes_trace_when_present(self):
        prompt = Critic._build_reasoning_verification(
            "Task",
            "Answer",
            "The agent reasoned step by step...",
        )
        assert "GENERATOR'S RESPONSE" in prompt
        assert "step by step" in prompt

    def test_skips_trace_when_empty(self):
        prompt = Critic._build_reasoning_verification(
            "Task",
            "Answer",
            "",
        )
        assert "GENERATOR'S RESPONSE" not in prompt

    def test_includes_verdict_format(self):
        prompt = Critic._build_reasoning_verification(
            "Task",
            "Result",
            "",
        )
        assert '"verdict"' in prompt
        assert '"score"' in prompt


# ================================================================== #
# _extract_reasoning_trace                                             #
# ================================================================== #


class TestExtractReasoningTrace:
    def test_extracts_tool_calls(self):
        msgs = _tool_messages()
        trace = Critic._extract_reasoning_trace(msgs)
        assert "[Tool Call] add" in trace
        assert "[Tool Result] 5" in trace

    def test_extracts_assistant_text(self):
        msgs = _tool_messages()
        trace = Critic._extract_reasoning_trace(msgs)
        assert "[Assistant] The answer is 5." in trace

    def test_skips_system_and_user(self):
        msgs = _tool_messages()
        trace = Critic._extract_reasoning_trace(msgs)
        assert "You are an agent" not in trace
        assert "Calculate 2+3" not in trace

    def test_returns_empty_for_none(self):
        assert Critic._extract_reasoning_trace(None) == ""

    def test_returns_empty_for_empty_list(self):
        assert Critic._extract_reasoning_trace([]) == ""

    def test_reasoning_only_messages(self):
        msgs = _reasoning_messages()
        trace = Critic._extract_reasoning_trace(msgs)
        assert "[Assistant]" in trace
        assert "entanglement" in trace
        assert "[Tool Call]" not in trace

    def test_truncates_long_traces(self):
        msgs = []
        for i in range(60):
            msgs.append(ChatMessage(role="assistant", content=f"Step {i}"))
        trace = Critic._extract_reasoning_trace(msgs)
        assert "middle steps omitted" in trace

    def test_handles_dict_format_messages(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "answer", "tool_calls": None},
        ]
        trace = Critic._extract_reasoning_trace(msgs)
        assert "[Assistant] answer" in trace

    def test_handles_dict_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "calc", "arguments": '{"x":1}'}},
                ],
            },
            {"role": "tool", "content": "result-42"},
        ]
        trace = Critic._extract_reasoning_trace(msgs)
        assert "[Tool Call] calc" in trace
        assert "[Tool Result] result-42" in trace


# ================================================================== #
# verifier_answer field                                                #
# ================================================================== #


class TestVerifierAnswerField:
    def test_critique_result_default_none(self):
        cr = CritiqueResult(verdict=Verdict.PASS)
        assert cr.verifier_answer is None

    def test_critique_result_with_verifier_answer(self):
        cr = CritiqueResult(
            verdict=Verdict.PASS,
            verifier_answer="10.95",
        )
        assert cr.verifier_answer == "10.95"

    def test_parse_extracts_verifier_answer(self):
        critic = Critic()
        text = json.dumps(
            {
                "verdict": "pass",
                "score": 0.95,
                "verifier_answer": "10.95",
                "feedback": "Confirmed",
                "issues": [],
                "suggestions": [],
            }
        )
        result = critic.parse_result_text(text)
        assert result.verifier_answer == "10.95"
        assert result.verdict == Verdict.PASS

    def test_parse_handles_missing_verifier_answer(self):
        critic = Critic()
        text = json.dumps(
            {
                "verdict": "pass",
                "score": 0.9,
                "feedback": "Looks good",
            }
        )
        result = critic.parse_result_text(text)
        assert result.verifier_answer is None
        assert result.verdict == Verdict.PASS

    def test_parse_handles_numeric_verifier_answer(self):
        critic = Critic()
        text = json.dumps(
            {
                "verdict": "fail",
                "score": 0.2,
                "verifier_answer": 42.5,
                "feedback": "Wrong answer",
            }
        )
        result = critic.parse_result_text(text)
        assert result.verifier_answer == "42.5"

    def test_tool_verification_prompt_includes_verifier_answer(self):
        prompt = Critic._build_tool_verification(
            "Calculate X",
            "42",
            "[Tool Call] calc()\n[Tool Result] 42",
        )
        assert "verifier_answer" in prompt
        assert "RE-DERIVE THE FINAL ANSWER" in prompt

    def test_reasoning_verification_prompt_includes_verifier_answer(self):
        prompt = Critic._build_reasoning_verification(
            "Explain X",
            "answer",
            "",
        )
        assert "verifier_answer" in prompt
