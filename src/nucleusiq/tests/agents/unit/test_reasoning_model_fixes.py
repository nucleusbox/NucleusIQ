"""
Tests for reasoning-model compatibility fixes.

Validates all findings from the autonomous-mode audit:

1. AgentConfig: single llm_max_output_tokens propagates everywhere
   (no separate critic/decomposer/multiplier configs exposed to user)
2. BaseLLM: is_reasoning_model property (provider contract)
3. Critic: uses agent.config.llm_max_output_tokens (was hardcoded 1024)
4. Critic: uses reasoning-only prompt when tools unavailable
5. Decomposer: uses agent.config.llm_max_output_tokens (was hardcoded 512)
6. Synthesis pass: falls back to pre-synthesis content when synthesis is empty
7. Empty-response retries: hardcoded to 2 (was 1)
8. Responses API streaming: extracts reasoning_tokens from output_details
9. Task C: has explicit llm_max_output_tokens override
10. Provider-level reasoning detection (not hardcoded in core)

Design principles validated:
- ONE user-facing knob: llm_max_output_tokens
- Reasoning model detection lives in provider (BaseLLM.is_reasoning_model)
- Internal constants are not exposed as config
"""

from __future__ import annotations

import importlib
import inspect
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import (
    Critic,
    CritiqueResult,
    Verdict,
)
from nucleusiq.agents.components.decomposer import Decomposer
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
    is_reasoning: bool = False,
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

    if config is None:
        config = AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS)
    agent.config = config

    llm = MagicMock()
    llm.model_name = model_name
    llm.is_reasoning_model = is_reasoning
    llm.call = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(message=MagicMock(content=llm_response_content))
            ],
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
# 1. AgentConfig — Clean Single Knob                                  #
# ================================================================== #


class TestAgentConfigCleanDesign:
    """Validate that AgentConfig exposes only llm_max_output_tokens,
    and does NOT expose internal fields."""

    def test_llm_max_output_tokens_exists(self):
        cfg = AgentConfig()
        assert cfg.llm_max_output_tokens == 2048

    def test_no_critic_config_exposed(self):
        assert not hasattr(AgentConfig(), "critic_max_output_tokens")

    def test_no_decomposer_config_exposed(self):
        assert not hasattr(AgentConfig(), "decomposer_max_output_tokens")

    def test_no_reasoning_multiplier_exposed(self):
        assert not hasattr(AgentConfig(), "reasoning_token_multiplier")

    def test_no_empty_retries_config_exposed(self):
        assert not hasattr(AgentConfig(), "empty_response_retries")

    def test_no_reasoning_prefixes_in_core(self):
        assert not hasattr(AgentConfig(), "_REASONING_PREFIXES")

    def test_no_is_reasoning_model_on_config(self):
        """Reasoning model detection is provider's job, not config's."""
        assert not hasattr(AgentConfig, "is_reasoning_model")

    def test_custom_token_budget(self):
        cfg = AgentConfig(llm_max_output_tokens=8192)
        assert cfg.llm_max_output_tokens == 8192


# ================================================================== #
# 2. BaseLLM — Provider Contract                                      #
# ================================================================== #


class TestBaseLLMReasoningContract:
    """Validate that reasoning model detection is a provider concern."""

    def test_base_llm_has_is_reasoning_model(self):
        from nucleusiq.llms.base_llm import BaseLLM

        assert hasattr(BaseLLM, "is_reasoning_model")

    def test_base_llm_defaults_false(self):
        from nucleusiq.llms.base_llm import BaseLLM

        class DummyLLM(BaseLLM):
            async def call(self, **kwargs):
                pass

        llm = DummyLLM()
        assert llm.is_reasoning_model is False

    def test_openai_gpt5_is_reasoning(self):
        """OpenAI provider correctly identifies gpt-5.x as reasoning."""
        from nucleusiq_openai._shared.model_config import (
            uses_max_completion_tokens,
        )

        assert uses_max_completion_tokens("gpt-5") is True
        assert uses_max_completion_tokens("gpt-5.1") is True
        assert uses_max_completion_tokens("gpt-5.4-mini") is True

    def test_openai_o_series_is_reasoning(self):
        from nucleusiq_openai._shared.model_config import (
            uses_max_completion_tokens,
        )

        assert uses_max_completion_tokens("o1") is True
        assert uses_max_completion_tokens("o3-mini") is True
        assert uses_max_completion_tokens("o4-mini") is True

    def test_openai_gpt4_not_reasoning(self):
        from nucleusiq_openai._shared.model_config import (
            uses_max_completion_tokens,
        )

        assert uses_max_completion_tokens("gpt-4.1") is False
        assert uses_max_completion_tokens("gpt-4.1-mini") is False
        assert uses_max_completion_tokens("gpt-4o") is False


# ================================================================== #
# 3. Critic — Uses Uniform Token Budget                               #
# ================================================================== #


class TestCriticUsesUniformBudget:
    """Verify _run_critic reads llm_max_output_tokens from config."""

    @pytest.mark.asyncio
    async def test_critic_uses_config_budget(self):
        """Critic should use agent.config.llm_max_output_tokens, not hardcoded 1024."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = _make_mock_agent(
            model_name="gpt-5.1",
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                llm_max_output_tokens=4096,
            ),
        )

        critique_json = json.dumps({
            "verdict": "pass", "score": 0.9, "feedback": "Looks good",
        })
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=critique_json))]
            )
        )

        captured_kwargs: dict[str, Any] = {}

        async def spy_call_llm(ag, kwargs, **kw):
            captured_kwargs.update(kwargs)
            return await ag.llm.call(**kwargs)

        mode.call_llm = spy_call_llm

        result = await mode._run_critic(
            agent, Critic(), "Test task", "Result text",
            [ChatMessage(role="assistant", content="Result text")],
        )

        assert captured_kwargs["max_output_tokens"] == 4096
        assert result.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_critic_uses_default_2048(self):
        """With default config, Critic gets 2048 (not old hardcoded 1024)."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = _make_mock_agent(model_name="gpt-4.1")

        critique_json = json.dumps({
            "verdict": "pass", "score": 0.85, "feedback": "OK",
        })
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=critique_json))]
            )
        )

        captured_kwargs: dict[str, Any] = {}

        async def spy_call_llm(ag, kwargs, **kw):
            captured_kwargs.update(kwargs)
            return await ag.llm.call(**kwargs)

        mode.call_llm = spy_call_llm

        await mode._run_critic(agent, Critic(), "Test", "Result", [])

        assert captured_kwargs["max_output_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_critic_empty_response_returns_uncertain(self):
        """When LLM returns empty (budget exhaustion), Critic returns UNCERTAIN."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = _make_mock_agent(model_name="gpt-5.1")
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=None))]
            )
        )

        async def passthrough(ag, kwargs, **kw):
            return await ag.llm.call(**kwargs)

        mode.call_llm = passthrough

        result = await mode._run_critic(agent, Critic(), "Task", "Result", [])

        assert result.verdict == Verdict.UNCERTAIN
        assert result.score == 0.5


# ================================================================== #
# 4. Critic Prompt — No Tool Instructions When Tools Unavailable      #
# ================================================================== #


class TestCriticPromptSelection:
    """Verify Critic never instructs 'call tools' when tools aren't provided."""

    def test_allow_tool_instructions_false_forces_reasoning_prompt(self):
        critic = Critic()
        generator_msgs = [
            MagicMock(role="assistant", content=None, tool_calls=[
                MagicMock(
                    function=MagicMock(name="read_file", arguments='{"path": "x"}'),
                    id="tc1",
                )
            ]),
            MagicMock(role="tool", content="file content", tool_call_id="tc1"),
        ]

        prompt = critic.build_verification_prompt(
            task_objective="Analyze code",
            final_result="Code is clean",
            generator_messages=generator_msgs,
            allow_tool_instructions=False,
        )

        assert "MUST call tools" not in prompt
        assert "MUST call" not in prompt

    def test_default_preserves_tool_instructions(self):
        critic = Critic()
        generator_msgs = [
            MagicMock(role="assistant", content=None, tool_calls=[
                MagicMock(
                    function=MagicMock(name="read_file", arguments='{"path": "x"}'),
                    id="tc1",
                )
            ]),
            MagicMock(role="tool", content="file content", tool_call_id="tc1"),
        ]

        prompt = critic.build_verification_prompt(
            task_objective="Analyze code",
            final_result="Code is clean",
            generator_messages=generator_msgs,
            allow_tool_instructions=True,
        )

        assert "MUST call" in prompt or "call tools" in prompt.lower()


# ================================================================== #
# 5. Decomposer — Uses Uniform Token Budget                          #
# ================================================================== #


class TestDecomposerUsesUniformBudget:
    """Verify Decomposer reads llm_max_output_tokens from config."""

    @pytest.mark.asyncio
    async def test_decomposer_uses_config_budget(self):
        """Decomposer should use llm_max_output_tokens, not hardcoded 512."""
        agent = _make_mock_agent(
            model_name="gpt-5.1",
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                llm_max_output_tokens=4096,
            ),
        )

        analysis_json = json.dumps({
            "gate1": False, "gate2": False, "gate3": False,
            "complexity": "simple",
        })
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=analysis_json))]
            )
        )

        captured_tokens: list[int] = []
        original_call = agent.llm.call

        async def spy_call(**kwargs):
            captured_tokens.append(kwargs.get("max_output_tokens", -1))
            return await original_call(**kwargs)

        agent.llm.call = spy_call

        decomposer = Decomposer()
        task = Task(id="t1", objective="Analyze TCS stock")
        await decomposer.analyze(agent, task)

        assert captured_tokens[0] == 4096

    @pytest.mark.asyncio
    async def test_decomposer_default_2048(self):
        """With default config, Decomposer gets 2048 (not old hardcoded 512)."""
        agent = _make_mock_agent(model_name="gpt-4.1")

        analysis_json = json.dumps({
            "gate1": False, "gate2": False, "gate3": False,
            "complexity": "simple",
        })
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=analysis_json))]
            )
        )

        captured_tokens: list[int] = []
        original_call = agent.llm.call

        async def spy_call(**kwargs):
            captured_tokens.append(kwargs.get("max_output_tokens", -1))
            return await original_call(**kwargs)

        agent.llm.call = spy_call

        await Decomposer().analyze(agent, Task(id="t1", objective="Simple task"))

        assert captured_tokens[0] == 2048

    @pytest.mark.asyncio
    async def test_decomposer_empty_response_defaults_simple(self):
        """Empty response defaults to SIMPLE, not crash."""
        agent = _make_mock_agent(model_name="gpt-5.1")
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=None))]
            )
        )

        result = await Decomposer().analyze(
            agent, Task(id="t1", objective="Complex analysis")
        )

        assert result.is_complex is False


# ================================================================== #
# 6. Synthesis Pass — Fallback on Empty                               #
# ================================================================== #


class TestSynthesisFallback:
    """Verify synthesis pass preserves pre-synthesis content when empty."""

    @pytest.mark.asyncio
    async def test_synthesis_empty_logs_warning(self):
        from nucleusiq.agents.modes.standard_mode import StandardMode

        mode = StandardMode()
        agent = _make_mock_agent(model_name="gpt-5.1")
        agent.config = AgentConfig(
            execution_mode=ExecutionMode.STANDARD,
            llm_max_output_tokens=4096,
            enable_synthesis=True,
        )
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=""))]
            )
        )

        async def mock_call_llm(ag, kwargs, msgs=None, tools=None, purpose=None):
            return await ag.llm.call(**kwargs)

        mode.call_llm = mock_call_llm
        mode.validate_response = MagicMock()
        mode.extract_content = MagicMock(return_value="")
        mode.build_call_kwargs = MagicMock(return_value={"model": "gpt-5.1"})

        result = await mode._synthesis_pass(agent, [])

        assert result == ""
        agent._logger.warning.assert_called_once()
        assert "empty" in agent._logger.warning.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_synthesis_non_empty_returns_new_content(self):
        from nucleusiq.agents.modes.standard_mode import StandardMode

        mode = StandardMode()
        agent = _make_mock_agent(model_name="gpt-4.1")
        agent.config = AgentConfig(
            execution_mode=ExecutionMode.STANDARD,
            llm_max_output_tokens=4096,
        )
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="Synthesized report"))]
            )
        )

        async def mock_call_llm(ag, kwargs, msgs=None, tools=None, purpose=None):
            return await ag.llm.call(**kwargs)

        mode.call_llm = mock_call_llm
        mode.validate_response = MagicMock()
        mode.extract_content = MagicMock(return_value="Synthesized report")
        mode.build_call_kwargs = MagicMock(return_value={"model": "gpt-4.1"})

        result = await mode._synthesis_pass(agent, [])

        assert result == "Synthesized report"

    @pytest.mark.asyncio
    async def test_tool_call_loop_synthesis_fallback(self):
        """When synthesis returns empty, pre-synthesis content is preserved."""
        from nucleusiq.agents.modes.standard_mode import StandardMode

        mode = StandardMode()
        agent = _make_mock_agent(model_name="gpt-5.1")
        agent.config = AgentConfig(
            execution_mode=ExecutionMode.STANDARD,
            max_tool_calls=10,
            llm_max_output_tokens=4096,
            enable_synthesis=True,
        )
        agent.memory = None

        pre_synth_content = "This is the valid pre-synthesis content"
        call_count = [0]

        async def mock_call_llm(ag, kwargs, msgs=None, tools=None, purpose=None):
            call_count[0] += 1
            if call_count[0] <= 3:
                tool_call_fn = MagicMock()
                tool_call_fn.name = "read_file"
                tool_call_fn.arguments = '{"path": "data.csv"}'
                tool_call = MagicMock()
                tool_call.id = f"tc_{call_count[0]}"
                tool_call.type = "function"
                tool_call.function = tool_call_fn
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(
                        content=None, tool_calls=[tool_call], refusal=None,
                    ))]
                )
            elif call_count[0] == 4:
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(
                        content=pre_synth_content, tool_calls=None, refusal=None,
                    ))]
                )
            else:
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(
                        content="", tool_calls=None, refusal=None,
                    ))]
                )

        mode.call_llm = mock_call_llm
        mode.validate_response = MagicMock()
        mode.extract_content = lambda msg: getattr(msg, "content", None) or ""
        mode.build_call_kwargs = MagicMock(return_value={"model": "gpt-5.1"})

        async def mock_process_tool_calls(ag, msg, tc, msgs, tool_round=1):
            msgs.append(ChatMessage(role="tool", content="tool result"))
            return None

        mode._process_tool_calls = mock_process_tool_calls

        task = Task(id="t1", objective="Test")
        messages = [ChatMessage(role="user", content="Test")]

        result = await mode._tool_call_loop(agent, task, messages, [])
        assert result == pre_synth_content


# ================================================================== #
# 7. Empty Retries — Hardcoded to 2                                   #
# ================================================================== #


class TestEmptyRetriesHardcoded:
    """Verify empty retries are hardcoded to 2 (3 total attempts)."""

    @pytest.mark.asyncio
    async def test_three_total_attempts_before_error(self):
        """With 2 retries: 1 initial + 2 retries = 3 calls, then error."""
        from nucleusiq.agents.modes.standard_mode import StandardMode

        mode = StandardMode()
        agent = _make_mock_agent(model_name="gpt-5.1")
        agent.config = AgentConfig(
            execution_mode=ExecutionMode.STANDARD,
            max_tool_calls=10,
        )
        agent.memory = None

        call_count = [0]

        async def mock_call_llm(ag, kwargs, msgs=None, tools=None, purpose=None):
            call_count[0] += 1
            return MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content=None, tool_calls=None, refusal=None,
                ))]
            )

        mode.call_llm = mock_call_llm
        mode.validate_response = MagicMock()
        mode.extract_content = lambda msg: getattr(msg, "content", None) or ""
        mode.build_call_kwargs = MagicMock(return_value={"model": "gpt-5.1"})
        mode.handle_structured_output = MagicMock(return_value=None)
        mode.get_objective = lambda task: getattr(task, "objective", "")

        task = Task(id="t1", objective="Test")
        messages = [ChatMessage(role="user", content="Test")]

        result = await mode._tool_call_loop(agent, task, messages, [])

        assert call_count[0] == 3  # 1 initial + 2 retries
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_recovery_within_retries(self):
        """Model recovers on 3rd attempt (within retry budget)."""
        from nucleusiq.agents.modes.standard_mode import StandardMode

        mode = StandardMode()
        agent = _make_mock_agent(model_name="gpt-5.1")
        agent.config = AgentConfig(
            execution_mode=ExecutionMode.STANDARD,
            max_tool_calls=10,
        )
        agent.memory = None

        call_count = [0]

        async def mock_call_llm(ag, kwargs, msgs=None, tools=None, purpose=None):
            call_count[0] += 1
            if call_count[0] <= 2:
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(
                        content=None, tool_calls=None, refusal=None,
                    ))]
                )
            return MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content="Recovered answer", tool_calls=None, refusal=None,
                ))]
            )

        mode.call_llm = mock_call_llm
        mode.validate_response = MagicMock()
        mode.extract_content = lambda msg: getattr(msg, "content", None) or ""
        mode.build_call_kwargs = MagicMock(return_value={"model": "gpt-5.1"})
        mode.handle_structured_output = MagicMock(return_value=None)

        task = Task(id="t1", objective="Test")
        messages = [ChatMessage(role="user", content="Test")]

        result = await mode._tool_call_loop(agent, task, messages, [])

        assert result == "Recovered answer"
        assert call_count[0] == 3


# ================================================================== #
# 8. Responses API — reasoning_tokens Extraction                      #
# ================================================================== #


class TestResponsesAPIReasoningTokens:
    """Verify Responses API streaming extracts reasoning_tokens."""

    @pytest.mark.asyncio
    async def test_reasoning_tokens_extracted(self):
        from nucleusiq_openai.nb_openai.stream_adapters import (
            _process_responses_events,
        )

        events = []

        text_delta = MagicMock()
        text_delta.type = "response.output_text.delta"
        text_delta.delta = "Hello world"
        events.append(text_delta)

        output_details = MagicMock()
        output_details.reasoning_tokens = 350

        raw_usage = MagicMock()
        raw_usage.input_tokens = 500
        raw_usage.output_tokens = 400
        raw_usage.total_tokens = 900
        raw_usage.output_tokens_details = output_details

        response_obj = MagicMock()
        response_obj.id = "resp_123"
        response_obj.usage = raw_usage

        completed = MagicMock()
        completed.type = "response.completed"
        completed.response = response_obj
        events.append(completed)

        async def _event_gen():
            for e in events:
                yield e

        result_events = []
        async for ev in _process_responses_events(_event_gen()):
            result_events.append(ev)

        complete_events = [e for e in result_events if e.type == "complete"]
        assert len(complete_events) == 1

        usage = complete_events[0].metadata.get("usage", {})
        assert usage.get("reasoning_tokens") == 350

    @pytest.mark.asyncio
    async def test_no_reasoning_tokens_when_absent(self):
        from nucleusiq_openai.nb_openai.stream_adapters import (
            _process_responses_events,
        )

        events = []

        text_delta = MagicMock()
        text_delta.type = "response.output_text.delta"
        text_delta.delta = "Response"
        events.append(text_delta)

        raw_usage = MagicMock()
        raw_usage.input_tokens = 100
        raw_usage.output_tokens = 50
        raw_usage.total_tokens = 150
        raw_usage.output_tokens_details = None

        response_obj = MagicMock()
        response_obj.id = "resp_456"
        response_obj.usage = raw_usage

        completed = MagicMock()
        completed.type = "response.completed"
        completed.response = response_obj
        events.append(completed)

        async def _event_gen():
            for e in events:
                yield e

        result_events = []
        async for ev in _process_responses_events(_event_gen()):
            result_events.append(ev)

        complete_events = [e for e in result_events if e.type == "complete"]
        usage = complete_events[0].metadata.get("usage", {})
        assert "reasoning_tokens" not in usage


# ================================================================== #
# 9. Task C — llm_max_output_tokens Override                          #
# ================================================================== #


class TestTaskCConfig:
    """Verify Task C has explicit llm_max_output_tokens."""

    def test_task_c_has_explicit_token_budget(self):
        mod = importlib.import_module(
            "research.experiments.tasks.task_c_multistep_qa"
        )
        source = inspect.getsource(mod.get_task)
        assert "llm_max_output_tokens" in source
        assert "4096" in source


# ================================================================== #
# 10. End-to-End: Uniform Budget Propagation                          #
# ================================================================== #


class TestUniformBudgetPropagation:
    """Prove that ONE config value propagates to all internal components."""

    @pytest.mark.asyncio
    async def test_critic_gets_same_budget_as_main(self):
        """If user sets llm_max_output_tokens=6000, Critic uses 6000."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = _make_mock_agent(
            model_name="gpt-5.1",
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                llm_max_output_tokens=6000,
            ),
        )

        critique_json = json.dumps({
            "verdict": "pass", "score": 0.92, "feedback": "Thorough analysis.",
        })

        captured_kwargs: dict[str, Any] = {}

        async def spy_call_llm(ag, kwargs, **kw):
            captured_kwargs.update(kwargs)
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content=critique_json))]
            )

        mode.call_llm = spy_call_llm

        result = await mode._run_critic(
            agent, Critic(), "Analyze TCS", "TCS is bullish", [],
        )

        assert captured_kwargs["max_output_tokens"] == 6000
        assert result.verdict == Verdict.PASS
        assert result.score >= 0.9

    @pytest.mark.asyncio
    async def test_decomposer_gets_same_budget_as_main(self):
        """If user sets llm_max_output_tokens=6000, Decomposer uses 6000."""
        agent = _make_mock_agent(
            model_name="o3-mini",
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                llm_max_output_tokens=6000,
            ),
        )

        analysis_json = json.dumps({
            "gate1": False, "gate2": False, "gate3": False,
            "complexity": "simple",
        })
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=analysis_json))]
            )
        )

        captured: list[int] = []
        original = agent.llm.call

        async def spy(**kwargs):
            captured.append(kwargs.get("max_output_tokens", -1))
            return await original(**kwargs)

        agent.llm.call = spy

        await Decomposer().analyze(agent, Task(id="t1", objective="Test"))

        assert captured[0] == 6000

    @pytest.mark.asyncio
    async def test_old_hardcoded_budget_was_too_low(self):
        """Prove that old hardcoded values (Critic=1024, Decomposer=512)
        are below the new default of 2048."""
        old_critic = 1024
        old_decomposer = 512
        new_default = AgentConfig().llm_max_output_tokens

        assert new_default > old_critic
        assert new_default > old_decomposer
        assert new_default == 2048

    @pytest.mark.asyncio
    async def test_full_flow_reasoning_model_success(self):
        """Simulate full Critic flow: with 2048 budget, reasoning model
        has enough room for both internal reasoning and visible JSON."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = _make_mock_agent(model_name="gpt-5.1")

        critique_json = json.dumps({
            "verdict": "pass", "score": 0.92,
            "feedback": "Analysis is thorough.",
        })

        async def budget_aware_llm(ag, kwargs, **kw):
            budget = kwargs.get("max_output_tokens", 0)
            if budget >= 2048:
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(content=critique_json))]
                )
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content=None))]
            )

        mode.call_llm = budget_aware_llm

        result = await mode._run_critic(
            agent, Critic(), "Analyze TCS", "TCS report...", [],
        )

        assert result.verdict == Verdict.PASS
        assert result.score >= 0.9

    @pytest.mark.asyncio
    async def test_old_budget_would_have_failed(self):
        """With old 1024 budget, reasoning model returns empty → UNCERTAIN."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = _make_mock_agent(
            model_name="gpt-5.1",
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                llm_max_output_tokens=1024,
            ),
        )

        async def budget_aware_llm(ag, kwargs, **kw):
            budget = kwargs.get("max_output_tokens", 0)
            if budget <= 1024:
                return MagicMock(
                    choices=[MagicMock(message=MagicMock(content=None))]
                )
            return MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content='{"verdict":"pass","score":0.9,"feedback":"ok"}'
                ))]
            )

        mode.call_llm = budget_aware_llm

        result = await mode._run_critic(
            agent, Critic(), "Task", "Result", [],
        )

        assert result.verdict == Verdict.UNCERTAIN
        assert result.score == 0.5
