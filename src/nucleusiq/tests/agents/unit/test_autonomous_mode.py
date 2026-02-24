"""
Tests for AutonomousMode — structured orchestrator over Standard mode.

Covers:
- Task routing: simple vs complex (via Decomposer 3-gate)
- Simple path: execute + validate + retry
- Complex path: decompose → parallel → synthesize + validate
- Validation pipeline integration
- Progress tracking
- Error handling and graceful fallbacks
- Retry message construction
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nucleusiq.agents.components.decomposer import TaskAnalysis
from nucleusiq.agents.components.validation import ValidationResult
from nucleusiq.agents.config.agent_config import (
    AgentConfig,
    AgentState,
    ExecutionMode,
)
from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
from nucleusiq.agents.task import Task

# ================================================================== #
# Helpers                                                              #
# ================================================================== #


def _make_agent(
    tools=None,
    config=None,
    execution_result="answer 42",
):
    """Build a mock agent for AutonomousMode testing."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.role = "Analyst"
    agent.tools = tools or []
    agent._logger = MagicMock()
    agent._last_messages = []
    agent._execution_progress = None
    agent._plugin_manager = None
    agent.config = config or AgentConfig(
        execution_mode=ExecutionMode.AUTONOMOUS,
        max_retries=3,
    )
    agent.state = AgentState.EXECUTING

    llm = MagicMock()
    llm.model_name = "test-model"
    llm.call = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=execution_result))],
        )
    )
    llm.convert_tool_specs = MagicMock(return_value=[])
    agent.llm = llm

    agent._executor = MagicMock()
    agent._executor.execute = AsyncMock(return_value="tool result")
    return agent


def _simple_analysis():
    return TaskAnalysis(is_complex=False)


def _complex_analysis():
    return TaskAnalysis(
        is_complex=True,
        sub_tasks=[
            {"id": "s1", "objective": "Sub-task 1"},
            {"id": "s2", "objective": "Sub-task 2"},
        ],
    )


def _valid_result():
    return ValidationResult(valid=True, layer="all", reason="All checks passed")


def _invalid_result(reason="Tool error detected"):
    return ValidationResult(
        valid=False,
        layer="tool_output",
        reason=reason,
        details=["Error executing tool: connection timeout"],
    )


# ================================================================== #
# Routing: Simple vs Complex                                           #
# ================================================================== #


class TestRouting:
    @patch.object(AutonomousMode, "_run_simple", new_callable=AsyncMock)
    @patch("nucleusiq.agents.modes.autonomous_mode.Decomposer")
    async def test_routes_to_simple_for_simple_task(
        self,
        MockDecomposer,
        mock_simple,
    ):
        mock_simple.return_value = "simple result"
        MockDecomposer.return_value.analyze = AsyncMock(
            return_value=_simple_analysis(),
        )

        agent = _make_agent()
        mode = AutonomousMode()
        result = await mode.run(agent, Task(id="t1", objective="Add 2+3"))
        mock_simple.assert_awaited_once()
        assert result == "simple result"

    @patch.object(AutonomousMode, "_run_complex", new_callable=AsyncMock)
    @patch("nucleusiq.agents.modes.autonomous_mode.Decomposer")
    async def test_routes_to_complex_for_complex_task(
        self,
        MockDecomposer,
        mock_complex,
    ):
        mock_complex.return_value = "complex result"
        MockDecomposer.return_value.analyze = AsyncMock(
            return_value=_complex_analysis(),
        )

        agent = _make_agent()
        mode = AutonomousMode()
        result = await mode.run(agent, Task(id="t1", objective="Compare A, B, C"))
        mock_complex.assert_awaited_once()
        assert result == "complex result"

    @patch.object(AutonomousMode, "_run_simple", new_callable=AsyncMock)
    @patch("nucleusiq.agents.modes.autonomous_mode.Decomposer")
    async def test_routes_to_simple_on_analysis_failure(
        self,
        MockDecomposer,
        mock_simple,
    ):
        """Analysis error → safe fallback to simple path."""
        mock_simple.return_value = "fallback result"
        MockDecomposer.return_value.analyze = AsyncMock(
            return_value=TaskAnalysis(is_complex=False, reasoning="Error"),
        )

        agent = _make_agent()
        mode = AutonomousMode()
        result = await mode.run(agent, Task(id="t1", objective="Any task"))
        mock_simple.assert_awaited_once()

    async def test_no_llm_falls_back_to_standard(self):
        agent = _make_agent()
        agent.llm = None
        mode = AutonomousMode()
        with patch("nucleusiq.agents.modes.autonomous_mode.StandardMode") as MockStd:
            MockStd.return_value.run = AsyncMock(return_value="std result")
            result = await mode.run(agent, Task(id="t1", objective="X"))
            assert result == "std result"

    @patch.object(AutonomousMode, "_run_simple", new_callable=AsyncMock)
    @patch("nucleusiq.agents.modes.autonomous_mode.Decomposer")
    async def test_accepts_dict_task(self, MockDecomposer, mock_simple):
        mock_simple.return_value = "ok"
        MockDecomposer.return_value.analyze = AsyncMock(
            return_value=_simple_analysis(),
        )

        agent = _make_agent()
        mode = AutonomousMode()
        result = await mode.run(agent, {"objective": "Test", "id": "t1"})
        mock_simple.assert_awaited_once()


# ================================================================== #
# Simple Path: execute + validate + retry                              #
# ================================================================== #


class TestSimplePath:
    @patch("nucleusiq.agents.modes.autonomous_mode.ValidationPipeline")
    @patch("nucleusiq.agents.modes.autonomous_mode.StandardMode")
    async def test_returns_on_valid_result(self, MockStd, MockValidation):
        std = MockStd.return_value
        std._ensure_executor = MagicMock()
        std._get_tool_specs = MagicMock(return_value=[])
        std.build_messages = MagicMock(return_value=[])
        std._tool_call_loop = AsyncMock(return_value="result 42")

        MockValidation.return_value.validate = AsyncMock(
            return_value=_valid_result(),
        )

        agent = _make_agent()
        mode = AutonomousMode()
        result = await mode._run_simple(agent, Task(id="t1", objective="X"))
        assert result == "result 42"
        assert agent.state == AgentState.COMPLETED

    @patch("nucleusiq.agents.modes.autonomous_mode.ValidationPipeline")
    @patch("nucleusiq.agents.modes.autonomous_mode.StandardMode")
    async def test_retries_on_validation_failure(self, MockStd, MockValidation):
        std = MockStd.return_value
        std._ensure_executor = MagicMock()
        std._get_tool_specs = MagicMock(return_value=[])
        std.build_messages = MagicMock(return_value=[])
        std._tool_call_loop = AsyncMock(
            side_effect=["wrong answer", "correct answer"],
        )

        MockValidation.return_value.validate = AsyncMock(
            side_effect=[_invalid_result(), _valid_result()],
        )

        agent = _make_agent(config=AgentConfig(max_retries=3))
        mode = AutonomousMode()
        result = await mode._run_simple(agent, Task(id="t1", objective="X"))
        assert result == "correct answer"
        assert std._tool_call_loop.await_count == 2

    @patch("nucleusiq.agents.modes.autonomous_mode.ValidationPipeline")
    @patch("nucleusiq.agents.modes.autonomous_mode.StandardMode")
    async def test_returns_last_result_after_max_retries(
        self,
        MockStd,
        MockValidation,
    ):
        std = MockStd.return_value
        std._ensure_executor = MagicMock()
        std._get_tool_specs = MagicMock(return_value=[])
        std.build_messages = MagicMock(return_value=[])
        std._tool_call_loop = AsyncMock(return_value="best effort")

        MockValidation.return_value.validate = AsyncMock(
            return_value=_invalid_result(),
        )

        agent = _make_agent(config=AgentConfig(max_retries=2))
        mode = AutonomousMode()
        result = await mode._run_simple(agent, Task(id="t1", objective="X"))
        assert result == "best effort"
        assert std._tool_call_loop.await_count == 2
        assert agent.state == AgentState.COMPLETED

    @patch("nucleusiq.agents.modes.autonomous_mode.ValidationPipeline")
    @patch("nucleusiq.agents.modes.autonomous_mode.StandardMode")
    async def test_progress_tracking_on_simple_path(
        self,
        MockStd,
        MockValidation,
    ):
        std = MockStd.return_value
        std._ensure_executor = MagicMock()
        std._get_tool_specs = MagicMock(return_value=[])
        std.build_messages = MagicMock(return_value=[])
        std._tool_call_loop = AsyncMock(return_value="tracked result")

        MockValidation.return_value.validate = AsyncMock(
            return_value=_valid_result(),
        )

        agent = _make_agent()
        mode = AutonomousMode()
        result = await mode._run_simple(agent, Task(id="t1", objective="X"))
        assert result == "tracked result"
        assert agent._execution_progress is not None
        assert len(agent._execution_progress.steps) == 1
        assert agent._execution_progress.steps[0].status == "completed"


# ================================================================== #
# Complex Path: decompose + parallel + synthesize                      #
# ================================================================== #


class TestComplexPath:
    @patch("nucleusiq.agents.modes.autonomous_mode.ValidationPipeline")
    @patch("nucleusiq.agents.modes.autonomous_mode.StandardMode")
    async def test_decompose_synthesize_valid(self, MockStd, MockValidation):
        std = MockStd.return_value
        std._ensure_executor = MagicMock()
        std._get_tool_specs = MagicMock(return_value=[])
        std.build_messages = MagicMock(return_value=[])
        std._tool_call_loop = AsyncMock(return_value="synthesized answer")

        MockValidation.return_value.validate = AsyncMock(
            return_value=_valid_result(),
        )

        decomposer = MagicMock()
        decomposer.run_sub_tasks = AsyncMock(
            return_value=[
                {"id": "s1", "objective": "Part 1", "result": "Data A"},
                {"id": "s2", "objective": "Part 2", "result": "Data B"},
            ]
        )
        decomposer.build_synthesis_prompt = MagicMock(
            return_value="Synthesize A and B",
        )

        agent = _make_agent()
        mode = AutonomousMode()
        result = await mode._run_complex(
            agent,
            Task(id="t1", objective="Compare A and B"),
            decomposer,
            _complex_analysis(),
        )
        assert result == "synthesized answer"
        decomposer.run_sub_tasks.assert_awaited_once()
        assert agent.state == AgentState.COMPLETED

    @patch("nucleusiq.agents.modes.autonomous_mode.ValidationPipeline")
    @patch("nucleusiq.agents.modes.autonomous_mode.StandardMode")
    async def test_complex_retries_synthesis(self, MockStd, MockValidation):
        std = MockStd.return_value
        std._ensure_executor = MagicMock()
        std._get_tool_specs = MagicMock(return_value=[])
        std.build_messages = MagicMock(return_value=[])
        std._tool_call_loop = AsyncMock(
            side_effect=["bad synthesis", "good synthesis"],
        )

        MockValidation.return_value.validate = AsyncMock(
            side_effect=[_invalid_result(), _valid_result()],
        )

        decomposer = MagicMock()
        decomposer.run_sub_tasks = AsyncMock(
            return_value=[
                {"id": "s1", "objective": "Part 1", "result": "Data A"},
            ]
        )
        decomposer.build_synthesis_prompt = MagicMock(
            return_value="Synthesize",
        )

        agent = _make_agent(config=AgentConfig(max_retries=3))
        mode = AutonomousMode()
        result = await mode._run_complex(
            agent,
            Task(id="t1", objective="Analyze"),
            decomposer,
            _complex_analysis(),
        )
        assert result == "good synthesis"
        assert std._tool_call_loop.await_count == 2

    @patch("nucleusiq.agents.modes.autonomous_mode.ValidationPipeline")
    @patch("nucleusiq.agents.modes.autonomous_mode.StandardMode")
    async def test_complex_progress_has_two_steps(
        self,
        MockStd,
        MockValidation,
    ):
        std = MockStd.return_value
        std._ensure_executor = MagicMock()
        std._get_tool_specs = MagicMock(return_value=[])
        std.build_messages = MagicMock(return_value=[])
        std._tool_call_loop = AsyncMock(return_value="result")

        MockValidation.return_value.validate = AsyncMock(
            return_value=_valid_result(),
        )

        decomposer = MagicMock()
        decomposer.run_sub_tasks = AsyncMock(return_value=[])
        decomposer.build_synthesis_prompt = MagicMock(return_value="Synth")

        agent = _make_agent()
        mode = AutonomousMode()
        await mode._run_complex(
            agent,
            Task(id="t1", objective="Compare"),
            decomposer,
            _complex_analysis(),
        )
        progress = agent._execution_progress
        assert progress is not None
        assert len(progress.steps) == 2
        assert progress.steps[0].step_id == "decompose"
        assert progress.steps[1].step_id == "synthesize"


# ================================================================== #
# Error Handling                                                       #
# ================================================================== #


class TestErrorHandling:
    def test_is_error_detects_error_strings(self):
        mode = AutonomousMode()
        assert mode._is_error("Error: something went wrong")
        assert not mode._is_error("The answer is 42")
        assert not mode._is_error("")

    @patch("nucleusiq.agents.modes.autonomous_mode.ValidationPipeline")
    @patch("nucleusiq.agents.modes.autonomous_mode.StandardMode")
    async def test_returns_error_on_execution_exception(
        self,
        MockStd,
        MockValidation,
    ):
        std = MockStd.return_value
        std._ensure_executor = MagicMock()
        std._get_tool_specs = MagicMock(return_value=[])
        std.build_messages = MagicMock(return_value=[])
        std._tool_call_loop = AsyncMock(side_effect=RuntimeError("Boom"))

        agent = _make_agent()
        mode = AutonomousMode()
        result = await mode._run_simple(agent, Task(id="t1", objective="X"))
        assert "Error" in result
        assert agent.state == AgentState.ERROR

    @patch("nucleusiq.agents.modes.autonomous_mode.ValidationPipeline")
    @patch("nucleusiq.agents.modes.autonomous_mode.StandardMode")
    async def test_returns_error_result_directly(
        self,
        MockStd,
        MockValidation,
    ):
        std = MockStd.return_value
        std._ensure_executor = MagicMock()
        std._get_tool_specs = MagicMock(return_value=[])
        std.build_messages = MagicMock(return_value=[])
        std._tool_call_loop = AsyncMock(return_value="Error: tool failed")

        agent = _make_agent()
        mode = AutonomousMode()
        result = await mode._run_simple(agent, Task(id="t1", objective="X"))
        assert result == "Error: tool failed"

    @patch("nucleusiq.agents.modes.autonomous_mode.ValidationPipeline")
    @patch("nucleusiq.agents.modes.autonomous_mode.StandardMode")
    async def test_plugin_halt_propagates(self, MockStd, MockValidation):
        from nucleusiq.plugins.errors import PluginHalt

        std = MockStd.return_value
        std._ensure_executor = MagicMock()
        std._get_tool_specs = MagicMock(return_value=[])
        std.build_messages = MagicMock(return_value=[])
        std._tool_call_loop = AsyncMock(side_effect=PluginHalt("limit"))

        agent = _make_agent()
        mode = AutonomousMode()
        with pytest.raises(PluginHalt):
            await mode._run_simple(agent, Task(id="t1", objective="X"))


# ================================================================== #
# Retry message construction                                           #
# ================================================================== #


class TestRetryMessage:
    def test_builds_message_with_reason(self):
        vr = ValidationResult(
            valid=False,
            layer="tool_output",
            reason="Empty result",
        )
        msg = AutonomousMode._build_retry_message(vr)
        assert "Empty result" in msg
        assert "fix the issue" in msg

    def test_includes_details_when_present(self):
        vr = ValidationResult(
            valid=False,
            layer="plugin",
            reason="Plugin failed",
            details=["Detail A", "Detail B"],
        )
        msg = AutonomousMode._build_retry_message(vr)
        assert "Detail A" in msg
        assert "Detail B" in msg
