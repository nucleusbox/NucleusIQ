"""
Tests for ValidationPipeline — layered validation for autonomous mode.

Covers:
- Layer 1: Tool output checks (empty, error, tool errors in messages)
- Layer 2: Plugin validators (PluginHalt → invalid, exception → skip)
- Layer 3: LLM review (opt-in, PASS/FAIL parsing)
- Pipeline short-circuiting (stops on first failure)
- All layers passing
"""

from unittest.mock import AsyncMock, MagicMock

from nucleusiq.agents.components.validation import (
    ValidationPipeline,
)

# ================================================================== #
# Helpers                                                              #
# ================================================================== #


def _make_agent(validators=None):
    agent = MagicMock()
    agent.name = "test-agent"
    agent.state = "executing"
    agent.config = MagicMock()
    agent.llm = MagicMock()
    agent.plugins = validators or []
    agent._current_task = {"objective": "Test task"}
    return agent


def _msg(role="assistant", content="Result"):
    m = MagicMock()
    m.role = role
    m.content = content
    return m


# ================================================================== #
# Layer 1: Tool output checks                                         #
# ================================================================== #


class TestLayer1ToolOutputChecks:
    def test_empty_result_fails(self):
        vr = ValidationPipeline._check_tool_outputs("", [])
        assert not vr.valid
        assert vr.layer == "tool_output"
        assert "Empty" in vr.reason

    def test_none_result_fails(self):
        vr = ValidationPipeline._check_tool_outputs(None, [])
        assert not vr.valid

    def test_error_result_fails(self):
        vr = ValidationPipeline._check_tool_outputs("Error: connection timeout", [])
        assert not vr.valid
        assert "error" in vr.reason.lower()

    def test_tool_error_in_messages_fails(self):
        msgs = [_msg("tool", "Error executing tool: divide by zero")]
        vr = ValidationPipeline._check_tool_outputs("Some result", msgs)
        assert not vr.valid
        assert "tool error" in vr.reason.lower()
        assert len(vr.details) == 1

    def test_valid_result_passes(self):
        vr = ValidationPipeline._check_tool_outputs("The answer is 42", [])
        assert vr.valid
        assert vr.layer == "tool_output"

    def test_normal_messages_pass(self):
        msgs = [_msg("assistant", "Here is the result: 42")]
        vr = ValidationPipeline._check_tool_outputs("42", msgs)
        assert vr.valid


# ================================================================== #
# Layer 2: Plugin validators                                           #
# ================================================================== #


class TestLayer2PluginValidators:
    async def test_no_validators_passes(self):
        agent = _make_agent(validators=[])
        vr = await ValidationPipeline._run_plugin_validators(agent, "result")
        assert vr.valid
        assert vr.layer == "plugin"

    async def test_passing_validator(self):
        from nucleusiq.plugins.builtin.result_validator import ResultValidatorPlugin

        class PassValidator(ResultValidatorPlugin):
            async def validate_result(self, result, context):
                return True, ""

        agent = _make_agent(validators=[PassValidator()])
        vr = await ValidationPipeline._run_plugin_validators(agent, "result")
        assert vr.valid

    async def test_failing_validator(self):
        from nucleusiq.plugins.builtin.result_validator import ResultValidatorPlugin

        class FailValidator(ResultValidatorPlugin):
            async def validate_result(self, result, context):
                return False, "Result is wrong"

        agent = _make_agent(validators=[FailValidator()])
        vr = await ValidationPipeline._run_plugin_validators(agent, "result")
        assert not vr.valid
        assert vr.layer == "plugin"
        assert "Result is wrong" in vr.reason

    async def test_validator_exception_skipped(self):
        from nucleusiq.plugins.builtin.result_validator import ResultValidatorPlugin

        class BrokenValidator(ResultValidatorPlugin):
            async def validate_result(self, result, context):
                raise RuntimeError("unexpected")

        agent = _make_agent(validators=[BrokenValidator()])
        vr = await ValidationPipeline._run_plugin_validators(agent, "result")
        assert vr.valid

    async def test_non_validator_plugins_ignored(self):
        """Regular plugins (not ResultValidatorPlugin) are skipped by Layer 2."""
        from nucleusiq.plugins.builtin.model_call_limit import ModelCallLimitPlugin

        agent = _make_agent(validators=[ModelCallLimitPlugin(max_calls=10)])
        vr = await ValidationPipeline._run_plugin_validators(agent, "result")
        assert vr.valid


# ================================================================== #
# Layer 3: LLM review (opt-in)                                        #
# ================================================================== #


class TestLayer3LLMReview:
    async def test_llm_review_pass(self):
        agent = _make_agent()
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="PASS"))],
            )
        )

        pipeline = ValidationPipeline(llm_review=True)
        vr = await pipeline._run_llm_review(agent, "result", [_msg("system", "task")])
        assert vr.valid
        assert vr.layer == "llm_review"

    async def test_llm_review_fail(self):
        agent = _make_agent()
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(message=MagicMock(content="FAIL: wrong calculation"))
                ],
            )
        )

        pipeline = ValidationPipeline(llm_review=True)
        vr = await pipeline._run_llm_review(agent, "result", [_msg("system", "task")])
        assert not vr.valid
        assert "FAIL" in vr.reason

    async def test_llm_review_error_skipped(self):
        agent = _make_agent()
        agent.llm.call = AsyncMock(side_effect=RuntimeError("API error"))

        pipeline = ValidationPipeline(llm_review=True)
        vr = await pipeline._run_llm_review(agent, "result", [])
        assert vr.valid

    async def test_no_llm_passes(self):
        agent = _make_agent()
        agent.llm = None

        pipeline = ValidationPipeline(llm_review=True)
        vr = await pipeline._run_llm_review(agent, "result", [])
        assert vr.valid


# ================================================================== #
# Full pipeline integration                                            #
# ================================================================== #


class TestFullPipeline:
    async def test_all_layers_pass(self):
        agent = _make_agent()
        pipeline = ValidationPipeline()
        vr = await pipeline.validate(agent, "The answer is 42", [])
        assert vr.valid

    async def test_layer1_fails_short_circuits(self):
        from nucleusiq.plugins.builtin.result_validator import ResultValidatorPlugin

        class ShouldNotRun(ResultValidatorPlugin):
            async def validate_result(self, result, context):
                raise AssertionError("Should not be called")

        agent = _make_agent(validators=[ShouldNotRun()])
        pipeline = ValidationPipeline()
        vr = await pipeline.validate(agent, "Error: boom", [])
        assert not vr.valid
        assert vr.layer == "tool_output"

    async def test_layer2_fails_short_circuits(self):
        from nucleusiq.plugins.builtin.result_validator import ResultValidatorPlugin

        class FailValidator(ResultValidatorPlugin):
            async def validate_result(self, result, context):
                return False, "bad result"

        agent = _make_agent(validators=[FailValidator()])
        pipeline = ValidationPipeline(llm_review=True)
        vr = await pipeline.validate(agent, "some result", [])
        assert not vr.valid
        assert vr.layer == "plugin"

    async def test_layer3_opt_in_runs(self):
        agent = _make_agent()
        agent.llm.call = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="FAIL: wrong"))],
            )
        )

        pipeline = ValidationPipeline(llm_review=True)
        vr = await pipeline.validate(agent, "some result", [])
        assert not vr.valid
        assert vr.layer == "llm_review"

    async def test_layer3_not_called_when_disabled(self):
        agent = _make_agent()
        agent.llm.call = AsyncMock()

        pipeline = ValidationPipeline(llm_review=False)
        vr = await pipeline.validate(agent, "some result", [])
        assert vr.valid
        agent.llm.call.assert_not_awaited()
