"""Extra coverage for standard_mode.py tool processing paths."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.memory.full_history import FullHistoryMemory
from nucleusiq.tools.base_tool import BaseTool


def _make_agent(**overrides):
    defaults = dict(name="A", role="R", objective="O", narrative="N")
    defaults.update(overrides)
    return Agent(**defaults)


def _build_tool_resp(tool_calls):
    msg = MagicMock()
    msg.tool_calls = tool_calls
    msg.refusal = None
    msg.content = None
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _build_text_resp(content):
    msg = MagicMock()
    msg.tool_calls = None
    msg.refusal = None
    msg.content = content
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _make_calc_tool():
    """Create a real BaseTool that won't be flagged as native."""

    def calc(x: int) -> int:
        return x * 2

    return BaseTool.from_function(calc, name="calc", description="double a number")


def _make_bad_tool():
    """Create a real tool that will raise on execution."""

    def bad_tool() -> str:
        raise RuntimeError("boom")

    return BaseTool.from_function(bad_tool, name="bad_tool", description="always fails")


def _make_inc_tool():
    def inc() -> int:
        return 1

    return BaseTool.from_function(inc, name="inc", description="increment")


class TestToolCallProcessing:
    @pytest.mark.asyncio
    async def test_tool_call_then_final_answer(self):
        """LLM returns tool call first, then a final answer."""
        tool = _make_calc_tool()

        tc = {
            "id": "call_1",
            "function": {"name": "calc", "arguments": '{"x": 5}'},
        }

        llm = MockLLM()
        llm.call = AsyncMock(
            side_effect=[
                _build_tool_resp([tc]),
                _build_text_resp("The answer is 10"),
            ]
        )

        agent = _make_agent(llm=llm, tools=[tool])
        await agent.initialize()

        mode = StandardMode()
        result = await mode.run(agent, {"id": "1", "objective": "calc"})
        assert "10" in result

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        tool = _make_bad_tool()

        tc = {
            "id": "call_1",
            "function": {"name": "bad_tool", "arguments": "{}"},
        }

        llm = MockLLM()
        llm.call = AsyncMock(return_value=_build_tool_resp([tc]))

        agent = _make_agent(llm=llm, tools=[tool])
        await agent.initialize()

        mode = StandardMode()
        result = await mode.run(agent, {"id": "1", "objective": "test"})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_max_tool_calls_reached(self):
        """Hit the MAX_TOOL_CALLS limit."""
        tool = _make_inc_tool()

        tc = {"id": "call_1", "function": {"name": "inc", "arguments": "{}"}}

        llm = MockLLM()
        llm.call = AsyncMock(return_value=_build_tool_resp([tc]))

        agent = _make_agent(llm=llm, tools=[tool])
        await agent.initialize()

        mode = StandardMode()
        mode._DEFAULT_MAX_TOOL_CALLS = 2
        result = await mode.run(agent, {"id": "1", "objective": "x"})
        assert "Maximum tool calls" in result

    @pytest.mark.asyncio
    async def test_tool_call_missing_name(self):
        tc = {"id": "call_1", "function": {"name": None, "arguments": "{}"}}

        llm = MockLLM()
        llm.call = AsyncMock(
            side_effect=[
                _build_tool_resp([tc]),
                _build_text_resp("done"),
            ]
        )

        agent = _make_agent(llm=llm)
        await agent.initialize()

        mode = StandardMode()
        result = await mode.run(agent, {"id": "1", "objective": "x"})
        assert result is not None


class TestParseToolCall:
    def test_dict_format(self):
        tc = {"id": "c1", "function": {"name": "f", "arguments": '{"a":1}'}}
        tc_id, fn_name, args = StandardMode._parse_tool_call(tc)
        assert tc_id == "c1"
        assert fn_name == "f"
        assert args == '{"a":1}'

    def test_object_format(self):
        fn = MagicMock()
        fn.name = "f"
        fn.arguments = '{"a":1}'
        tc = MagicMock()
        tc.id = "c1"
        tc.function = fn
        tc_id, fn_name, args = StandardMode._parse_tool_call(tc)
        assert fn_name == "f"


class TestStoreInMemory:
    @pytest.mark.asyncio
    async def test_stores_successfully(self):
        mem = FullHistoryMemory()
        agent = _make_agent(llm=MockLLM(), memory=mem)
        await agent.initialize()

        await StandardMode()._store_in_memory(
            agent, {"id": "1", "objective": "x"}, "answer"
        )
        ctx = mem.get_context()
        assert len(ctx) >= 1
