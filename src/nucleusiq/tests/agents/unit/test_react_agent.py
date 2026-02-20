"""Tests for ReActAgent."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional

from nucleusiq.agents.react_agent import ReActAgent
from nucleusiq.agents.config import AgentState
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.prompts.zero_shot import ZeroShotPrompt


def _make_agent(**overrides):
    defaults = dict(
        name="TestReAct",
        role="Assistant",
        objective="Answer questions",
        narrative="A test agent",
    )
    defaults.update(overrides)
    return ReActAgent(**defaults)


def _mock_llm_returning(*messages):
    """Create a MockLLM subclass that returns preset messages in sequence."""
    class SequenceLLM(MockLLM):
        def __init__(self):
            super().__init__()
            self._responses = list(messages)
            self._idx = 0

        async def call(self, **kwargs):
            if self._idx < len(self._responses):
                msg = self._responses[self._idx]
                self._idx += 1
            else:
                msg = MockLLM.Message(content="Echo: done")
            return MockLLM.LLMResponse([MockLLM.Choice(msg)])

        def convert_tool_specs(self, tools):
            return [{"name": "t"}] if tools else []

    return SequenceLLM()


class TestReActAgentNoLLM:

    @pytest.mark.asyncio
    async def test_echo_without_llm_dict(self):
        agent = _make_agent()
        result = await agent.execute({"id": "t1", "objective": "Hello"})
        assert "Echo" in result
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_echo_without_llm_task_object(self):
        agent = _make_agent()

        class FakeTask:
            objective = "Hi"

        result = await agent.execute(FakeTask())
        assert "Echo" in result


class TestReActAgentWithMockLLM:

    @pytest.mark.asyncio
    async def test_final_answer_in_content(self):
        msg = MockLLM.Message(
            content="Thought: I know the answer.\nAction: Final Answer\nAction Input: 42"
        )
        llm = _mock_llm_returning(msg)
        agent = _make_agent(llm=llm)
        result = await agent.execute({"id": "t1", "objective": "What is 6*7?"})
        assert "42" in result
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_tool_call_then_answer(self):
        tool_msg = MockLLM.Message(
            content="Thought: I need to calculate.",
            function_call={"name": "calc", "arguments": json.dumps({"a": 5, "b": 3})},
        )
        final_msg = MockLLM.Message(
            content="Thought: Got it.\nAction: Final Answer\nAction Input: 8"
        )
        llm = _mock_llm_returning(tool_msg, final_msg)

        tool = MagicMock()
        tool.name = "calc"
        tool.is_native = False
        tool.execute = AsyncMock(return_value="8")

        agent = _make_agent(llm=llm, tools=[tool])
        result = await agent.execute({"id": "t1", "objective": "5+3"})
        assert "8" in result

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        loop_msg = MockLLM.Message(
            content="Thought: thinking...\nAction: calc\nAction Input: {}"
        )

        class LoopLLM(MockLLM):
            async def call(self, **kwargs):
                return MockLLM.LLMResponse([MockLLM.Choice(loop_msg)])

            def convert_tool_specs(self, tools):
                return []

        tool = MagicMock()
        tool.name = "calc"
        tool.is_native = False
        tool.execute = AsyncMock(return_value="done")

        agent = _make_agent(llm=LoopLLM(), tools=[tool], max_iterations=2)
        result = await agent.execute({"id": "t1", "objective": "loop forever"})
        assert "Max iterations" in result

    @pytest.mark.asyncio
    async def test_empty_response(self):
        class EmptyLLM(MockLLM):
            async def call(self, **kwargs):
                return MockLLM.LLMResponse([])

            def convert_tool_specs(self, tools):
                return []

        agent = _make_agent(llm=EmptyLLM())
        with pytest.raises(ValueError, match="empty response"):
            await agent.execute({"id": "t1", "objective": "x"})

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        tool_msg = MockLLM.Message(
            content="Thought: need tool.",
            function_call={"name": "nonexistent", "arguments": "{}"},
        )
        final_msg = MockLLM.Message(
            content="Thought: ok.\nAction: Final Answer\nAction Input: error handled"
        )
        llm = _mock_llm_returning(tool_msg, final_msg)
        agent = _make_agent(llm=llm, tools=[])
        result = await agent.execute({"id": "t1", "objective": "x"})
        assert "error handled" in result.lower()

    @pytest.mark.asyncio
    async def test_native_tool_rejected(self):
        tool_msg = MockLLM.Message(
            content="Thought: use native.",
            function_call={"name": "web", "arguments": "{}"},
        )
        final_msg = MockLLM.Message(
            content="Action: Final Answer\nAction Input: done"
        )
        llm = _mock_llm_returning(tool_msg, final_msg)

        native_tool = MagicMock()
        native_tool.name = "web"
        native_tool.is_native = True

        agent = _make_agent(llm=llm, tools=[native_tool])
        result = await agent.execute({"id": "t1", "objective": "search"})
        assert "done" in result

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        tool_msg = MockLLM.Message(
            content="Thought: calc.",
            function_call={"name": "calc", "arguments": "{}"},
        )
        final_msg = MockLLM.Message(
            content="Action: Final Answer\nAction Input: handled"
        )
        llm = _mock_llm_returning(tool_msg, final_msg)

        tool = MagicMock()
        tool.name = "calc"
        tool.is_native = False
        tool.execute = AsyncMock(side_effect=RuntimeError("boom"))

        agent = _make_agent(llm=llm, tools=[tool])
        result = await agent.execute({"id": "t1", "objective": "x"})
        assert "handled" in result


class TestReActParseHelpers:

    def test_extract_content_dict(self):
        agent = _make_agent()
        assert agent._extract_content({"content": "hello"}) == "hello"

    def test_extract_content_object(self):
        agent = _make_agent()
        m = MagicMock()
        m.content = "world"
        assert agent._extract_content(m) == "world"

    def test_extract_content_none(self):
        agent = _make_agent()
        assert agent._extract_content({"role": "system"}) == ""

    def test_parse_response_no_action(self):
        agent = _make_agent()
        thought, action = agent._parse_react_response(
            "Just some text", MagicMock(function_call=None)
        )
        assert action["type"] == "final_answer"

    def test_parse_response_function_call_dict(self):
        agent = _make_agent()
        msg = {"function_call": {"name": "tool1", "arguments": '{"x": 1}'}}
        _, action = agent._parse_react_response("Thought: x", msg)
        assert action["type"] == "tool"
        assert action["name"] == "tool1"

    def test_parse_response_bad_json_args(self):
        agent = _make_agent()
        msg = {"function_call": {"name": "t", "arguments": "invalid"}}
        _, action = agent._parse_react_response("Thought: x", msg)
        assert action["args"] == {}

    def test_parse_response_answer_synonyms(self):
        agent = _make_agent()
        content = "Thought: done.\nAction: answer\nAction Input: 42"
        _, action = agent._parse_react_response(
            content, MagicMock(function_call=None)
        )
        assert action["type"] == "final_answer"

    def test_parse_response_unknown_action_bad_json(self):
        agent = _make_agent()
        content = "Thought: hm\nAction: something\nAction Input: not json"
        _, action = agent._parse_react_response(
            content, MagicMock(function_call=None)
        )
        assert action["type"] == "unknown"


class TestReActHistoryAccessors:

    @pytest.mark.asyncio
    async def test_history_empty(self):
        agent = _make_agent()
        assert agent.get_react_history() == []
        assert agent.get_last_thought() is None
        assert agent.get_last_action() is None

    @pytest.mark.asyncio
    async def test_history_populated(self):
        msg = MockLLM.Message(
            content="Thought: done.\nAction: Final Answer\nAction Input: ok"
        )
        llm = _mock_llm_returning(msg)
        agent = _make_agent(llm=llm)
        await agent.execute({"id": "t1", "objective": "test"})
        history = agent.get_react_history()
        assert len(history) == 1
        assert agent.get_last_thought() is not None
        assert agent.get_last_action()["type"] == "final_answer"


class TestBuildReactMessages:

    def test_basic_dict_task(self):
        agent = _make_agent()
        msgs = agent._build_react_messages({"objective": "Do something"})
        assert msgs[0].role == "system"
        assert "ReAct" in (msgs[0].content or "")
        assert (msgs[-1].content or "") == "Task: Do something"

    def test_with_prompt(self):
        prompt = ZeroShotPrompt()
        prompt.configure(system="Be helpful", user="User context")
        agent = _make_agent(prompt=prompt)
        msgs = agent._build_react_messages({"objective": "x"})
        systems = [m for m in msgs if m.role == "system"]
        assert len(systems) >= 2

    def test_task_object(self):
        agent = _make_agent()

        class FakeTask:
            objective = "ObjectiveFromObj"

        msgs = agent._build_react_messages(FakeTask())
        assert any("ObjectiveFromObj" in (m.content or "") for m in msgs)
