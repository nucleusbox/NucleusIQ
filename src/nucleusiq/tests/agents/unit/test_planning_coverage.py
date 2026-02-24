"""
Full coverage tests for planning modules:
plan_creator.py, plan_executor.py, plan_parser.py, planner.py
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.plan import Plan, PlanResponse, PlanStep
from nucleusiq.agents.planning.plan_creator import PlanCreator
from nucleusiq.agents.planning.plan_executor import PlanExecutor
from nucleusiq.agents.planning.plan_parser import PlanParser
from nucleusiq.agents.planning.planner import Planner
from nucleusiq.llms.mock_llm import MockLLM


def _make_agent(**overrides):
    defaults = dict(
        name="TestAgent",
        role="Tester",
        objective="Test",
        narrative="Tester",
    )
    defaults.update(overrides)
    return Agent(**defaults)


def _mock_response(content=None, tool_calls=None, refusal=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.refusal = refusal
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# PlanParser
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanParser:
    def test_parse_valid_json(self):
        parser = PlanParser()
        json_text = json.dumps(
            {
                "steps": [
                    {
                        "step": 1,
                        "action": "search",
                        "details": "find data",
                        "args": {"query": "test"},
                    }
                ]
            }
        )
        result = parser.parse(json_text)
        assert isinstance(result, PlanResponse)
        assert len(result.steps) == 1

    def test_parse_code_block(self):
        parser = PlanParser()
        text = f"```json\n{json.dumps({'steps': [{'step': 1, 'action': 'go'}]})}\n```"
        result = parser.parse(text)
        assert len(result.steps) == 1

    def test_parse_balanced_json_with_nested(self):
        parser = PlanParser()
        text = json.dumps(
            {
                "steps": [
                    {"step": 1, "action": "run", "args": {"config": {"nested": True}}}
                ]
            }
        )
        result = parser.parse(text)
        assert result.steps[0].action == "run"

    def test_parse_text_plan_fallback(self):
        """When JSON fails, falls back to text parsing."""
        parser = PlanParser()
        text = (
            "Step 1: search for information\n"
            "   Action: search\n"
            "Step 2: summarize\n"
            "   Details about this step\n"
        )
        result = parser.parse(text)
        assert len(result.steps) >= 2

    def test_parse_empty_creates_default_plan(self):
        parser = PlanParser()
        result = parser.parse("")
        assert len(result.steps) == 1
        assert result.steps[0].action == "execute"

    def test_balanced_json_with_escape(self):
        """Tests the escape handling in balanced JSON extractor."""
        parser = PlanParser()
        raw = json.dumps(
            {"steps": [{"step": 1, "action": "test", "details": 'escaped \\"quote\\"'}]}
        )
        result = parser.parse(raw)
        assert len(result.steps) == 1

    def test_text_plan_with_action_detail_lines(self):
        """Lines 124-137: Action in line and detail accumulation."""
        parser = PlanParser()
        text = (
            "Step 1:\n"
            "   action: search_web\n"
            "   Some extra detail line\n"
            "   More details here\n"
            "Step 2: finalize\n"
        )
        result = parser.parse(text)
        steps = result.steps
        assert any(s.action == "search_web" for s in steps)


# ═══════════════════════════════════════════════════════════════════════════════
# PlanCreator
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanCreator:
    @pytest.mark.asyncio
    async def test_no_llm_raises(self):
        creator = PlanCreator()
        agent = _make_agent(llm=None)
        with pytest.raises(ValueError, match="LLM is required"):
            await creator.create_plan(agent, {"id": "1", "objective": "x"}, {})

    @pytest.mark.asyncio
    async def test_create_via_content_fallback(self):
        plan_json = json.dumps({"steps": [{"step": 1, "action": "search", "args": {}}]})
        llm = MockLLM()
        llm.call = AsyncMock(return_value=_mock_response(content=plan_json))
        agent = _make_agent(llm=llm)
        await agent.initialize()

        creator = PlanCreator()
        plan = await creator.create_plan(
            agent, {"id": "1", "objective": "do stuff"}, {}
        )
        assert isinstance(plan, Plan)
        assert len(plan.steps) >= 1

    @pytest.mark.asyncio
    async def test_create_via_tool_call(self):
        plan_data = {"steps": [{"step": 1, "action": "analyze", "args": {}}]}
        tc = {
            "function": {"name": "create_plan", "arguments": json.dumps(plan_data)},
            "id": "call_1",
        }
        llm = MockLLM()
        llm.call = AsyncMock(return_value=_mock_response(tool_calls=[tc]))
        agent = _make_agent(llm=llm)
        await agent.initialize()

        creator = PlanCreator()
        plan = await creator.create_plan(agent, {"id": "1", "objective": "test"}, {})
        assert isinstance(plan, Plan)

    @pytest.mark.asyncio
    async def test_create_tool_call_parse_error_falls_to_content(self):
        tc = {
            "function": {"name": "create_plan", "arguments": "not json"},
            "id": "call_1",
        }
        plan_json = json.dumps({"steps": [{"step": 1, "action": "fallback"}]})
        llm = MockLLM()
        llm.call = AsyncMock(
            return_value=_mock_response(content=plan_json, tool_calls=[tc])
        )
        agent = _make_agent(llm=llm)
        await agent.initialize()

        creator = PlanCreator()
        plan = await creator.create_plan(agent, {"id": "1", "objective": "test"}, {})
        assert isinstance(plan, Plan)

    @pytest.mark.asyncio
    async def test_create_empty_response_retries(self):
        llm = MockLLM()
        plan_json = json.dumps({"steps": [{"step": 1, "action": "ok"}]})
        llm.call = AsyncMock(
            side_effect=[
                _mock_response(content=None),
                _mock_response(content=plan_json),
            ]
        )
        agent = _make_agent(llm=llm)
        await agent.initialize()

        creator = PlanCreator()
        plan = await creator.create_plan(agent, {"id": "1", "objective": "x"}, {})
        assert isinstance(plan, Plan)

    @pytest.mark.asyncio
    async def test_create_refusal(self):
        llm = MockLLM()
        llm.call = AsyncMock(return_value=_mock_response(refusal="I refuse"))
        agent = _make_agent(llm=llm)
        await agent.initialize()

        creator = PlanCreator()
        with pytest.raises(ValueError, match="refused"):
            await creator.create_plan(agent, {"id": "1", "objective": "x"}, {})

    def test_get_tool_param_lines_with_tools(self):
        tool = MagicMock()
        tool.name = "search"
        tool.get_spec = MagicMock(
            return_value={
                "name": "search",
                "parameters": {
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }
        )
        agent = _make_agent(llm=MockLLM(), tools=[tool])
        lines = PlanCreator()._get_tool_param_lines(agent)
        assert len(lines) == 1
        assert "search" in lines[0]

    def test_get_tool_param_lines_error(self):
        tool = MagicMock()
        tool.name = "bad"
        tool.get_spec = MagicMock(side_effect=Exception("broken"))
        agent = _make_agent(llm=MockLLM(), tools=[tool])
        lines = PlanCreator()._get_tool_param_lines(agent)
        assert "bad" in lines[0]

    def test_get_tool_param_lines_empty_props(self):
        tool = MagicMock()
        tool.name = "noop"
        tool.get_spec = MagicMock(return_value={"name": "noop", "parameters": {}})
        agent = _make_agent(llm=MockLLM(), tools=[tool])
        lines = PlanCreator()._get_tool_param_lines(agent)
        assert "noop(...)" in lines[0]

    def test_parse_tool_call_info_dict(self):
        tc = {"function": {"name": "myf", "arguments": '{"x":1}'}}
        name, args = PlanCreator._parse_tool_call_info(tc)
        assert name == "myf"
        assert args == '{"x":1}'

    def test_parse_tool_call_info_object(self):
        fn = MagicMock()
        fn.name = "myf"
        fn.arguments = '{"x":1}'
        tc = MagicMock()
        tc.function = fn
        name, args = PlanCreator._parse_tool_call_info(tc)
        assert name == "myf"


# ═══════════════════════════════════════════════════════════════════════════════
# PlanExecutor
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanExecutor:
    @pytest.mark.asyncio
    async def test_execute_single_step(self):
        llm = MockLLM()
        agent = _make_agent(llm=llm)
        await agent.initialize()
        task = {"id": "1", "objective": "x"}

        plan = Plan(
            steps=[PlanStep(step=1, action="respond", details="test", task=task)],
            task=task,
        )
        executor = PlanExecutor()
        result = await executor.execute_plan(agent, task, plan)
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_with_tool(self):
        from nucleusiq.tools.base_tool import BaseTool

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool = BaseTool.from_function(greet, name="greet", description="Greet someone")

        llm = MockLLM()
        agent = _make_agent(llm=llm, tools=[tool])
        await agent.initialize()
        task = {"id": "1", "objective": "greet"}

        plan = Plan(
            steps=[PlanStep(step=1, action="greet", args={"name": "Alice"}, task=task)],
            task=task,
        )
        executor = PlanExecutor()
        result = await executor.execute_plan(agent, task, plan)
        assert "Hello" in str(result)

    @pytest.mark.asyncio
    async def test_run_with_retries_timeout(self):
        executor = PlanExecutor()
        llm = MockLLM()
        agent = _make_agent(llm=llm)
        await agent.initialize()

        step = PlanStep(step=1, action="slow_tool")
        task = {"id": "1", "objective": "x"}

        async def slow_step(*a, **kw):
            await asyncio.sleep(10)

        with patch.object(executor, "_execute_step", side_effect=slow_step):
            result, error = await executor._run_with_retries(
                agent,
                step,
                1,
                "slow_tool",
                task,
                "",
                {},
                task,
                timeout=1,
                max_retries=0,
            )
        assert error is not None
        assert "timed out" in error

    def test_resolve_arg_value_references(self):
        ctx = {"step_1": "result1", "step_2": "result2"}
        assert PlanExecutor._resolve_arg_value("$step_1", ctx) == "result1"
        assert PlanExecutor._resolve_arg_value("${step_2}", ctx) == "result2"
        assert PlanExecutor._resolve_arg_value("{{step_1}}", ctx) == "result1"

    def test_resolve_arg_value_nested(self):
        ctx = {"step_1": "val1"}
        result = PlanExecutor._resolve_arg_value({"key": "$step_1"}, ctx)
        assert result == {"key": "val1"}

    def test_resolve_arg_value_list(self):
        ctx = {"step_1": "val1"}
        result = PlanExecutor._resolve_arg_value(["$step_1", "literal"], ctx)
        assert result == ["val1", "literal"]

    def test_resolve_args_none(self):
        assert PlanExecutor._resolve_args(None, {}) == {}

    def test_resolve_step_task_with_task(self):
        step = PlanStep(step=1, action="test", task={"id": "s1", "objective": "sub"})
        result = PlanExecutor._resolve_step_task(
            step, {"id": "main", "objective": "main"}
        )
        assert result["objective"] == "sub"

    def test_get_required_keys(self):
        specs = [
            {"function": {"name": "search", "parameters": {"required": ["query"]}}},
            {"function": {"name": "other", "parameters": {"required": ["x"]}}},
        ]
        keys = PlanExecutor._get_required_keys(specs, "search")
        assert keys == ["query"]

    def test_get_required_keys_not_found(self):
        keys = PlanExecutor._get_required_keys([], "missing")
        assert keys == []

    def test_extract_args_from_response_dict_tc(self):
        executor = PlanExecutor()
        tc = {"function": {"name": "search", "arguments": '{"q": "test"}'}, "id": "c1"}
        resp = _mock_response(tool_calls=[tc])
        resp.choices[0].message = {"tool_calls": [tc], "content": None}
        result = executor._extract_args_from_response(resp, "search", 1)
        assert result["q"] == "test"

    def test_extract_args_from_response_no_tool_calls(self):
        executor = PlanExecutor()
        resp = _mock_response(content="no tools")
        with pytest.raises(ValueError, match="no args"):
            executor._extract_args_from_response(resp, "tool", 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Planner facade
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanner:
    @pytest.mark.asyncio
    async def test_create_and_execute_plan(self):
        llm = MockLLM()
        agent = _make_agent(llm=llm)
        await agent.initialize()

        planner = Planner(agent)
        task = {"id": "1", "objective": "test task"}
        context = await planner.get_context(task)
        assert isinstance(context, dict)

        plan = await planner.create_plan(task, context)
        assert isinstance(plan, Plan)

        result = await planner.execute_plan(task, plan)
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_context_with_memory(self):
        from nucleusiq.memory.full_history import FullHistoryMemory

        class MemoryWithRelevant(FullHistoryMemory):
            async def aget_relevant_context(self, query=None, **kwargs):
                return ["previous context"]

        mem = MemoryWithRelevant()
        mem.add_message("user", "hello")
        llm = MockLLM()
        agent = _make_agent(llm=llm, memory=mem)
        await agent.initialize()

        planner = Planner(agent)
        ctx = await planner.get_context({"id": "1", "objective": "test"})
        assert "memory" in ctx
