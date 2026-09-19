"""
Tests for the Decomposer component.

Covers:
- TaskAnalysis dataclass
- Task complexity analysis (simple/complex classification)
- _parse_analysis (JSON parsing, fallbacks)
- Sub-agent creation
- Parallel sub-task execution
- Synthesis prompt construction
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from nucleusiq.agents.components.decomposer import (
    Decomposer,
    SubTaskFinding,
    TaskAnalysis,
    _sub_agent_config,
)
from nucleusiq.agents.config.agent_config import AgentConfig, ExecutionMode
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.task import Task

# ================================================================== #
# Test helpers                                                         #
# ================================================================== #


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLMResponse:
    def __init__(self, content: str):
        self.choices = [MagicMock(message=FakeMessage(content=content))]


def _make_parent(llm_content: str = "", tools=None):
    """Create a mock parent agent."""
    parent = MagicMock()
    parent.name = "test-agent"
    parent.role = "Analyst"
    parent.tools = tools or []
    parent._logger = MagicMock()

    llm = MagicMock()
    llm.model_name = "test-model"
    llm.call = AsyncMock(return_value=FakeLLMResponse(llm_content))
    llm.convert_tool_specs = MagicMock(return_value=[])
    parent.llm = llm
    return parent


# ================================================================== #
# TaskAnalysis                                                         #
# ================================================================== #


class TestTaskAnalysis:
    def test_defaults(self):
        ta = TaskAnalysis(is_complex=False)
        assert not ta.is_complex
        assert ta.sub_tasks == []
        assert ta.reasoning == ""

    def test_complex_with_sub_tasks(self):
        ta = TaskAnalysis(
            is_complex=True,
            sub_tasks=[{"id": "s1", "objective": "Do A"}],
            reasoning="Multiple topics",
        )
        assert ta.is_complex
        assert len(ta.sub_tasks) == 1


# ================================================================== #
# analyze()                                                            #
# ================================================================== #


class TestAnalyze:
    async def test_classifies_simple_task(self):
        parent = _make_parent(json.dumps({"complexity": "simple"}))
        d = Decomposer()
        result = await d.analyze(parent, Task(id="t1", objective="Calculate 2+3"))
        assert not result.is_complex
        assert result.sub_tasks == []

    async def test_classifies_complex_task(self):
        parent = _make_parent(
            json.dumps(
                {
                    "gate1": True,
                    "gate2": True,
                    "gate3": True,
                    "complexity": "complex",
                    "sub_tasks": [
                        {"id": "s1", "objective": "Research topic A"},
                        {"id": "s2", "objective": "Research topic B"},
                    ],
                }
            )
        )
        d = Decomposer()
        result = await d.analyze(parent, Task(id="t1", objective="Compare A and B"))
        assert result.is_complex
        assert len(result.sub_tasks) == 2

    async def test_falls_back_to_simple_on_llm_error(self):
        parent = _make_parent()
        parent.llm.call = AsyncMock(side_effect=RuntimeError("LLM down"))
        d = Decomposer()
        result = await d.analyze(parent, Task(id="t1", objective="Anything"))
        assert not result.is_complex

    async def test_complex_without_subtasks_becomes_simple(self):
        parent = _make_parent(json.dumps({"complexity": "complex"}))
        d = Decomposer()
        result = await d.analyze(parent, Task(id="t1", objective="X"))
        assert not result.is_complex


# ================================================================== #
# _parse_analysis()                                                    #
# ================================================================== #


class TestParseAnalysis:
    def test_parses_simple(self):
        resp = FakeLLMResponse(json.dumps({"complexity": "simple"}))
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert not ta.is_complex

    def test_parses_complex(self):
        resp = FakeLLMResponse(
            json.dumps(
                {
                    "gate1": True,
                    "gate2": True,
                    "gate3": True,
                    "complexity": "complex",
                    "sub_tasks": [
                        {"id": "s1", "objective": "Do A"},
                        {"id": "s2", "objective": "Do B"},
                    ],
                }
            )
        )
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert ta.is_complex
        assert len(ta.sub_tasks) == 2

    def test_handles_unparseable_json(self):
        resp = FakeLLMResponse("Not JSON at all")
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert not ta.is_complex

    def test_handles_empty_response(self):
        resp = FakeLLMResponse("")
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert not ta.is_complex

    def test_complex_with_empty_subtasks_becomes_simple(self):
        resp = FakeLLMResponse(
            json.dumps(
                {
                    "complexity": "complex",
                    "sub_tasks": [],
                }
            )
        )
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert not ta.is_complex

    def test_handles_json_in_markdown(self):
        resp = FakeLLMResponse(
            "Here is my analysis:\n```json\n"
            '{"gate1": true, "gate2": true, "gate3": true, '
            '"complexity": "complex", "sub_tasks": ['
            '{"id": "a", "objective": "X"}, {"id": "b", "objective": "Y"}]}'
            "\n```"
        )
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert ta.is_complex


# ================================================================== #
# Gate Enforcement                                                      #
# ================================================================== #


class TestGateEnforcement:
    """Verify the three-gate checklist is enforced structurally."""

    def test_gate2_false_overrides_complex_to_simple(self):
        resp = FakeLLMResponse(
            json.dumps(
                {
                    "gate1": True,
                    "gate2": False,
                    "gate3": True,
                    "complexity": "complex",
                    "sub_tasks": [
                        {"id": "s1", "objective": "A"},
                        {"id": "s2", "objective": "B"},
                    ],
                }
            )
        )
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert not ta.is_complex
        assert ta.sub_tasks == []

    def test_gate1_false_overrides_complex_to_simple(self):
        resp = FakeLLMResponse(
            json.dumps(
                {
                    "gate1": False,
                    "gate2": True,
                    "gate3": True,
                    "complexity": "complex",
                    "sub_tasks": [
                        {"id": "s1", "objective": "A"},
                        {"id": "s2", "objective": "B"},
                    ],
                }
            )
        )
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert not ta.is_complex

    def test_gate3_false_overrides_complex_to_simple(self):
        resp = FakeLLMResponse(
            json.dumps(
                {
                    "gate1": True,
                    "gate2": True,
                    "gate3": False,
                    "complexity": "complex",
                    "sub_tasks": [
                        {"id": "s1", "objective": "A"},
                        {"id": "s2", "objective": "B"},
                    ],
                }
            )
        )
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert not ta.is_complex

    def test_all_gates_true_with_enough_subtasks_is_complex(self):
        resp = FakeLLMResponse(
            json.dumps(
                {
                    "gate1": True,
                    "gate2": True,
                    "gate3": True,
                    "complexity": "complex",
                    "sub_tasks": [
                        {"id": "s1", "objective": "A"},
                        {"id": "s2", "objective": "B"},
                    ],
                }
            )
        )
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert ta.is_complex
        assert len(ta.sub_tasks) == 2

    def test_all_gates_true_but_one_subtask_becomes_simple(self):
        resp = FakeLLMResponse(
            json.dumps(
                {
                    "gate1": True,
                    "gate2": True,
                    "gate3": True,
                    "complexity": "complex",
                    "sub_tasks": [{"id": "s1", "objective": "A"}],
                }
            )
        )
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert not ta.is_complex
        assert ta.sub_tasks == []

    def test_missing_gates_default_to_false(self):
        resp = FakeLLMResponse(
            json.dumps(
                {
                    "complexity": "complex",
                    "sub_tasks": [
                        {"id": "s1", "objective": "A"},
                        {"id": "s2", "objective": "B"},
                    ],
                }
            )
        )
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert not ta.is_complex

    def test_simple_classification_ignores_gates(self):
        resp = FakeLLMResponse(
            json.dumps(
                {
                    "gate1": True,
                    "gate2": True,
                    "gate3": True,
                    "complexity": "simple",
                }
            )
        )
        d = Decomposer()
        ta = d._parse_analysis(resp)
        assert not ta.is_complex


# ================================================================== #
# create_sub_agent()                                                   #
# ================================================================== #


class TestCreateSubAgent:
    @patch("nucleusiq.agents.agent.Agent")
    async def test_creates_agent_with_correct_name(self, MockAgent):
        instance = MagicMock()
        instance.initialize = AsyncMock()
        MockAgent.return_value = instance

        parent = _make_parent()
        agent = await Decomposer.create_sub_agent(parent, "sub1", "Do something")
        assert agent is instance
        call_kwargs = MockAgent.call_args
        name = call_kwargs.kwargs.get("name", "")
        assert "sub1" in name

    @patch("nucleusiq.agents.agent.Agent")
    async def test_sub_agent_has_standard_mode(self, MockAgent):
        instance = MagicMock()
        instance.initialize = AsyncMock()
        MockAgent.return_value = instance

        parent = _make_parent()
        await Decomposer.create_sub_agent(parent, "sub1", "Do something")
        call_kwargs = MockAgent.call_args
        config = call_kwargs.kwargs.get("config")
        assert config.execution_mode.value == "standard"

    @patch("nucleusiq.agents.agent.Agent")
    async def test_sub_agent_inherits_parent_context_budget(self, MockAgent):
        """COMPLEX children must not fall back to the 8K default window."""
        instance = MagicMock()
        instance.initialize = AsyncMock()
        MockAgent.return_value = instance

        parent = _make_parent()
        parent.config = AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            context=ContextConfig(
                max_context_tokens=65_536,
                optimal_budget=48_000,
                response_reserve=4_096,
            ),
        )
        await Decomposer.create_sub_agent(parent, "sub1", "Do something")
        config = MockAgent.call_args.kwargs["config"]
        assert config.execution_mode is ExecutionMode.STANDARD
        assert config.context is not None
        assert config.context.max_context_tokens == 65_536
        assert config.context.optimal_budget == 48_000
        assert config.context.response_reserve == 4_096
        assert config.context is not parent.config.context

    @patch("nucleusiq.agents.agent.Agent")
    async def test_sub_agent_context_override_wins(self, MockAgent):
        instance = MagicMock()
        instance.initialize = AsyncMock()
        MockAgent.return_value = instance

        parent = _make_parent()
        parent.config = AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            context=ContextConfig(max_context_tokens=65_536),
            sub_agent_context=ContextConfig(
                max_context_tokens=32_768,
                optimal_budget=24_000,
            ),
        )
        await Decomposer.create_sub_agent(parent, "sub1", "Do something")
        config = MockAgent.call_args.kwargs["config"]
        assert config.context.max_context_tokens == 32_768
        assert config.context.optimal_budget == 24_000

    @patch("nucleusiq.agents.agent.Agent")
    async def test_sub_agent_inherits_max_tool_calls(self, MockAgent):
        instance = MagicMock()
        instance.initialize = AsyncMock()
        MockAgent.return_value = instance

        parent = _make_parent()
        parent.config = AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            max_tool_calls=40,
            max_retries=2,
        )
        await Decomposer.create_sub_agent(parent, "sub1", "Do something")
        kwargs = MockAgent.call_args.kwargs
        config = kwargs["config"]
        assert config.max_tool_calls == 40
        assert config.max_retries == 2
        assert config.get_effective_max_tool_calls() == 40
        assert "plugins" not in kwargs or not kwargs["plugins"]

    @patch("nucleusiq.agents.agent.Agent")
    async def test_sub_agent_unset_max_tool_calls_uses_standard_default(
        self, MockAgent
    ):
        instance = MagicMock()
        instance.initialize = AsyncMock()
        MockAgent.return_value = instance

        parent = _make_parent()
        parent.config = AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS)
        await Decomposer.create_sub_agent(parent, "sub1", "Do something")
        config = MockAgent.call_args.kwargs["config"]
        assert config.max_tool_calls is None
        assert config.get_effective_max_tool_calls() == 80

    @patch("nucleusiq.agents.agent.Agent")
    async def test_sub_agent_has_no_memory(self, MockAgent):
        instance = MagicMock()
        instance.initialize = AsyncMock()
        MockAgent.return_value = instance

        parent = _make_parent()
        await Decomposer.create_sub_agent(parent, "sub1", "Do something")
        call_kwargs = MockAgent.call_args
        memory = call_kwargs.kwargs.get("memory")
        assert memory is None

    @patch("nucleusiq.agents.agent.Agent")
    async def test_returns_none_on_failure(self, MockAgent):
        MockAgent.side_effect = RuntimeError("Bad agent")
        parent = _make_parent()
        agent = await Decomposer.create_sub_agent(parent, "sub1", "X")
        assert agent is None


# ================================================================== #
# run_sub_tasks()                                                      #
# ================================================================== #


class TestRunSubTasks:
    @patch.object(Decomposer, "create_sub_agent")
    async def test_runs_sub_tasks_and_collects_results(self, mock_create):
        sub_agent = MagicMock()
        sub_agent.execute = AsyncMock(return_value="Finding A")
        mock_create.return_value = sub_agent

        d = Decomposer()
        parent = _make_parent()
        findings = await d.run_sub_tasks(
            parent,
            [{"id": "s1", "objective": "Research A"}],
        )
        assert len(findings) == 1
        assert findings[0]["result"] == "Finding A"
        assert findings[0]["id"] == "s1"

    @patch.object(Decomposer, "create_sub_agent")
    async def test_caps_sub_agents(self, mock_create):
        sub_agent = MagicMock()
        sub_agent.execute = AsyncMock(return_value="ok")
        mock_create.return_value = sub_agent

        d = Decomposer()
        parent = _make_parent()
        many_tasks = [{"id": f"s{i}", "objective": f"Task {i}"} for i in range(10)]
        findings = await d.run_sub_tasks(parent, many_tasks, max_sub_agents=3)
        assert len(findings) == 3

    @patch.object(Decomposer, "create_sub_agent")
    async def test_handles_sub_agent_failure(self, mock_create):
        sub_agent = MagicMock()
        sub_agent.execute = AsyncMock(side_effect=RuntimeError("Boom"))
        mock_create.return_value = sub_agent

        d = Decomposer()
        parent = _make_parent()
        findings = await d.run_sub_tasks(
            parent,
            [{"id": "s1", "objective": "Do X"}],
        )
        assert len(findings) == 1
        assert "Error" in findings[0]["result"]

    @patch.object(Decomposer, "create_sub_agent")
    async def test_skips_failed_agent_creation(self, mock_create):
        mock_create.return_value = None

        d = Decomposer()
        parent = _make_parent()
        findings = await d.run_sub_tasks(
            parent,
            [{"id": "s1", "objective": "Do X"}],
        )
        assert findings == []

    @patch.object(Decomposer, "create_sub_agent")
    async def test_parallel_execution(self, mock_create):
        """Verify multiple sub-agents run concurrently."""
        import asyncio

        async def slow_execute(task):
            await asyncio.sleep(0.05)
            return f"Result for {task.objective}"

        sub_agent = MagicMock()
        sub_agent.execute = slow_execute
        mock_create.return_value = sub_agent

        d = Decomposer()
        parent = _make_parent()
        tasks = [{"id": f"s{i}", "objective": f"Task {i}"} for i in range(3)]
        findings = await d.run_sub_tasks(parent, tasks)
        assert len(findings) == 3


# ================================================================== #
# build_synthesis_prompt()                                             #
# ================================================================== #


class TestBuildSynthesisPrompt:
    def test_includes_original_task(self):
        prompt = Decomposer.build_synthesis_prompt(
            "Analyze market trends",
            [{"objective": "Sub A", "result": "Finding A"}],
        )
        assert "Analyze market trends" in prompt

    def test_includes_all_findings(self):
        findings = [
            {"objective": "Topic A", "result": "Result A"},
            {"objective": "Topic B", "result": "Result B"},
        ]
        prompt = Decomposer.build_synthesis_prompt("Task", findings)
        assert "Topic A" in prompt
        assert "Result A" in prompt
        assert "Topic B" in prompt
        assert "Result B" in prompt

    def test_instructs_synthesis(self):
        prompt = Decomposer.build_synthesis_prompt(
            "Task",
            [{"objective": "X", "result": "Y"}],
        )
        assert "SYNTHESIS INSTRUCTIONS" in prompt
        assert "FINAL ANSWER" in prompt
        assert "RE-READ the original task" in prompt

    def test_handles_empty_findings(self):
        prompt = Decomposer.build_synthesis_prompt("Task", [])
        assert "ORIGINAL TASK" in prompt


# ================================================================== #
# Config integration                                                   #
# ================================================================== #


class TestConfigFields:
    def test_max_sub_agents_default(self):
        config = AgentConfig()
        assert config.max_sub_agents == 5
        assert config.sub_agent_context is None

    def test_sub_agent_config_without_parent_config_stays_standard(self):
        parent = _make_parent()
        cfg = _sub_agent_config(parent)
        assert cfg.execution_mode is ExecutionMode.STANDARD
        assert cfg.context is None

    def test_sub_agent_inherits_resolved_engine_window_when_context_unset(self):
        """Child must not re-hit DEFAULT_CONTEXT_WINDOW (8192) if the parent
        engine already sized a real window."""
        parent = _make_parent()
        parent.config = AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS)
        parent._context_engine = MagicMock(resolved_max_tokens=65_536)
        cfg = _sub_agent_config(parent)
        assert cfg.context is not None
        assert cfg.context.max_context_tokens == 65_536

    def test_llm_review_default(self):
        config = AgentConfig()
        assert config.llm_review is False

    def test_custom_values(self):
        config = AgentConfig(max_sub_agents=10, llm_review=True)
        assert config.max_sub_agents == 10
        assert config.llm_review is True


# ================================================================== #
# Child failure visibility + auxiliary child tool budget               #
# ================================================================== #


class _ErrorResult:
    """Shape of an ``AgentResult`` whose child ended in error: no output,
    a status enum, a termination reason and the error text."""

    class _Status:
        value = "error"

    status = _Status()
    termination_reason = "error"
    error = (
        "Agent 'x-sub-coverage-followup' has 19 tools but STANDARD mode allows max 8."
    )
    output = None

    def __str__(self) -> str:
        return ""


class TestChildFailureVisibility:
    def test_absorb_result_keeps_the_error_text(self):
        finding = SubTaskFinding(id="coverage-followup", objective="o")
        finding.absorb_result(_ErrorResult())
        assert finding.status == "error"
        assert finding.termination_reason == "error"
        assert finding.error.startswith("Agent 'x-sub-coverage-followup' has 19 tools")
        # ``str(raw)`` was "" — the result now carries the reason so
        # synthesis can state the gap instead of seeing an empty finding.
        assert finding.result == f"Error: {finding.error}"
        assert finding.failed
        assert finding.to_dict()["error"] == finding.error

    def test_absorb_result_prefers_real_output_when_present(self):
        class _Ok:
            class _Status:
                value = "success"

            status = _Status()
            termination_reason = "completed"
            error = None

            def __str__(self) -> str:
                return "nine records"

        finding = SubTaskFinding(id="b1", objective="o")
        finding.absorb_result(_Ok())
        assert finding.result == "nine records"
        assert finding.error == ""
        assert not finding.failed
        assert "error" not in finding.to_dict()


class TestAuxChildToolBudget:
    """A follow-up over four resources gets ``2 × 4 = 8`` calls — but the
    parent hands it every business tool it has.  ``Agent.execute`` treats
    ``max_tool_calls`` as the maximum number of user tools, so the child
    died before its first LLM call.  The budget must cover the tool list."""

    @staticmethod
    def _tools(names):
        out = []
        for n in names:
            tool = MagicMock()
            tool.name = n
            out.append(tool)
        return out

    def test_budget_is_raised_to_the_tool_count(self):
        tools = self._tools([f"tool_{i}" for i in range(19)])
        assert Decomposer._aux_tool_budget(tools, 8) == 19

    def test_budget_is_kept_when_it_already_covers_the_tools(self):
        tools = self._tools(["read_document"])
        assert Decomposer._aux_tool_budget(tools, 8) == 8

    def test_context_management_tools_do_not_count(self):
        tools = self._tools(
            [
                "read_document",
                "recall_tool_result",
                "search_document_corpus",
                "list_evidence",
                "write_workspace_note",
            ]
        )
        assert Decomposer._aux_tool_budget(tools, 2) == 2

    def test_never_below_one(self):
        assert Decomposer._aux_tool_budget([], 0) == 1
