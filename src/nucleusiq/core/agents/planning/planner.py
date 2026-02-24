"""
Planner — thin facade over ``PlanCreator`` and ``PlanExecutor``.

This class keeps the existing public API (``Planner(agent).create_plan()``,
``Planner(agent).execute_plan()``, etc.) while delegating the heavy lifting
to the focused sub-components.

Static helpers (``get_plan_schema``, ``get_plan_function_spec``) remain
here since they are part of the planning contract.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

from nucleusiq.agents.plan import Plan
from nucleusiq.agents.planning.plan_creator import PlanCreator
from nucleusiq.agents.planning.plan_executor import PlanExecutor
from nucleusiq.agents.planning.prompt_strategy import (
    DefaultPlanPromptStrategy,
    PlanPromptStrategy,
)
from nucleusiq.agents.planning.schema import (
    get_plan_function_spec as _get_plan_function_spec,
)
from nucleusiq.agents.planning.schema import (
    get_plan_schema as _get_plan_schema,
)
from nucleusiq.agents.task import Task

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent


class Planner:
    """Facade that unifies plan creation, execution, and context gathering.

    Usage::

        planner = Planner(agent)
        context = await planner.get_context(task)
        plan = await planner.create_plan(task, context)
        result = await planner.execute_plan(task, plan)

    Custom prompt strategy::

        from nucleusiq.agents.planning.prompt_strategy import PlanPromptStrategy

        class MyPromptStrategy:
            def build_planning_prompt(self, ...): ...
            def build_tool_call_prompt(self, ...): ...
            def build_step_inference_prompt(self, ...): ...

        planner = Planner(agent, prompt_strategy=MyPromptStrategy())
    """

    def __init__(
        self,
        agent: "Agent",
        prompt_strategy: PlanPromptStrategy | None = None,
    ):
        self._agent = agent
        self._logger = agent._logger
        self._prompt_strategy = prompt_strategy or DefaultPlanPromptStrategy()
        self._creator = PlanCreator(
            logger=self._logger,
            prompt_strategy=self._prompt_strategy,
        )
        self._executor = PlanExecutor(
            logger=self._logger,
            prompt_strategy=self._prompt_strategy,
        )

    # ------------------------------------------------------------------ #
    # Static schema helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_plan_schema() -> Dict[str, Any]:
        """Minimal, OpenAI-friendly JSON schema for plan output.

        Delegates to :func:`planning.schema.get_plan_schema`.
        """
        return _get_plan_schema()

    @staticmethod
    def get_plan_function_spec() -> Dict[str, Any]:
        """Function specification for structured plan generation.

        Delegates to :func:`planning.schema.get_plan_function_spec`.
        """
        return _get_plan_function_spec()

    # ------------------------------------------------------------------ #
    # Context                                                             #
    # ------------------------------------------------------------------ #

    async def get_context(self, task: Task | Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context for task execution."""
        task_dict = task.to_dict() if isinstance(task, Task) else task
        context: Dict[str, Any] = {
            "task": task_dict,
            "agent_role": self._agent.role,
            "agent_objective": self._agent.objective,
            "timestamp": datetime.now().isoformat(),
        }
        if self._agent.memory:
            memory_context = await self._agent.memory.aget_relevant_context(task_dict)
            context["memory"] = memory_context
        return context

    # ------------------------------------------------------------------ #
    # Delegated methods                                                   #
    # ------------------------------------------------------------------ #

    async def create_plan(
        self,
        task: Task | Dict[str, Any],
        context: Dict[str, Any],
    ) -> Plan:
        """Create an execution plan using the LLM."""
        return await self._creator.create_plan(self._agent, task, context)

    async def execute_plan(
        self,
        task: Task | Dict[str, Any],
        plan: Plan | List[Dict[str, Any]],
    ) -> Any:
        """Execute a task following a multi-step plan."""
        return await self._executor.execute_plan(self._agent, task, plan)

    def construct_planning_prompt(
        self,
        task: Task | Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """Build the structured planning prompt."""
        task_dict = task.to_dict() if isinstance(task, Task) else task
        task_obj = task_dict.get("objective", str(task_dict))

        return self._prompt_strategy.build_planning_prompt(
            task_objective=task_obj,
            tool_names=[
                getattr(t, "name", "unknown") for t in (self._agent.tools or [])
            ],
            role=self._agent.role,
            objective=self._agent.objective,
        )
