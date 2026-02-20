"""
PlanExecutor — executes multi-step plans step-by-step.

Extracted from ``Planner.execute_plan()`` and ``Planner._execute_step()``.

Responsibilities:
- Walk through a ``Plan``'s steps sequentially
- Execute each step (direct LLM call or tool call)
- Resolve ``$step_N`` references between steps
- Handle timeouts, retries, and error propagation
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from nucleusiq.agents.task import Task
from nucleusiq.agents.plan import Plan, PlanStep
from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.planning.prompt_strategy import (
    PlanPromptStrategy,
    DefaultPlanPromptStrategy,
)
from nucleusiq.agents.chat_models import ToolCallRequest

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent


class PlanExecutor:
    """Executes a ``Plan`` step-by-step against an ``Agent``."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        prompt_strategy: Optional[PlanPromptStrategy] = None,
        step_runner: Optional[Any] = None,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._prompt_strategy = prompt_strategy or DefaultPlanPromptStrategy()
        self._step_runner = step_runner

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def execute_plan(
        self,
        agent: "Agent",
        task: Union[Task, Dict[str, Any]],
        plan: Union[Plan, List[Dict[str, Any]]],
    ) -> Any:
        """Execute a task following a multi-step plan.

        Args:
            agent: The Agent instance
            task: Original task (Task or dict)
            plan: Plan instance or list of step dicts

        Returns:
            Final result from the last step
        """
        # Normalise inputs
        if isinstance(plan, list):
            if isinstance(task, dict):
                task = Task.from_dict(task)
            plan = Plan.from_list(plan, task)

        self._logger.info("Executing plan with %d steps", len(plan))
        agent.state = AgentState.EXECUTING

        context: Dict[str, Any] = {}
        results: List[Any] = []

        step_timeout = getattr(agent.config, "step_timeout", 60)
        step_max_retries = getattr(agent.config, "step_max_retries", 2)

        for step in plan.steps:
            step_num = step.step
            action = step.action
            step_task = self._resolve_step_task(step, task)
            step_details = step.details or ""

            self._logger.info(
                "Executing plan step %d: %s", step_num, action
            )
            if step_details:
                self._logger.debug("Step details: %s", step_details)

            step_result, step_error = await self._run_with_retries(
                agent, step, step_num, action, step_task,
                step_details, context, task,
                step_timeout, step_max_retries,
            )

            if step_result is not None:
                results.append(step_result)
                context[f"step_{step_num}"] = step_result
                context[f"step_{step_num}_action"] = action
            elif step_error:
                agent.state = AgentState.ERROR
                return (
                    f"Error: Step {step_num} ({action}) failed: "
                    f"{step_error}"
                )

        final_result = results[-1] if results else None
        agent.state = AgentState.COMPLETED
        return final_result

    # ------------------------------------------------------------------ #
    # Step execution                                                      #
    # ------------------------------------------------------------------ #

    def _get_step_runner(self) -> Any:
        """Return the mode used to execute 'execute' plan steps."""
        if self._step_runner is not None:
            return self._step_runner
        from nucleusiq.agents.modes.standard_mode import StandardMode
        return StandardMode()

    async def _execute_step(
        self,
        agent: "Agent",
        step: PlanStep,
        step_num: int,
        action: str,
        step_task: Dict[str, Any],
        step_details: str,
        context: Dict[str, Any],
        task: Union[Task, Dict[str, Any]],
    ) -> Any:
        """Execute a single plan step."""
        if action == "execute":
            return await self._get_step_runner().run(agent, step_task)

        tool_names = [
            t.name for t in (agent.tools or []) if hasattr(t, "name")
        ]
        if action in tool_names:
            return await self._execute_tool_step(
                agent, step, step_num, action,
                step_details, context, task,
            )

        self._logger.warning(
            "Unknown action '%s' in plan step %d, skipping",
            action, step_num,
        )
        return f"Skipped unknown action: {action}"

    async def _execute_tool_step(
        self,
        agent: "Agent",
        step: PlanStep,
        step_num: int,
        action: str,
        step_details: str,
        context: Dict[str, Any],
        task: Union[Task, Dict[str, Any]],
    ) -> Any:
        """Execute a plan step that calls a tool."""
        if hasattr(agent, "_executor") and agent._executor:
            resolved_args = self._resolve_args(step.args, context)

            # If args are empty, try to infer them via LLM
            if (not resolved_args) and agent.llm:
                resolved_args = await self._infer_tool_args(
                    agent, action, step_num, step_details, context, task,
                )

            fn_call = ToolCallRequest(
                name=action,
                arguments=json.dumps(resolved_args),
            )
            return await agent._executor.execute(fn_call)

        # Fallback: use old-style _execute_tool
        tool_args = self._resolve_args(step.args, context)
        return await agent._execute_tool(action, tool_args)

    async def _infer_tool_args(
        self,
        agent: "Agent",
        action: str,
        step_num: int,
        step_details: str,
        context: Dict[str, Any],
        task: Union[Task, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ask the LLM to infer tool arguments for a step."""
        tool_specs = (
            agent.llm.convert_tool_specs(agent.tools)
            if agent.tools else []
        )
        if not tool_specs:
            raise ValueError(
                f"Plan step {step_num} requires tool '{action}' "
                "but no args were provided and argument inference failed."
            )

        required_keys = self._get_required_keys(tool_specs, action)
        task_obj = (
            task.to_dict() if isinstance(task, Task) else task
        ).get("objective", "")

        step_prompt = self._prompt_strategy.build_step_inference_prompt(
            action, required_keys, task_obj, context, step_num, step_details,
        )

        step_kwargs: Dict[str, Any] = {
            "model": getattr(agent.llm, "model_name", "default"),
            "messages": [{"role": "user", "content": step_prompt}],
            "tools": tool_specs,
            "max_tokens": getattr(
                agent.config, "step_inference_max_tokens", 2048
            ),
        }
        step_kwargs.update(getattr(agent, "_current_llm_overrides", {}))

        step_resp = await agent.llm.call(**step_kwargs)
        return self._extract_args_from_response(
            step_resp, action, step_num
        )

    # ------------------------------------------------------------------ #
    # Retry and timeout                                                   #
    # ------------------------------------------------------------------ #

    async def _run_with_retries(
        self,
        agent: "Agent",
        step: PlanStep,
        step_num: int,
        action: str,
        step_task: Dict[str, Any],
        step_details: str,
        context: Dict[str, Any],
        task: Union[Task, Dict[str, Any]],
        timeout: int,
        max_retries: int,
    ) -> tuple:
        """Run a step with timeout and retry logic.

        Returns ``(result, error)`` — one of them is always ``None``.
        """
        step_result = None
        step_error = None

        for attempt in range(max_retries + 1):
            try:
                step_result = await asyncio.wait_for(
                    self._execute_step(
                        agent, step, step_num, action,
                        step_task, step_details, context, task,
                    ),
                    timeout=timeout,
                )
                return step_result, None
            except asyncio.TimeoutError:
                step_error = (
                    f"Step {step_num} ({action}) timed out after "
                    f"{timeout}s"
                )
                if attempt < max_retries:
                    self._logger.warning(
                        "%s (attempt %d/%d)",
                        step_error, attempt + 1, max_retries + 1,
                    )
                    await asyncio.sleep(1)
                else:
                    self._logger.error(
                        "%s - max retries exceeded", step_error
                    )
            except Exception as e:
                step_error = str(e)
                self._logger.error(
                    "Error in step %d: %s", step_num, step_error
                )
                break

        return None, step_error

    # ------------------------------------------------------------------ #
    # Argument resolution                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_arg_value(val: Any, context: Dict[str, Any]) -> Any:
        """Resolve ``$step_N`` references in argument values."""
        if isinstance(val, str):
            s = val.strip()
            for prefix, suffix in [("$", ""), ("${", "}"), ("{{", "}}")]:
                if s.startswith(prefix) and s.endswith(suffix):
                    key = s[
                        len(prefix): len(s) - (len(suffix) if suffix else 0)
                    ]
                    key = key.strip()
                    if key in context:
                        return context[key]
            if s in context:
                return context[s]
        if isinstance(val, dict):
            return {
                k: PlanExecutor._resolve_arg_value(v, context)
                for k, v in val.items()
            }
        if isinstance(val, list):
            return [
                PlanExecutor._resolve_arg_value(v, context) for v in val
            ]
        return val

    @staticmethod
    def _resolve_args(
        args: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve all ``$step_N`` references in an args dict."""
        if not args:
            return {}
        return {
            k: PlanExecutor._resolve_arg_value(v, context)
            for k, v in args.items()
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_step_task(
        step: PlanStep,
        task: Union[Task, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Normalise the task for a given plan step."""
        step_task = step.task if step.task else task
        if isinstance(step_task, Task):
            return step_task.to_dict()
        if isinstance(step_task, dict):
            return step_task
        return task.to_dict() if isinstance(task, Task) else task

    @staticmethod
    def _get_required_keys(
        tool_specs: List[Dict[str, Any]],
        action: str,
    ) -> List[str]:
        """Extract required parameter keys for a tool from its spec."""
        for spec in tool_specs:
            try:
                fn = (
                    spec.get("function", {})
                    if isinstance(spec, dict) else {}
                )
                if fn.get("name") == action:
                    params = (
                        fn.get("parameters", {})
                        if isinstance(fn, dict) else {}
                    )
                    return (
                        params.get("required", [])
                        if isinstance(params, dict) else []
                    )
            except Exception:
                continue
        return []

    def _extract_args_from_response(
        self,
        response: Any,
        action: str,
        step_num: int,
    ) -> Dict[str, Any]:
        """Extract tool arguments from an LLM inference response."""
        msg = (
            response.choices[0].message
            if response and response.choices else {}
        )
        tool_calls = (
            msg.get("tool_calls")
            if isinstance(msg, dict)
            else getattr(msg, "tool_calls", None)
        )
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                fn = (
                    tc.get("function")
                    if isinstance(tc, dict)
                    else getattr(tc, "function", None)
                ) or {}
                fn_name = (
                    fn.get("name")
                    if isinstance(fn, dict)
                    else getattr(fn, "name", None)
                )
                if fn_name == action:
                    args_str = (
                        fn.get("arguments")
                        if isinstance(fn, dict)
                        else getattr(fn, "arguments", "{}")
                    )
                    try:
                        return (
                            json.loads(args_str)
                            if isinstance(args_str, str)
                            else (args_str or {})
                        )
                    except Exception:
                        return {}

        raise ValueError(
            f"Plan step {step_num} requires tool '{action}' but no "
            "args were provided and argument inference failed."
        )
