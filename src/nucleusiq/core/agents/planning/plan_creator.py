"""
PlanCreator — generates execution plans via the LLM.

Extracted from ``Planner.create_plan()`` and
``Planner.construct_planning_prompt()``.

Responsibilities:
- Build planning prompts (delegates to ``PlanPromptStrategy``)
- Call the LLM with function-calling or content-based fallback
- Parse the LLM response into a ``Plan`` instance
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from nucleusiq.agents.task import Task
from nucleusiq.agents.plan import Plan, PlanResponse
from nucleusiq.agents.planning.plan_parser import PlanParser
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.agents.planning.prompt_strategy import (
    PlanPromptStrategy,
    DefaultPlanPromptStrategy,
)
from nucleusiq.agents.chat_models import LLMCallKwargs

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent


class PlanCreator:
    """Creates execution plans using an LLM.

    This class is intentionally decoupled from ``Agent``: it receives
    the dependencies it needs (LLM, tools, config, prompt) via its
    public methods so that it can be unit-tested in isolation.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        prompt_strategy: Optional[PlanPromptStrategy] = None,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._parser = PlanParser(logger=self._logger)
        self._prompt_strategy = prompt_strategy or DefaultPlanPromptStrategy()

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def create_plan(
        self,
        agent: "Agent",
        task: Union[Task, Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Plan:
        """Create an execution plan using the LLM.

        Tries function-calling first, then falls back to content-based
        JSON parsing.
        """
        if not agent.llm:
            raise ValueError("LLM is required for LLM-based planning")

        task_dict = task.to_dict() if isinstance(task, Task) else task
        task_obj = task_dict.get("objective", str(task_dict))
        tool_names = [
            getattr(t, "name", "unknown") for t in (agent.tools or [])
        ]
        tool_param_lines = self._get_tool_param_lines(agent)

        # Full structured-JSON prompt (content fallback)
        plan_prompt = self._prompt_strategy.build_planning_prompt(
            task_objective=task_obj,
            tool_names=tool_names,
            role=agent.role,
            objective=agent.objective,
        )

        # Short tool-call prompt
        tool_call_prompt = self._prompt_strategy.build_tool_call_prompt(
            task_obj, tool_param_lines
        )

        from nucleusiq.agents.planning.schema import get_plan_function_spec

        plan_function_spec = get_plan_function_spec()
        empty_retries_remaining = 1

        while True:
            messages = [{"role": "user", "content": plan_prompt}]
            llm_response = None

            # --- Try function-calling first -------------------------
            try:
                plan_kwargs = self._build_llm_kwargs(
                    agent, tool_call_prompt, [plan_function_spec]
                )
                llm_response = await agent.llm.call(**plan_kwargs)

                result = self._extract_plan_from_tool_calls(
                    llm_response, task
                )
                if result is not None:
                    return result
            except Exception as e:
                self._logger.debug(
                    "Tool calling not available or failed: %s. "
                    "Using content-based parsing.",
                    e,
                )

            # --- Fallback: parse from content -----------------------
            if not llm_response:
                fb_kwargs = self._build_llm_kwargs(agent, plan_prompt)
                llm_response = await agent.llm.call(**fb_kwargs)

            response_content = self._extract_content(llm_response)

            if not response_content:
                if empty_retries_remaining > 0:
                    empty_retries_remaining -= 1
                    plan_prompt += (
                        "\n\nIMPORTANT: Do not return an empty message. "
                        "Return either a tool call to create_plan, or "
                        "JSON content."
                    )
                    continue
                raise ValueError("LLM returned no content for planning")

            plan_response = self._parser.parse(response_content)
            return plan_response.to_plan(task)

    # ------------------------------------------------------------------ #
    # Tool helpers                                                         #
    # ------------------------------------------------------------------ #

    def _get_tool_param_lines(self, agent: "Agent") -> List[str]:
        """Build human-readable tool parameter lines."""
        lines: List[str] = []
        for t in agent.tools or []:
            try:
                spec = t.get_spec() if hasattr(t, "get_spec") else None
                params = (spec or {}).get("parameters", {})
                props = (
                    (params or {}).get("properties", {})
                    if isinstance(params, dict)
                    else {}
                )
                required = (
                    (params or {}).get("required", [])
                    if isinstance(params, dict)
                    else []
                )
                if isinstance(props, dict) and props:
                    lines.append(
                        f"- {t.name}({', '.join(props.keys())}) "
                        f"required={required}"
                    )
                else:
                    lines.append(f"- {t.name}(...)")
            except Exception:
                lines.append(
                    f"- {getattr(t, 'name', 'unknown')}(...)"
                )
        return lines

    # ------------------------------------------------------------------ #
    # LLM interaction helpers                                             #
    # ------------------------------------------------------------------ #

    def _build_llm_kwargs(
        self,
        agent: "Agent",
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMCallKwargs:
        """Build kwargs for ``agent.llm.call()``."""
        kwargs: Dict[str, Any] = {
            "model": getattr(agent.llm, "model_name", "default"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": getattr(
                agent.config, "planning_max_tokens", 4096
            ),
        }
        if tools:
            kwargs["tools"] = tools
        kwargs.update(getattr(agent, "_current_llm_overrides", {}))
        return kwargs

    def _extract_plan_from_tool_calls(
        self,
        response: Any,
        task: Union[Task, Dict[str, Any]],
    ) -> Optional[Plan]:
        """Try to extract a Plan from tool_calls in the response."""
        if (
            not response
            or not hasattr(response, "choices")
            or not response.choices
        ):
            return None

        msg = response.choices[0].message
        tool_calls = (
            msg.get("tool_calls")
            if isinstance(msg, dict)
            else getattr(msg, "tool_calls", None)
        )
        if not tool_calls or not isinstance(tool_calls, list):
            return None

        for tc in tool_calls:
            fn_name, fn_args_str = self._parse_tool_call_info(tc)
            if fn_name == "create_plan":
                try:
                    plan_data = json.loads(fn_args_str)
                    plan_response = PlanResponse.from_dict(plan_data)
                    self._logger.debug(
                        "Successfully received structured plan "
                        "via tool calling"
                    )
                    return plan_response.to_plan(task)
                except (json.JSONDecodeError, ValueError) as e:
                    self._logger.warning(
                        "Failed to parse tool call arguments: %s. "
                        "Falling back to content parsing.",
                        e,
                    )
                    return None
        return None

    @staticmethod
    def _parse_tool_call_info(tc: Any) -> tuple:
        """Return ``(fn_name, fn_args_str)`` from a tool call."""
        if isinstance(tc, dict):
            fn_info = tc.get("function", {})
            fn_name = (
                fn_info.get("name")
                if isinstance(fn_info, dict) else None
            )
            fn_args_str = (
                fn_info.get("arguments", "{}")
                if isinstance(fn_info, dict) else "{}"
            )
        else:
            fn_info = getattr(tc, "function", None)
            fn_name = getattr(fn_info, "name", None) if fn_info else None
            fn_args_str = (
                getattr(fn_info, "arguments", "{}") if fn_info else "{}"
            )
        return fn_name, fn_args_str

    def _extract_content(self, response: Any) -> Optional[str]:
        """Extract text content from an LLM response, checking for refusals."""
        if (
            not response
            or not hasattr(response, "choices")
            or not response.choices
        ):
            raise ValueError("LLM returned empty response for planning")

        msg = response.choices[0].message
        if isinstance(msg, dict):
            refusal = msg.get("refusal")
            content = MessageBuilder.content_to_text(msg.get("content"))
        else:
            refusal = getattr(msg, "refusal", None)
            content = MessageBuilder.content_to_text(
                getattr(msg, "content", None)
            )

        if refusal:
            raise ValueError(
                f"LLM refused to generate plan: {refusal}"
            )
        return content
