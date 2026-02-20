"""
MessageBuilder — constructs LLM message lists from agent context.

Extracted from ``Agent._build_messages()`` and ``Agent._format_plan()``.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from nucleusiq.agents.task import Task
from nucleusiq.agents.plan import Plan
from nucleusiq.agents.chat_models import ChatMessage


class MessageBuilder:
    """Stateless utility for building LLM message lists."""

    @staticmethod
    def build(
        task: Union[Task, Dict[str, Any]],
        plan: Optional[Union[Plan, List[Dict[str, Any]]]] = None,
        *,
        prompt: Optional[Any] = None,
        role: Optional[str] = None,
        objective: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> List[ChatMessage]:
        """
        Build messages for an LLM call.

        Prompt Precedence:
        - If ``prompt`` is provided, it takes precedence over ``role``/``objective``
          for LLM message construction (prompt.system and prompt.user are used).
        - If ``prompt`` is None, ``role`` and ``objective`` are used to construct
          the system message: "You are a {role}. Your objective is to {objective}."

        Message structure:
        1. System message: From prompt.system (if prompt exists) or role/objective
        2. User template: From prompt.user (if prompt exists and has user field)
        3. Plan context: Execution plan if plan exists and has multiple steps
        4. User request: Actual user request (from task["objective"])

        Args:
            task: Task instance or dictionary
            plan: Optional plan (Plan object or list of dicts)
            prompt: Optional prompt object with .system / .user attributes
            role: Agent role string
            objective: Agent objective string
            logger: Optional logger instance

        Returns:
            List of message dictionaries
        """
        messages: List[ChatMessage] = []
        _log = logger or logging.getLogger(__name__)

        task_dict = task.to_dict() if isinstance(task, Task) else task

        if prompt:
            has_system = hasattr(prompt, "system") and prompt.system
            has_user = hasattr(prompt, "user") and prompt.user

            if has_system:
                messages.append(ChatMessage(role="system", content=prompt.system))
                if role or objective:
                    _log.info(
                        "Using prompt.system for execution (overriding role='%s', "
                        "objective='%s'). role/objective will still be used for "
                        "planning context.",
                        role,
                        objective,
                    )
            if has_user:
                messages.append(ChatMessage(role="user", content=prompt.user))
        else:
            if role:
                system_msg = f"You are a {role}."
                if objective:
                    system_msg += f" Your objective is to {objective}."
                messages.append(ChatMessage(role="system", content=system_msg))
                _log.debug("Using role/objective for system message: %s", system_msg)

        if plan:
            if isinstance(plan, Plan):
                plan_steps = plan.steps
            else:
                plan_steps = plan

            if len(plan_steps) > 1:
                plan_text = MessageBuilder.format_plan(plan)
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"Execution Plan:\n{plan_text}\n\n"
                            "Now execute the task following this plan."
                        ),
                    )
                )

        messages.append(ChatMessage(role="user", content=task_dict.get("objective", "")))

        return messages

    @staticmethod
    def format_plan(plan: Union[Plan, List[Dict[str, Any]]]) -> str:
        """
        Format plan steps into a readable string.

        Args:
            plan: Plan instance or list of plan step dictionaries

        Returns:
            Formatted plan string
        """
        plan_lines: List[str] = []
        # Convert Plan to list of dicts if needed
        if isinstance(plan, Plan):
            steps = plan.to_list()
        else:
            steps = plan

        for step in steps:
            step_num = step.get("step", 0)
            action = step.get("action", "")
            details = step.get("details", "")
            plan_lines.append(f"Step {step_num}: {action}")
            if details:
                plan_lines.append(f"  {details.strip()}")
        return "\n".join(plan_lines)

    @staticmethod
    def content_to_text(content: Any) -> Optional[str]:
        """
        Coerce OpenAI-style message content into plain text.

        Chat completions generally return ``content: str | None``, but some
        SDKs / modes may represent content as a list of parts.
        """
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            joined = "\n".join(p for p in parts if p.strip())
            return joined if joined.strip() else None
        # Last resort
        s = str(content)
        return s if s.strip() else None
