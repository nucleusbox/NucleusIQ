"""
MessageBuilder — constructs LLM message lists from agent context.

Extracted from ``Agent._build_messages()`` and ``Agent._format_plan()``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from nucleusiq.agents.attachments import AttachmentProcessor, ContentPart
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.plan import Plan
from nucleusiq.agents.task import Task


class MessageBuilder:
    """Stateless utility for building LLM message lists."""

    @staticmethod
    def build(
        task: Task | dict[str, Any],
        plan: Plan | list[dict[str, Any]] | None = None,
        *,
        prompt: Any = None,
        logger: logging.Logger | None = None,
        attachment_processor: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> list[ChatMessage]:
        """
        Build messages for an LLM call.

        The ``prompt`` object (a ``BasePrompt`` subclass) defines the
        system message and optional user preamble.  ``role`` and
        ``objective`` on the Agent are labels for logging only and are
        **not** used for message construction.

        Message structure:
        1. System message — from ``prompt.system`` (when set)
        2. User preamble — from ``prompt.user`` (when set)
        3. Plan context — execution plan (when multi-step)
        4. User request — from ``task.objective``
           (multimodal content array when the Task has attachments)

        Args:
            task: Task instance or dictionary.
            plan: Optional plan (Plan object or list of dicts).
            prompt: A BasePrompt instance whose ``.system`` and
                ``.user`` attributes define the LLM messages.
            logger: Optional logger instance.
            attachment_processor: Optional callable that converts a list
                of Attachment objects into content-part dicts.

        Returns:
            List of ChatMessage objects.
        """
        messages: list[ChatMessage] = []
        _log = logger or logging.getLogger(__name__)

        task_dict = task.to_dict() if isinstance(task, Task) else task

        if prompt:
            has_system = hasattr(prompt, "system") and prompt.system
            has_user = hasattr(prompt, "user") and prompt.user

            if has_system:
                messages.append(ChatMessage(role="system", content=prompt.system))
            if has_user:
                messages.append(ChatMessage(role="user", content=prompt.user))

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

        attachments = (
            task.attachments if isinstance(task, Task) and task.attachments else None
        )

        if attachments:
            content_parts = MessageBuilder._build_multimodal_content(
                task_dict.get("objective", ""),
                attachments,
                attachment_processor=attachment_processor,
            )
            messages.append(ChatMessage(role="user", content=content_parts))
        else:
            messages.append(
                ChatMessage(role="user", content=task_dict.get("objective", ""))
            )

        return messages

    # ------------------------------------------------------------------ #
    # Multimodal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_multimodal_content(
        objective: str,
        attachments: list,
        *,
        attachment_processor: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build a content array mixing text and attachment parts.

        When *attachment_processor* is provided (e.g.
        ``llm.process_attachments``), it is called with the raw
        ``Attachment`` list and its output is used directly.  Otherwise
        the framework-level ``AttachmentProcessor`` is used as a fallback.

        Returns a list of dicts suitable for the ``content`` field of a
        ChatMessage.
        """
        parts: list[dict[str, Any]] = []

        if objective:
            parts.append({"type": "text", "text": objective})

        if attachment_processor is not None:
            parts.extend(attachment_processor(attachments))
        else:
            processed: list[ContentPart] = AttachmentProcessor.process(attachments)
            for cp in processed:
                if cp.type == "text" and cp.text is not None:
                    parts.append({"type": "text", "text": cp.text})
                elif cp.type == "image_url" and cp.image_url is not None:
                    parts.append({"type": "image_url", "image_url": cp.image_url})
                elif cp.metadata is not None:
                    parts.append({"type": cp.type, **cp.metadata})

        return parts

    @staticmethod
    def format_plan(plan: Plan | list[dict[str, Any]]) -> str:
        """
        Format plan steps into a readable string.

        Args:
            plan: Plan instance or list of plan step dictionaries

        Returns:
            Formatted plan string
        """
        plan_lines: list[str] = []
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
    def content_to_text(content: Any) -> str | None:
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
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            joined = "\n".join(p for p in parts if p.strip())
            return joined if joined.strip() else None
        s = str(content)
        return s if s.strip() else None
