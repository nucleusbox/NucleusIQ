"""
HumanApprovalPlugin — pause for human approval before executing tools.

Supports both sync and async approval callbacks, per-tool configuration,
and auto-approve/deny lists.

Usage::

    # Interactive approval via console
    agent = Agent(
        ...,
        plugins=[HumanApprovalPlugin()],
    )

    # Custom callback
    async def my_approval(tool_name, tool_args):
        return tool_name not in DANGEROUS_TOOLS

    agent = Agent(
        ...,
        plugins=[
            HumanApprovalPlugin(
                approval_callback=my_approval,
                require_approval=["delete_file", "run_command"],
            )
        ],
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence, Union

from nucleusiq.plugins.base import BasePlugin, ToolRequest, ToolHandler

logger = logging.getLogger(__name__)

ApprovalCallback = Callable[[str, Dict[str, Any]], Union[bool, Awaitable[bool]]]


def _default_console_approval(tool_name: str, tool_args: Dict[str, Any]) -> bool:
    """Default approval via console input."""
    print(f"\n{'='*50}")
    print(f"  Tool approval required")
    print(f"  Tool:  {tool_name}")
    print(f"  Args:  {tool_args}")
    print(f"{'='*50}")
    response = input("  Approve? (y/n): ").strip().lower()
    return response in ("y", "yes")


class HumanApprovalPlugin(BasePlugin):
    """Requires human approval before executing specified tools.

    Args:
        approval_callback: Function that receives ``(tool_name, tool_args)``
            and returns ``True`` to approve or ``False`` to deny. Can be
            sync or async. Defaults to console-based input.
        require_approval: List of tool names that need approval. If ``None``,
            all tools require approval.
        auto_approve: List of tool names to always approve (skip callback).
        deny_message: Message returned when a tool call is denied.
    """

    def __init__(
        self,
        approval_callback: Optional[ApprovalCallback] = None,
        require_approval: Optional[Sequence[str]] = None,
        auto_approve: Optional[Sequence[str]] = None,
        deny_message: str = "Tool execution denied by human reviewer.",
    ) -> None:
        self._callback = approval_callback or _default_console_approval
        self._require_approval = set(require_approval) if require_approval else None
        self._auto_approve = set(auto_approve or [])
        self._deny_message = deny_message

    @property
    def name(self) -> str:
        return "human_approval"

    def _needs_approval(self, tool_name: str) -> bool:
        if tool_name in self._auto_approve:
            return False
        if self._require_approval is not None:
            return tool_name in self._require_approval
        return True

    async def wrap_tool_call(
        self, request: ToolRequest, handler: ToolHandler
    ) -> Any:
        if not self._needs_approval(request.tool_name):
            return await handler(request)

        result = self._callback(request.tool_name, request.tool_args)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            approved = await result
        else:
            approved = result

        if approved:
            logger.info("Tool '%s' approved by human", request.tool_name)
            return await handler(request)
        else:
            logger.warning("Tool '%s' denied by human", request.tool_name)
            return f"{self._deny_message} (tool: {request.tool_name})"
