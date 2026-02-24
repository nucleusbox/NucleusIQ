"""
HumanApprovalPlugin — pause for human approval before executing tools.

Two ways to control approval:

**Quick — pass a callback function:**

    HumanApprovalPlugin(
        approval_callback=lambda name, args: name != "delete_file",
    )

**Structured — pass an ApprovalHandler:**

    class MyHandler(ApprovalHandler):
        async def decide(self, tool_name, tool_args):
            return tool_name in self.safe_tools

        async def on_deny(self, tool_name, tool_args):
            await notify_security(tool_name)

    HumanApprovalPlugin(approval_handler=MyHandler())

Built-in handlers:

- ``ConsoleApprovalHandler``  — interactive ``input()`` prompt
- ``PolicyApprovalHandler``   — rule-based with safe/dangerous lists and audit log
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Sequence

from nucleusiq.plugins.base import BasePlugin, ToolHandler, ToolRequest

logger = logging.getLogger(__name__)

ApprovalCallback = Callable[[str, Dict[str, Any]], bool | Awaitable[bool]]


# ------------------------------------------------------------------ #
# ApprovalHandler base class                                           #
# ------------------------------------------------------------------ #


class ApprovalHandler(ABC):
    """Base class for structured approval handlers.

    Subclass and implement ``decide()`` to control approval logic.
    Override ``on_approve()`` / ``on_deny()`` for side effects like
    logging, audit trails, Slack notifications, or escalation.

    Example::

        class SlackApprovalHandler(ApprovalHandler):
            def __init__(self, channel: str):
                self.channel = channel

            async def decide(self, tool_name, tool_args):
                return await ask_slack(self.channel, f"Approve {tool_name}?")

            async def on_deny(self, tool_name, tool_args):
                await post_slack("#security", f"DENIED: {tool_name}")
    """

    @abstractmethod
    async def decide(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        """Return ``True`` to approve, ``False`` to deny."""
        ...

    async def on_approve(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        """Called after a tool call is approved. Override for side effects."""
        pass

    async def on_deny(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        """Called after a tool call is denied. Override for side effects."""
        pass


# ------------------------------------------------------------------ #
# Built-in handlers                                                     #
# ------------------------------------------------------------------ #


class ConsoleApprovalHandler(ApprovalHandler):
    """Interactive approval via terminal ``input()`` prompt.

    This is the default handler when no callback or handler is provided.
    """

    async def decide(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        print(f"\n{'=' * 50}")
        print("  Tool approval required")
        print(f"  Tool:  {tool_name}")
        print(f"  Args:  {tool_args}")
        print(f"{'=' * 50}")
        response = input("  Approve? (y/n): ").strip().lower()
        return response in ("y", "yes")


class PolicyApprovalHandler(ApprovalHandler):
    """Rule-based approval with safe/dangerous tool lists and audit log.

    Maintains an audit trail of all decisions with timestamps.

    Args:
        safe_tools: Tools that are always approved.
        dangerous_tools: Tools that are always denied.
            If a tool is in neither list, ``default_allow`` decides.
        default_allow: Decision for tools not in any list. Defaults to ``False``.

    Example::

        handler = PolicyApprovalHandler(
            safe_tools=["add", "search"],
            dangerous_tools=["delete_file", "drop_database"],
            default_allow=False,
        )
        plugin = HumanApprovalPlugin(approval_handler=handler)

        # After execution, inspect the audit log:
        for entry in handler.audit_log:
            print(entry)
    """

    def __init__(
        self,
        safe_tools: Sequence[str] | None = None,
        dangerous_tools: Sequence[str] | None = None,
        default_allow: bool = False,
    ) -> None:
        self._safe = set(safe_tools or [])
        self._dangerous = set(dangerous_tools or [])
        self._default_allow = default_allow
        self._audit_log: List[Dict[str, Any]] = []

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Read-only access to the audit trail."""
        return list(self._audit_log)

    def _record(
        self, tool_name: str, tool_args: Dict[str, Any], approved: bool, reason: str
    ) -> None:
        self._audit_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool_name,
                "args": tool_args,
                "approved": approved,
                "reason": reason,
            }
        )

    async def decide(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        if tool_name in self._safe:
            self._record(tool_name, tool_args, True, "safe_list")
            return True
        if tool_name in self._dangerous:
            self._record(tool_name, tool_args, False, "dangerous_list")
            return False
        self._record(tool_name, tool_args, self._default_allow, "default_policy")
        return self._default_allow

    async def on_approve(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        logger.info("PolicyApprovalHandler approved '%s'", tool_name)

    async def on_deny(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        logger.warning("PolicyApprovalHandler denied '%s'", tool_name)


# ------------------------------------------------------------------ #
# HumanApprovalPlugin                                                  #
# ------------------------------------------------------------------ #


class HumanApprovalPlugin(BasePlugin):
    """Requires human approval before executing specified tools.

    Accepts **either** a structured ``ApprovalHandler`` **or** a simple
    callback function. Cannot pass both.

    Args:
        approval_handler: An ``ApprovalHandler`` subclass instance for
            structured approval with lifecycle hooks.
        approval_callback: A simple function ``(tool_name, tool_args) -> bool``.
            Can be sync or async. For quick one-off rules.
        require_approval: Tool names that need approval. ``None`` = all tools.
        auto_approve: Tool names to always skip approval for.
        deny_message: Message returned when a tool call is denied.

    Examples::

        # Quick — callback function
        HumanApprovalPlugin(
            approval_callback=lambda n, a: n != "delete_file",
            require_approval=["delete_file", "send_email"],
        )

        # Structured — handler class
        handler = PolicyApprovalHandler(
            safe_tools=["add", "search"],
            dangerous_tools=["delete_file"],
        )
        HumanApprovalPlugin(
            approval_handler=handler,
            auto_approve=["add"],
        )

        # Default — interactive console prompt
        HumanApprovalPlugin()
    """

    def __init__(
        self,
        approval_handler: ApprovalHandler | None = None,
        approval_callback: ApprovalCallback | None = None,
        require_approval: Sequence[str] | None = None,
        auto_approve: Sequence[str] | None = None,
        deny_message: str = "Tool execution denied by human reviewer.",
    ) -> None:
        if approval_handler and approval_callback:
            raise ValueError(
                "Pass either 'approval_handler' (class) or "
                "'approval_callback' (function), not both."
            )

        self._handler: ApprovalHandler | None = approval_handler
        self._callback: ApprovalCallback | None = approval_callback

        if not approval_handler and not approval_callback:
            self._handler = ConsoleApprovalHandler()

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

    async def _get_decision(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        """Route to handler or callback for the approval decision."""
        if self._handler:
            return await self._handler.decide(tool_name, tool_args)

        result = self._callback(tool_name, tool_args)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            return await result
        return result

    async def wrap_tool_call(self, request: ToolRequest, handler: ToolHandler) -> Any:
        if not self._needs_approval(request.tool_name):
            return await handler(request)

        approved = await self._get_decision(request.tool_name, request.tool_args)

        if approved:
            logger.info("Tool '%s' approved", request.tool_name)
            if self._handler:
                await self._handler.on_approve(request.tool_name, request.tool_args)
            return await handler(request)
        else:
            logger.warning("Tool '%s' denied", request.tool_name)
            if self._handler:
                await self._handler.on_deny(request.tool_name, request.tool_args)
            return f"{self._deny_message} (tool: {request.tool_name})"
