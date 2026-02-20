"""
ToolGuardPlugin — whitelist or blacklist tools by name.

Provides fine-grained access control over which tools an agent can execute,
with customizable deny messages.

Usage::

    # Block specific dangerous tools
    agent = Agent(
        ...,
        plugins=[
            ToolGuardPlugin(blocked=["delete_file", "drop_table"]),
        ],
    )

    # Only allow specific tools
    agent = Agent(
        ...,
        plugins=[
            ToolGuardPlugin(allowed=["search", "calculator"]),
        ],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Sequence, Union

from nucleusiq.plugins.base import BasePlugin, ToolRequest, ToolHandler

logger = logging.getLogger(__name__)

DenyHandler = Callable[[str, Dict[str, Any]], str]


def _default_deny_message(tool_name: str, tool_args: Dict[str, Any]) -> str:
    return f"Tool '{tool_name}' is blocked by security policy."


class ToolGuardPlugin(BasePlugin):
    """Controls which tools can be executed.

    Operates in one of two modes:
    - **Blocklist**: Set ``blocked`` to deny specific tools (all others allowed).
    - **Allowlist**: Set ``allowed`` to permit only specific tools (all others denied).

    Cannot use both ``blocked`` and ``allowed`` simultaneously.

    Args:
        blocked: Tool names to deny. Mutually exclusive with ``allowed``.
        allowed: Tool names to permit. Mutually exclusive with ``blocked``.
        on_deny: Custom function ``(tool_name, tool_args) -> str`` for the denial
            message, or a static string. Defaults to a standard message.
    """

    def __init__(
        self,
        blocked: Optional[Sequence[str]] = None,
        allowed: Optional[Sequence[str]] = None,
        on_deny: Union[str, DenyHandler, None] = None,
    ) -> None:
        if blocked and allowed:
            raise ValueError("Cannot specify both 'blocked' and 'allowed'")
        if not blocked and not allowed:
            raise ValueError("Must specify either 'blocked' or 'allowed'")

        self._blocked = set(blocked) if blocked else None
        self._allowed = set(allowed) if allowed else None

        if on_deny is None:
            self._deny_fn = _default_deny_message
        elif isinstance(on_deny, str):
            self._deny_fn = lambda name, args: on_deny
        else:
            self._deny_fn = on_deny

    @property
    def name(self) -> str:
        return "tool_guard"

    def _is_allowed(self, tool_name: str) -> bool:
        if self._blocked is not None:
            return tool_name not in self._blocked
        if self._allowed is not None:
            return tool_name in self._allowed
        return True

    async def wrap_tool_call(
        self, request: ToolRequest, handler: ToolHandler
    ) -> Any:
        if self._is_allowed(request.tool_name):
            logger.debug("ToolGuard: ALLOWED '%s'", request.tool_name)
            return await handler(request)

        logger.warning("ToolGuard: BLOCKED '%s'", request.tool_name)
        return self._deny_fn(request.tool_name, request.tool_args)
