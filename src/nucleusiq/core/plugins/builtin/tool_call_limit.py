"""
ToolCallLimitPlugin — halts execution if tool invocations exceed a threshold.

Usage::

    agent = Agent(
        ...,
        plugins=[ToolCallLimitPlugin(max_calls=20)],
    )
"""

from __future__ import annotations

from typing import Any

from nucleusiq.plugins.base import BasePlugin, ToolRequest, ToolHandler
from nucleusiq.plugins.errors import PluginHalt


class ToolCallLimitPlugin(BasePlugin):
    """Prevents runaway tool usage by capping the number of tool calls per execution."""

    def __init__(self, max_calls: int = 20) -> None:
        self._max_calls = max_calls

    @property
    def name(self) -> str:
        return "tool_call_limit"

    async def wrap_tool_call(self, request: ToolRequest, handler: ToolHandler) -> Any:
        if request.call_count > self._max_calls:
            raise PluginHalt(
                f"Tool call limit exceeded: {request.call_count} > {self._max_calls}"
            )
        return await handler(request)
