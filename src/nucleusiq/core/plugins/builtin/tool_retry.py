"""
ToolRetryPlugin — retries failed tool calls with exponential backoff.

Usage::

    agent = Agent(
        ...,
        plugins=[ToolRetryPlugin(max_retries=3, base_delay=1.0)],
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from nucleusiq.plugins.base import BasePlugin, ToolRequest, ToolHandler

logger = logging.getLogger(__name__)


class ToolRetryPlugin(BasePlugin):
    """Wraps tool calls with retry logic and exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    @property
    def name(self) -> str:
        return "tool_retry"

    async def wrap_tool_call(self, request: ToolRequest, handler: ToolHandler) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await handler(request)
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = min(
                        self._base_delay * (2 ** attempt),
                        self._max_delay,
                    )
                    logger.warning(
                        "Tool '%s' failed (attempt %d/%d), retrying in %.1fs: %s",
                        request.tool_name,
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]
