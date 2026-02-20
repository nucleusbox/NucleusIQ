"""
ModelCallLimitPlugin — halts execution if LLM calls exceed a threshold.

Usage::

    agent = Agent(
        ...,
        plugins=[ModelCallLimitPlugin(max_calls=10)],
    )
"""

from __future__ import annotations

from typing import Optional

from nucleusiq.plugins.base import BasePlugin, ModelRequest
from nucleusiq.plugins.errors import PluginHalt


class ModelCallLimitPlugin(BasePlugin):
    """Prevents runaway LLM usage by capping the number of calls per execution."""

    def __init__(self, max_calls: int = 10) -> None:
        self._max_calls = max_calls

    @property
    def name(self) -> str:
        return "model_call_limit"

    async def before_model(self, request: ModelRequest) -> Optional[ModelRequest]:
        if request.call_count > self._max_calls:
            raise PluginHalt(
                f"Model call limit exceeded: {request.call_count} > {self._max_calls}"
            )
        return None
