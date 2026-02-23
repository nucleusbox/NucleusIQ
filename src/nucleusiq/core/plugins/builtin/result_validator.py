"""
ResultValidatorPlugin — base class for result validation in Autonomous Mode.

Subclass this to create domain-specific validators that the
ValidationPipeline (Layer 2) runs after each execution attempt.

Usage::

    class MyValidator(ResultValidatorPlugin):
        async def validate_result(self, result, context):
            if looks_wrong(result):
                return False, "Result looks wrong because..."
            return True, ""

    agent = Agent(
        ...,
        config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS),
        plugins=[MyValidator()],
    )
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

from nucleusiq.plugins.base import BasePlugin


class ResultValidatorPlugin(BasePlugin):
    """Base class for autonomous-mode result validators.

    Subclass and override ``validate_result`` to provide
    domain-specific validation. The ValidationPipeline calls
    this method after each execution attempt; returning
    ``(False, reason)`` triggers a retry with error context.
    """

    @abstractmethod
    async def validate_result(
        self,
        result: Any,
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Validate an execution result.

        Args:
            result: The raw result string from the execution.
            context: Dict with keys:
                - ``task_objective``: The original task description.
                - ``agent_name``: Name of the executing agent.
                - ``attempt``: Current attempt number (0-based).

        Returns:
            Tuple of (valid, reason).
            - ``(True, "")`` if result looks acceptable.
            - ``(False, "explanation")`` if result should be retried.
        """
        ...
