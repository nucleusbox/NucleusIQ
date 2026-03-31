"""Plugin error types for the NucleusIQ plugin system.

Hierarchy::

    NucleusIQError
    └── PluginError
        ├── PluginExecutionError  — plugin hook raised unexpectedly
        └── PluginHalt            — intentional early-exit (not a "failure")
"""

from __future__ import annotations

from typing import Any

from nucleusiq.errors import NucleusIQError

__all__ = [
    "PluginError",
    "PluginExecutionError",
    "PluginHalt",
]


class PluginError(NucleusIQError):
    """Base exception for plugin system errors.

    Unlike ``PluginHalt``, this indicates a genuine failure rather
    than an intentional early-exit.

    Attributes:
        plugin_name: Name of the plugin that failed.
        hook: Hook point where the error occurred (e.g. "before_agent").
    """

    def __init__(
        self,
        message: str = "",
        *,
        plugin_name: str | None = None,
        hook: str | None = None,
    ) -> None:
        self.plugin_name = plugin_name
        self.hook = hook
        super().__init__(message)


class PluginExecutionError(PluginError):
    """Plugin hook raised an unexpected exception during execution."""


class PluginHalt(PluginError):
    """Raised by a plugin to abort execution and return a result early.

    This is an intentional control-flow signal, not a failure.

    Example::

        @before_model
        async def limit_calls(ctx):
            if ctx.call_count > 10:
                raise PluginHalt("Model call limit exceeded")
    """

    def __init__(self, result: Any = None) -> None:
        self.result = result
        super().__init__(
            str(result) if result is not None else "Plugin halted execution"
        )
