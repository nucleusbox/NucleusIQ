"""
Plugin error types for the NucleusIQ plugin system.
"""

from typing import Any


class PluginHalt(Exception):
    """Raised by a plugin to abort execution and return a result early.

    Example::

        @before_model
        async def limit_calls(ctx):
            if ctx.call_count > 10:
                raise PluginHalt("Model call limit exceeded")
    """

    def __init__(self, result: Any = None) -> None:
        self.result = result
        super().__init__(str(result) if result is not None else "Plugin halted execution")


class PluginError(Exception):
    """Raised when a plugin encounters an unrecoverable error.

    Unlike ``PluginHalt``, this indicates a genuine failure rather
    than an intentional early-exit.
    """
