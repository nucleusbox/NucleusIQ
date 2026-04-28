"""Back-compat shim — v1 ``ConversationCompactor`` lives in ``compactor``.

The compaction logic now lives in
:mod:`nucleusiq.agents.context.compactor`.  This 1-line shim keeps
older imports green; new code should import from ``compactor``.
"""

from nucleusiq.agents.context.compactor import ConversationCompactor

__all__ = ["ConversationCompactor"]
