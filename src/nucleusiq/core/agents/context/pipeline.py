"""Back-compat shim — v1 ``CompactionPipeline`` lives in ``compactor``.

Context Mgmt v2 — Step 3 consolidates pipeline + 4 strategy classes
into a single :class:`Compactor`.  The :class:`CompactionPipeline`
class is preserved here as a back-compat re-export so existing
``from nucleusiq.agents.context.pipeline import CompactionPipeline``
imports continue to work.

For new code, prefer :class:`nucleusiq.agents.context.compactor.Compactor`
directly — it's the one source of truth and the API the engine talks to.
"""

from nucleusiq.agents.context.compactor import CompactionPipeline

__all__ = ["CompactionPipeline"]
