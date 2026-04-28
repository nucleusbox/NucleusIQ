"""Compaction strategies — back-compat shim package.

Context Mgmt v2 — Step 3 consolidated the four v1 strategies and the
``CompactionPipeline`` into a single
:class:`nucleusiq.agents.context.compactor.Compactor`.  This package
remains for backwards compatibility — the individual modules
(``conversation.py``, ``emergency.py``, ``tool_result.py``,
``observation_masker.py``) are now thin re-exports of the wrappers
defined in ``compactor.py``.

Public re-exports here are intentionally limited to the abstract
types in ``base.py`` so that importing this package never triggers
the heavier ``compactor.py`` graph (and avoids a circular import
through ``strategies/__init__.py`` → ``compactor.py`` →
``strategies.base``).

Tests and external code that want the legacy strategy classes should
either:

* Import via the individual shim modules (e.g.
  ``from nucleusiq.agents.context.strategies.observation_masker import
  ObservationMasker``), or
* Import directly from ``nucleusiq.agents.context.compactor``
  (preferred for new code).
"""

from nucleusiq.agents.context.strategies.base import (
    CompactionResult,
    CompactionStrategy,
)

__all__ = [
    "CompactionResult",
    "CompactionStrategy",
]
