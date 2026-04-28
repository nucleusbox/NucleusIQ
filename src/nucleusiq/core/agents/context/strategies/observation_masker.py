"""Back-compat shim for the v1 ``observation_masker`` module.

Context Mgmt v2 — Step 3 consolidated all compaction logic into
:mod:`nucleusiq.agents.context.compactor`.  This file is kept as a
1-line shim so older imports continue to work::

    from nucleusiq.agents.context.strategies.observation_masker import (
        MASK_PREFIX,
        ObservationMasker,
        build_marker,
    )

is equivalent to::

    from nucleusiq.agents.context.compactor import (
        MASK_PREFIX,
        ObservationMasker,
        build_marker,
    )

The shim will be removed in v0.9 alongside the other deprecated
strategy modules.  New code should import from ``compactor`` directly.
"""

from nucleusiq.agents.context.compactor import (
    MASK_PREFIX,
    ObservationMasker,
    build_marker,
)

__all__ = ["MASK_PREFIX", "ObservationMasker", "build_marker"]
