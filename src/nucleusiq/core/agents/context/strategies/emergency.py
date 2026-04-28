"""Back-compat shim — v1 ``EmergencyCompactor`` lives in ``compactor``."""

from nucleusiq.agents.context.compactor import EmergencyCompactor

__all__ = ["EmergencyCompactor"]
