"""
Execution mode strategies for NucleusIQ agents.

Each mode implements the BaseExecutionMode interface and encapsulates
a distinct execution strategy (Direct, Standard, Autonomous).

New modes can be added by subclassing BaseExecutionMode and registering
via ``Agent.register_mode()`` — no changes to Agent required (Open/Closed).
"""

from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
from nucleusiq.agents.modes.base_mode import (
    BaseExecutionMode,
    build_attachment_metadata,
    build_attachment_prefix,
)
from nucleusiq.agents.modes.direct_mode import DirectMode
from nucleusiq.agents.modes.standard_mode import StandardMode

__all__ = [
    "BaseExecutionMode",
    "DirectMode",
    "StandardMode",
    "AutonomousMode",
    "build_attachment_metadata",
    "build_attachment_prefix",
]
