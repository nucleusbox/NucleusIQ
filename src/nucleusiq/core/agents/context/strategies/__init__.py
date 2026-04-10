"""Compaction strategies — pluggable algorithms for context reduction.

Four tiers modeled after JVM garbage collection generations:
    - Tier 0   → ``ObservationMasker`` (always, post-response)
    - Minor GC → ``ToolResultCompactor`` (cheap, frequent)
    - Major GC → ``ConversationCompactor`` (moderate cost)
    - Full GC  → ``EmergencyCompactor`` (expensive, last resort)
"""

from nucleusiq.agents.context.strategies.base import (
    CompactionResult,
    CompactionStrategy,
)
from nucleusiq.agents.context.strategies.conversation import ConversationCompactor
from nucleusiq.agents.context.strategies.emergency import EmergencyCompactor
from nucleusiq.agents.context.strategies.observation_masker import ObservationMasker
from nucleusiq.agents.context.strategies.tool_result import ToolResultCompactor

__all__ = [
    "CompactionResult",
    "CompactionStrategy",
    "ConversationCompactor",
    "EmergencyCompactor",
    "ObservationMasker",
    "ToolResultCompactor",
]
