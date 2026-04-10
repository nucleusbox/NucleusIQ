"""Context window management — region-aware progressive compaction.

Public API for context management in NucleusIQ agents.

Quick start::

    from nucleusiq.agents.context import ContextConfig, ContextEngine

    # Zero-config: uses optimal_budget=50K, observation masking ON
    config = ContextConfig()

    # Explicit configuration
    config = ContextConfig(
        optimal_budget=30_000,
        strategy=ContextStrategy.PROGRESSIVE,
        cost_per_million_input=3.0,  # Anthropic Sonnet rate
    )

    # Mode-aware defaults
    config = ContextConfig.for_mode("autonomous")

Architecture:
    ContextEngine (Facade)
    ├── ObservationMasker (Tier 0 — always, post-response)
    ├── ContextLedger (region-aware token accounting)
    ├── CompactionPipeline (progressive strategy chain)
    │   ├── ToolResultCompactor  (@ 60% — Minor GC)
    │   ├── ConversationCompactor (@ 75% — Major GC)
    │   └── EmergencyCompactor   (@ 90% — Full GC)
    └── ContentStore (offloaded artifact storage)
"""

from nucleusiq.agents.context.budget import ContextBudget, ContextLedger, Region
from nucleusiq.agents.context.config import (
    ContextConfig,
    ContextStrategy,
    SummarySchema,
)
from nucleusiq.agents.context.counter import DefaultTokenCounter, TokenCounter
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.pipeline import CompactionPipeline
from nucleusiq.agents.context.store import ContentRef, ContentStore
from nucleusiq.agents.context.strategies import (
    CompactionResult,
    CompactionStrategy,
    ConversationCompactor,
    EmergencyCompactor,
    ObservationMasker,
    ToolResultCompactor,
)
from nucleusiq.agents.context.telemetry import CompactionEvent, ContextTelemetry

__all__ = [
    "CompactionEvent",
    "CompactionPipeline",
    "CompactionResult",
    "CompactionStrategy",
    "ContentRef",
    "ContentStore",
    "ContextBudget",
    "ContextConfig",
    "ContextEngine",
    "ContextStrategy",
    "ContextLedger",
    "ContextTelemetry",
    "ConversationCompactor",
    "DefaultTokenCounter",
    "EmergencyCompactor",
    "ObservationMasker",
    "Region",
    "SummarySchema",
    "TokenCounter",
    "ToolResultCompactor",
]
