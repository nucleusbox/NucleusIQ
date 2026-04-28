"""Context window management — region-aware progressive compaction.

Public API for context management in NucleusIQ agents.

Quick start::

    from nucleusiq.agents.context import ContextConfig, ContextEngine

    # Zero-config: optimal_budget auto-resolves from the model's
    # real context window (70% × ctx_window, capped at 120K),
    # observation masking ON.
    config = ContextConfig()

    # Explicit configuration
    config = ContextConfig(
        optimal_budget=30_000,
        strategy=ContextStrategy.PROGRESSIVE,
        cost_per_million_input=3.0,  # Anthropic Sonnet rate
    )

    # Mode-aware defaults
    config = ContextConfig.for_mode("autonomous")

Architecture (Context Mgmt v2 — Step 3):
    ContextEngine (Facade)
    ├── Compactor (priority-ordered single class — replaces v1's
    │   pipeline + 4 strategies; see ``compactor.py``)
    ├── PolicyClassifier  (EVIDENCE / EPHEMERAL classification)
    ├── ContextLedger     (region-aware token accounting)
    ├── ContentStore      (offloaded artefact storage)
    ├── RecallTracker     (hot-set tracking for recall pinning)
    └── recall_tools      (auto-injected ``recall_tool_result``)
"""

from nucleusiq.agents.context.budget import ContextBudget, ContextLedger, Region
from nucleusiq.agents.context.compactor import (
    CompactionPipeline,
    CompactionResult,
    CompactionStrategy,
    Compactor,
    ConversationCompactor,
    EmergencyCompactor,
    ObservationMasker,
    ToolResultCompactor,
)
from nucleusiq.agents.context.config import (
    ContextConfig,
    ContextStrategy,
    SummarySchema,
)
from nucleusiq.agents.context.counter import DefaultTokenCounter, TokenCounter
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.store import ContentRef, ContentStore
from nucleusiq.agents.context.telemetry import CompactionEvent, ContextTelemetry

__all__ = [
    "Compactor",
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
