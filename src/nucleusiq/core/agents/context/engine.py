"""ContextEngine — facade for context window management.

This is the ONLY class that execution modes interact with.
Hides the complexity of ledger, pipeline, masker, and store behind
a simple 4-method API:

    prepare()           — before each LLM call
    post_response()     — after each LLM response (Tier 0 masking)
    ingest_tool_result() — after each tool execution
    checkpoint()        — at task boundaries (optional)

Lifecycle:
    1. Created per ``Agent.execute()`` call (never shared).
    2. ``prepare(messages)`` called before each LLM call.
    3. LLM call executes.
    4. ``post_response(messages)`` called after LLM response.
    5. ``ingest_tool_result(content, name)`` called after each tool.
    6. ``checkpoint(label)`` called at task boundaries (optional).
    7. ``telemetry`` property read at ``_build_result()`` time.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from nucleusiq.agents.context.budget import ContextLedger, Region
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.counter import DefaultTokenCounter, TokenCounter
from nucleusiq.agents.context.pipeline import CompactionPipeline
from nucleusiq.agents.context.store import ContentStore
from nucleusiq.agents.context.strategies.base import CompactionStrategy
from nucleusiq.agents.context.strategies.conversation import ConversationCompactor
from nucleusiq.agents.context.strategies.emergency import EmergencyCompactor
from nucleusiq.agents.context.strategies.observation_masker import ObservationMasker
from nucleusiq.agents.context.strategies.tool_result import ToolResultCompactor
from nucleusiq.agents.context.telemetry import CompactionEvent, ContextTelemetry

if TYPE_CHECKING:
    from nucleusiq.agents.chat_models import ChatMessage
    from nucleusiq.agents.context.budget import ContextBudget
    from nucleusiq.agents.observability.protocol import ExecutionTracerProtocol

logger = logging.getLogger(__name__)

_ROLE_TO_REGION: dict[str, Region] = {
    "system": Region.SYSTEM,
    "user": Region.USER,
    "assistant": Region.ASSISTANT,
    "tool": Region.TOOL_RESULT,
    "function": Region.TOOL_RESULT,
}


def _msg_id(msg: ChatMessage) -> str:
    """Generate or retrieve a stable ID for a message."""
    if msg.tool_call_id:
        return f"tool:{msg.tool_call_id}"
    return f"msg:{uuid.uuid4().hex[:12]}"


class ContextEngine:
    """Manages the context window for a single agent execution lifecycle.

    Facade pattern — hides ledger, pipeline, masker, and store behind
    a simple API that execution modes call at three points:

    1. ``prepare()`` — before each LLM call (compaction pipeline).
    2. ``post_response()`` — after each LLM response (observation masking).
    3. ``ingest_tool_result()`` — after each tool execution (offloading).

    Phase 2 changes:
        - Compaction thresholds fire against ``optimal_budget``, not
          ``max_context_tokens``.
        - ``post_response()`` runs ``ObservationMasker`` (Tier 0).
        - Telemetry includes cost estimation and masking counts.
    """

    __slots__ = (
        "_config",
        "_counter",
        "_ledger",
        "_store",
        "_pipeline",
        "_masker",
        "_tracer",
        "_events",
        "_peak_utilization",
        "_checkpoints",
        "_warnings",
        "_observations_masked",
        "_tokens_masked",
        "_total_tokens_sent",
        "_resolved_max",
    )

    def __init__(
        self,
        config: ContextConfig,
        token_counter: TokenCounter | None = None,
        *,
        max_tokens: int = 128_000,
        tracer: ExecutionTracerProtocol | None = None,
    ) -> None:
        self._config = config
        self._counter: TokenCounter = token_counter or DefaultTokenCounter()

        self._resolved_max = config.max_context_tokens or max_tokens
        self._ledger = ContextLedger(
            min(config.optimal_budget, self._resolved_max),
            config.response_reserve,
        )
        self._store = ContentStore()
        self._pipeline = self._build_pipeline(config)
        self._masker = (
            ObservationMasker() if config.enable_observation_masking else None
        )
        self._tracer = tracer
        self._events: list[CompactionEvent] = []
        self._peak_utilization: float = 0.0
        self._checkpoints: list[str] = []
        self._warnings: list[str] = []
        self._observations_masked: int = 0
        self._tokens_masked: int = 0
        self._total_tokens_sent: int = 0

    @staticmethod
    def _build_pipeline(config: ContextConfig) -> CompactionPipeline:
        """Build the compaction pipeline based on strategy config."""
        if config.strategy == "none":
            return CompactionPipeline([])

        tiers: list[tuple[float, CompactionStrategy]] = [
            (config.tool_compaction_trigger, ToolResultCompactor()),
            (config.compaction_trigger, ConversationCompactor()),
            (config.emergency_trigger, EmergencyCompactor()),
        ]
        if config.strategy == "truncate_only":
            tiers = [
                (config.tool_compaction_trigger, ToolResultCompactor()),
                (config.emergency_trigger, EmergencyCompactor()),
            ]
        return CompactionPipeline(tiers)

    async def prepare(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Pre-LLM-call hook: ensure messages fit within optimal budget.

        1. Recount all message tokens (messages may have changed).
        2. Track tokens being sent for cost estimation.
        3. If utilization > trigger threshold, run compaction pipeline.
        4. Return (possibly compacted) messages.
        """
        if self._config.strategy == "none":
            return messages

        self._recount(messages)
        budget = self._ledger.snapshot()
        self._peak_utilization = max(self._peak_utilization, budget.utilization)
        self._total_tokens_sent += budget.allocated

        if budget.utilization >= self._config.tool_compaction_trigger:
            logger.debug(
                "Context utilization %.1f%% of optimal budget — triggering compaction",
                budget.utilization * 100,
            )
            compacted, events = await self._pipeline.run(
                messages, budget, self._config, self._counter, self._store
            )
            self._events.extend(events)

            for event in events:
                if event.tokens_freed > 0:
                    logger.info(
                        "Compaction [%s]: freed %d tokens (%.1f%% → %.1f%%)",
                        event.strategy,
                        event.tokens_freed,
                        event.trigger_utilization * 100,
                        (event.tokens_after / budget.effective_limit * 100)
                        if budget.effective_limit > 0
                        else 100,
                    )

            for result_event in events:
                for w in getattr(result_event, "warnings", ()):
                    if isinstance(w, str):
                        self._warnings.append(w)

            self._recount(compacted)
            return compacted

        return messages

    def post_response(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Post-LLM-response hook: run Tier 0 observation masking.

        Replaces consumed tool results (those before the last assistant
        message) with slim markers.  Full content is preserved in
        ``ContentStore`` for potential rehydration.

        This runs synchronously — no LLM calls, no async needed.
        """
        if self._masker is None:
            return messages

        masked, count, freed = self._masker.mask(messages, self._counter, self._store)

        if count > 0:
            self._observations_masked += count
            self._tokens_masked += freed
            logger.info(
                "ObservationMasker: masked %d tool results, freed %d tokens",
                count,
                freed,
            )

        return masked

    def ingest_tool_result(self, content: str, tool_name: str) -> str:
        """Post-tool hook: register large tool results for lifecycle management.

        Stores oversized results in ``ContentStore`` for rehydration and
        telemetry, but **always returns full content** so the LLM sees
        real data on the first pass.  Compression happens later via
        ``ObservationMasker`` in ``post_response()`` — only after the
        model has processed the result at least once.

        Why not replace immediately?
            Replacing before the model sees data causes quality
            degradation — the model receives abbreviated previews and
            cannot reason over the actual content.  The masker handles
            cleanup post-response, giving real token savings without
            sacrificing output quality.
        """
        if self._config.strategy == "none":
            return content

        content_tokens = self._counter.count(content)

        if content_tokens <= self._config.tool_result_threshold:
            return content

        if self._config.enable_offloading:
            key = f"{tool_name}:{uuid.uuid4().hex[:12]}"
            self._store.store(
                key=key,
                content=content,
                original_tokens=content_tokens,
            )
            logger.debug(
                "Registered %s result (%d tokens) in ContentStore → %s "
                "(full content kept in context for model)",
                tool_name,
                content_tokens,
                key,
            )

        return content

    def checkpoint(self, label: str) -> None:
        """Mark a task boundary for telemetry and future smart compaction."""
        self._checkpoints.append(label)
        logger.debug(
            "Context checkpoint: %s (util=%.1f%%)", label, self.budget.utilization * 100
        )

    def _recount(self, messages: list[ChatMessage]) -> None:
        """Rebuild ledger from scratch for the current message list."""
        self._ledger.reset()
        for msg in messages:
            region = _ROLE_TO_REGION.get(msg.role, Region.USER)
            tokens = self._count_message_tokens(msg)
            msg_id = _msg_id(msg)

            source_type = "system" if msg.role == "system" else msg.role
            self._ledger.allocate(
                msg_id=msg_id,
                tokens=tokens,
                region=region,
                source_type=source_type,
                importance=1.0 if msg.role == "system" else 0.5,
                restorable=(
                    region == Region.TOOL_RESULT and self._store.contains(msg_id)
                ),
            )

    def _count_message_tokens(self, msg: ChatMessage) -> int:
        """Count tokens for a single message including framing."""
        tokens = 4  # role/name framing overhead
        content = msg.content
        if isinstance(content, str):
            tokens += self._counter.count(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if text:
                        tokens += self._counter.count(text)
        if msg.name:
            tokens += self._counter.count(msg.name)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tokens += self._counter.count(str(tc))
        return tokens

    @property
    def budget(self) -> ContextBudget:
        """Current budget snapshot (immutable)."""
        return self._ledger.snapshot()

    @property
    def store(self) -> ContentStore:
        """Access to the content store for retrieval."""
        return self._store

    @property
    def telemetry(self) -> ContextTelemetry:
        """Full observability into context management."""
        budget = self._ledger.snapshot()
        total_freed = sum(e.tokens_freed for e in self._events) + self._tokens_masked

        cost_without = 0.0
        cost_with = 0.0
        savings_pct = 0.0
        if (
            self._config.cost_per_million_input is not None
            and self._total_tokens_sent > 0
        ):
            rate = self._config.cost_per_million_input / 1_000_000
            tokens_without_mgmt = self._total_tokens_sent + total_freed
            cost_without = tokens_without_mgmt * rate
            cost_with = self._total_tokens_sent * rate
            if cost_without > 0:
                savings_pct = ((cost_without - cost_with) / cost_without) * 100

        return ContextTelemetry(
            peak_utilization=self._peak_utilization,
            final_utilization=budget.utilization,
            compaction_count=len(self._events),
            compaction_events=tuple(self._events),
            tokens_freed_total=total_freed,
            artifacts_offloaded=self._store.size,
            region_breakdown=budget.by_region,
            context_limit=budget.max_tokens,
            response_reserve=budget.response_reserve,
            warnings=tuple(self._warnings),
            observations_masked=self._observations_masked,
            tokens_masked=self._tokens_masked,
            optimal_budget=self._config.optimal_budget,
            estimated_cost_without_mgmt=round(cost_without, 6),
            estimated_cost_with_mgmt=round(cost_with, 6),
            estimated_savings_pct=round(savings_pct, 2),
        )
