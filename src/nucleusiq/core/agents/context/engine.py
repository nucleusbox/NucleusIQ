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
from nucleusiq.agents.context.compactor import Compactor
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.counter import DefaultTokenCounter, TokenCounter
from nucleusiq.agents.context.policy import (
    ContextPolicy,
    PolicyClassifier,
    ResolvedPolicy,
)
from nucleusiq.agents.context.recall_tracker import RecallTracker
from nucleusiq.agents.context.store import ContentStore
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


class _PipelineFacade:
    """Lightweight object exposing the ``tier_count`` attribute that
    callers used to read off ``engine._pipeline``.

    Step 3 removed the separate :class:`CompactionPipeline` object —
    :class:`Compactor` does the orchestration internally.  This facade
    computes ``tier_count`` from the strategy config so the legacy
    introspection surface keeps returning the same numbers
    (``2`` for ``truncate_only``, ``3`` for ``progressive``,
    ``0`` for ``none``).
    """

    __slots__ = ("_config",)

    def __init__(self, config: ContextConfig) -> None:
        self._config = config

    @property
    def tier_count(self) -> int:
        if self._config.strategy == "none":
            return 0
        if self._config.strategy == "truncate_only":
            return 2
        return 3


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
        "_compactor",
        "_masking_enabled",
        "_tracer",
        "_events",
        "_peak_utilization",
        "_checkpoints",
        "_warnings",
        "_observations_masked",
        "_tokens_masked",
        "_total_tokens_sent",
        "_tokens_before_mgmt",
        "_tokens_after_mgmt",
        "_synthesis_rehydrated_count",
        "_synthesis_rehydrated_tokens",
        "_synthesis_refs_selected",
        "_synthesis_refs_skipped",
        "_resolved_max",
        "_resolved_optimal",
        "_masker_triggered_count",
        "_masker_skipped_count",
        # Context Mgmt v2 — Step 2
        "_classifier",
        "_recall_tracker",
        "_policies",
        "_policy_breakdown",
        "_policy_source_breakdown",
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
        # v0.7.9 — optimal_budget is now adaptive to the model's real
        # context window.  When the user does not override it, we
        # compute ``min(fraction × ctx_window, ceiling)`` so that
        # compaction triggers fire at the right absolute utilization
        # for every provider (8K Llama through 2M Gemini).  The
        # ledger clamp ``min(optimal, resolved_max)`` remains as a
        # safety rail in case the user sets an explicit value larger
        # than the model's hard ceiling.
        self._resolved_optimal = ContextConfig.resolve_optimal_budget(
            config, self._resolved_max
        )
        self._ledger = ContextLedger(
            min(self._resolved_optimal, self._resolved_max),
            config.response_reserve,
        )
        self._store = ContentStore()
        # Step 3: a single Compactor replaces the v1 pipeline +
        # the four strategy classes.  See ``compactor.py``.
        self._compactor = Compactor()
        # Match the v1 contract: masking is governed by
        # ``enable_observation_masking`` alone.  ``strategy='none'``
        # disables prepare-time compaction but does **not** turn off
        # the post-response masker — callers explicitly opt out via
        # ``enable_observation_masking=False`` if they want that.
        self._masking_enabled = bool(config.enable_observation_masking)
        self._tracer = tracer
        self._events: list[CompactionEvent] = []
        self._peak_utilization: float = 0.0
        self._checkpoints: list[str] = []
        self._warnings: list[str] = []
        self._observations_masked: int = 0
        self._tokens_masked: int = 0
        self._total_tokens_sent: int = 0
        self._tokens_before_mgmt: int = 0
        self._tokens_after_mgmt: int = 0
        self._synthesis_rehydrated_count: int = 0
        self._synthesis_rehydrated_tokens: int = 0
        self._synthesis_refs_selected: tuple[str, ...] = ()
        self._synthesis_refs_skipped: tuple[str, ...] = ()
        self._masker_triggered_count: int = 0
        self._masker_skipped_count: int = 0

        # Context Mgmt v2 — Step 2: tool-result policy + recall tracking.
        # The classifier is stateless, the tracker is per-execution, and
        # ``_policies`` is the per-execution registry mapping
        # ``tool_call_id`` (or a generated key) → ResolvedPolicy.
        # ``_policy_breakdown`` aggregates counts for telemetry so
        # operators can see how the heuristic actually decided.
        self._classifier = PolicyClassifier(config)
        self._recall_tracker = RecallTracker()
        self._policies: dict[str, ResolvedPolicy] = {}
        self._policy_breakdown: dict[str, int] = {}
        self._policy_source_breakdown: dict[str, int] = {}

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
        self._tokens_before_mgmt = budget.allocated
        self._tokens_after_mgmt = budget.allocated
        self._peak_utilization = max(self._peak_utilization, budget.utilization)
        self._total_tokens_sent += budget.allocated

        if budget.utilization >= self._config.tool_compaction_trigger:
            logger.debug(
                "Context utilization %.1f%% of optimal budget — triggering compaction",
                budget.utilization * 100,
            )
            # Hot-set rescue (v2 Step 2): refs the model recalled in
            # the last few turns are pinned so the Compactor cannot
            # evict them out from under the next turn.
            lookback = getattr(self._config, "hot_set_lookback_turns", 3) or 0
            hot_set = (
                frozenset(self._recall_tracker.hot_set(lookback_turns=lookback))
                if lookback > 0
                else frozenset()
            )
            if self._config.strategy == "truncate_only":
                # ``truncate_only`` skipped the conversation tier in
                # the old pipeline.  Step 3 emulates that by gating
                # the conversation pass off via the trigger config —
                # see :meth:`Compactor._conv_pass`.  Truncation +
                # emergency are sufficient for this strategy.
                compacted, events = await self._compactor.compact(
                    messages,
                    budget,
                    self._config.model_copy(update={"compaction_trigger": 1.01}),
                    self._counter,
                    self._store,
                    hot_set=hot_set,
                )
            else:
                compacted, events = await self._compactor.compact(
                    messages,
                    budget,
                    self._config,
                    self._counter,
                    self._store,
                    hot_set=hot_set,
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
            self._tokens_after_mgmt = self._counter.count_messages(compacted)
            return compacted

        return messages

    def post_response(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Post-LLM-response hook: run Tier 0 observation masking when budget
        pressure is real.

        Replaces consumed tool results (those before the last assistant
        message) with slim markers.  Full content is preserved in
        ``ContentStore`` for potential rehydration.

        Runs synchronously — no LLM calls, no async needed.

        Context Mgmt v2 — Step 1 (budget gate)
        --------------------------------------
        Before v2 the masker ran unconditionally after every LLM response.
        That cost output quality on short/medium runs that never came
        close to the budget ceiling, because it stripped tool results
        from the Generator's view even when the Generator had ample
        room to keep them.

        With the gate, masking only fires once utilization of
        ``optimal_budget`` reaches ``config.squeeze_threshold`` (default
        0.70).  Below the threshold the masker is a strict no-op:
        messages are returned unchanged and ``masker_skipped_count``
        ticks up so we can see in telemetry exactly how often the gate
        intervened.

        The gate is bypassed (i.e. masking always runs) when
        ``squeeze_threshold == 0.0`` — useful for unit tests that
        isolate the masker's mechanics from the budget logic, and for
        operators who explicitly want v1 behaviour back.
        """
        # Bump the recall-tracker turn counter on every post_response,
        # whether or not the masker fires.  Step 3's Compactor uses
        # this to compute the hot set ("recalled within last N turns")
        # — the counter must advance per LLM round, not per masking
        # event.
        self._recall_tracker.mark_turn_completed()

        if not self._masking_enabled:
            return messages

        before_tokens = self._counter.count_messages(messages)
        self._tokens_before_mgmt = before_tokens
        self._tokens_after_mgmt = before_tokens

        if self._config.squeeze_threshold > 0.0:
            self._recount(messages)
            utilization = self._ledger.snapshot().utilization
            if utilization < self._config.squeeze_threshold:
                self._masker_skipped_count += 1
                logger.debug(
                    "ObservationMasker gated off: util=%.2f < squeeze=%.2f "
                    "(skipped, total skips=%d)",
                    utilization,
                    self._config.squeeze_threshold,
                    self._masker_skipped_count,
                )
                return messages

        masked, count, freed = self._compactor.mask(
            messages, self._counter, self._store
        )
        self._tokens_after_mgmt = self._counter.count_messages(masked)
        self._masker_triggered_count += 1

        if count > 0:
            self._observations_masked += count
            self._tokens_masked += freed
            logger.info(
                "ObservationMasker: masked %d tool results, freed %d tokens "
                "(util=%.2f, squeeze=%.2f)",
                count,
                freed,
                self._ledger.snapshot().utilization,
                self._config.squeeze_threshold,
            )

        return masked

    def prepare_for_synthesis(
        self,
        messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Auto-rehydrate evidence markers ahead of a tools=None synthesis call.

        Why this exists (Context Mgmt v2 — §7 of the redesign):

        The synthesis pass calls the LLM with ``tools=None``, which
        means the model **cannot** call ``recall_tool_result`` to fetch
        offloaded evidence.  That left a hole in invariant **I4**
        (Generator must have the same rehydration capability as Critic
        / Refiner).  This method closes the hole: before the synthesis
        LLM call we walk the message list newest-first and replace
        evidence markers with their original content, but **only as
        many as fit in the synthesis budget**.  Older markers stay as
        markers — the model still gets the structured fact slots from
        F1 even if the bytes aren't there.

        Algorithm:

        1. Compute available budget::

               budget = max_context_tokens - count(messages) - response_reserve

           If ``budget <= 0`` we have nothing to spare — return
           messages untouched.  This is the "fail-open" branch:
           markers are still informative.

        2. Walk masked tool messages **newest-first** (the model is
           most likely to need the latest evidence first).

        3. Look up each marker's ref in :class:`ContentStore`.  Hit →
           rehydrate (subject to ``tool_result_per_call_max_chars`` cap).
           Miss → skip silently; the marker remains.

        4. After each successful rehydration, decrement the running
           budget by the *delta* (rehydrated tokens - marker tokens).
           Stop as soon as the next rehydration would exceed budget.

        Properties:

        * **Pure** — returns a new list, does not mutate input.
        * **Fail-open** — any exception → original messages returned
          unchanged (logged at debug level).  Synthesis must never
          fail because of a rehydration glitch.
        * **Idempotent** — already-rehydrated tool messages no longer
          start with the mask prefix and are skipped.

        Args:
            messages: The synthesis-pass message list (already
                includes the synthesis nudge).  Order preserved.

        Returns:
            New message list with selected markers rehydrated.
        """
        if not messages:
            self._synthesis_rehydrated_count = 0
            self._synthesis_rehydrated_tokens = 0
            self._synthesis_refs_selected = ()
            self._synthesis_refs_skipped = ()
            return list(messages)

        try:
            from nucleusiq.agents.chat_models import ChatMessage as CM
            from nucleusiq.agents.context.store import _MASK_PREFIX, _REF_LINE_RE
        except Exception:  # pragma: no cover — defensive
            return list(messages)

        try:
            current_tokens = self._counter.count_messages(messages)
            window = max(0, int(self._resolved_max))
            # ``response_reserve`` is the canonical headroom for the
            # final completion in :class:`ContextConfig`.  We subtract
            # it so a long synthesis answer never collides with the
            # rehydrated prompt.
            reserve = max(0, int(self._config.response_reserve))
            available = window - current_tokens - reserve
        except Exception as exc:
            logger.debug("prepare_for_synthesis: budget calc failed: %s", exc)
            return list(messages)

        self._synthesis_rehydrated_count = 0
        self._synthesis_rehydrated_tokens = 0
        selected_refs: list[str] = []
        skipped_refs: list[str] = []

        if available <= 0:
            self._synthesis_refs_selected = ()
            self._synthesis_refs_skipped = ()
            return list(messages)

        max_chars = max(1, int(self._config.tool_result_per_call_max_chars))

        # Newest-first traversal — index walking so we can write back in place.
        out: list[ChatMessage] = list(messages)

        for i in range(len(out) - 1, -1, -1):
            if available <= 0:
                break
            msg = out[i]
            content = msg.content
            if (
                msg.role != "tool"
                or not isinstance(content, str)
                or not content.startswith(_MASK_PREFIX)
            ):
                continue

            try:
                match = _REF_LINE_RE.search(content)
                if not match:
                    continue
                key = match.group(1)
                raw = self._store.retrieve(key)
            except Exception as exc:
                logger.debug("prepare_for_synthesis: store lookup failed: %s", exc)
                continue

            if raw is None:
                skipped_refs.append(key)
                continue

            if len(raw) > max_chars:
                raw = raw[:max_chars] + "\n... (truncated)"

            try:
                marker_tokens = self._counter.count(content)
                raw_tokens = self._counter.count(raw)
            except Exception:
                continue

            delta = raw_tokens - marker_tokens
            # If this rehydration would over-spend, stop — older markers
            # are even larger and cheaper to keep masked.
            if delta > available:
                skipped_refs.append(key)
                break

            out[i] = CM(
                role=msg.role,
                content=raw,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
            )
            available -= delta
            self._synthesis_rehydrated_count += 1
            self._synthesis_rehydrated_tokens += raw_tokens
            selected_refs.append(key)

        self._synthesis_refs_selected = tuple(selected_refs)
        self._synthesis_refs_skipped = tuple(skipped_refs)

        if self._synthesis_rehydrated_count > 0:
            logger.info(
                "Synthesis rehydration: rehydrated %d evidence marker(s), "
                "%d tokens added to synthesis prompt",
                self._synthesis_rehydrated_count,
                self._synthesis_rehydrated_tokens,
            )
        return out

    def ingest_tool_result(
        self,
        content: str,
        tool_name: str,
        *,
        tool_call_id: str | None = None,
        declared_policy: ContextPolicy | None = None,
    ) -> str:
        """Post-tool hook: classify + register tool results for lifecycle mgmt.

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

        Context Mgmt v2 — Step 2 (classification)
        -----------------------------------------
        Every result is classified to ``EVIDENCE`` or ``EPHEMERAL`` at
        ingestion time via :class:`PolicyClassifier`.  The resolved
        policy is stored keyed by ``tool_call_id`` (or a generated
        fallback key) so the masker / compactor can look it up later
        without re-running the heuristic.  Telemetry records both the
        policy and its provenance (``tool_decoration`` /
        ``name_pattern`` / ``size`` / ``default``) so operators can
        audit decisions.

        Args:
            content: The tool's serialised return value.  Always
                returned unchanged — classification is a side effect.
            tool_name: Tool name; used by the heuristic and as part of
                the storage key.
            tool_call_id: The originating ``tool_call_id`` from the
                LLM.  If provided, the resolved policy is keyed on
                it so subsequent eviction passes can find it
                deterministically.  Optional — falls back to a
                generated UUID for callers that don't have an id
                yet (legacy path).
            declared_policy: The author's policy declaration from
                ``@tool(context_policy=...)``.  When ``EVIDENCE`` or
                ``EPHEMERAL`` it short-circuits the heuristic;
                ``AUTO`` / ``None`` defer to the classifier.
        """
        if self._config.strategy == "none":
            return content

        content_tokens = self._counter.count(content)

        # ----- Always classify (free, no I/O, deterministic). -----
        resolved = self._classifier.classify(
            tool_name=tool_name,
            content_tokens=content_tokens,
            declared_policy=declared_policy,
        )
        policy_key = tool_call_id or f"{tool_name}:{uuid.uuid4().hex[:12]}"
        self._policies[policy_key] = resolved
        self._policy_breakdown[resolved.policy.value] = (
            self._policy_breakdown.get(resolved.policy.value, 0) + 1
        )
        self._policy_source_breakdown[resolved.source] = (
            self._policy_source_breakdown.get(resolved.source, 0) + 1
        )

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
                "Registered %s result (%d tokens, policy=%s/%s) in "
                "ContentStore → %s (full content kept in context for model)",
                tool_name,
                content_tokens,
                resolved.policy.value,
                resolved.source,
                key,
            )

        return content

    # ------------------------------------------------------------------ #
    # Context Mgmt v2 — Step 2: policy / recall accessors                #
    # ------------------------------------------------------------------ #

    def get_policy_for(self, tool_call_id: str) -> ResolvedPolicy | None:
        """Return the resolved policy for a previously-ingested result.

        Lookup is by the ``tool_call_id`` passed to
        :meth:`ingest_tool_result`.  Returns ``None`` if the engine
        has not seen that id (e.g. the result was ingested by a
        legacy caller without an id, or the caller is pre-Step 2).
        """
        return self._policies.get(tool_call_id)

    @property
    def recall_tracker(self) -> RecallTracker:
        """Per-execution recall-event log (Context Mgmt v2 — Step 2)."""
        return self._recall_tracker

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
    def _pipeline(self) -> _PipelineFacade:
        """Back-compat alias for code that referenced ``engine._pipeline``.

        The Step-3 engine no longer holds a separate ``CompactionPipeline``
        — :class:`Compactor` is the single source of truth.  This
        property returns a tiny facade with the legacy ``tier_count``
        attribute so older tests / instrumentation that introspected
        ``engine._pipeline.tier_count`` keep working.

        For new code, prefer :attr:`compactor` (no underscore) and
        :attr:`config` for strategy / threshold inspection.
        """
        return _PipelineFacade(self._config)

    @property
    def compactor(self) -> Compactor:
        """The single :class:`Compactor` driving this engine.

        Exposed read-only so tests / instrumentation can introspect
        without poking through the leading-underscore slot.
        """
        return self._compactor

    @property
    def budget(self) -> ContextBudget:
        """Current budget snapshot (immutable)."""
        return self._ledger.snapshot()

    @property
    def store(self) -> ContentStore:
        """Access to the content store for retrieval."""
        return self._store

    @property
    def config(self) -> ContextConfig:
        """Read-only view of the engine's :class:`ContextConfig`.

        Exposed so per-execution helpers (e.g. the recall tools) can
        consult tunables like ``tool_result_per_call_max_chars``
        without re-reading the agent's config.  The config is
        immutable (Pydantic frozen) so this is a safe read.
        """
        return self._config

    @property
    def token_counter(self) -> TokenCounter:
        """The :class:`TokenCounter` used by this engine.

        Exposed for the recall tools so their telemetry (``tokens``
        recalled) is computed against the same counter the engine
        uses internally — matching numbers across telemetry sources
        is a debugging-quality-of-life feature, not a correctness
        requirement.
        """
        return self._counter

    @property
    def telemetry(self) -> ContextTelemetry:
        """Full observability into context management.

        F3 — reports masker vs compactor token savings separately so
        researchers and operators can see which mechanism is actually
        doing the work on a given run.  ``tokens_freed_total`` remains
        the additive sum of both for backward compatibility.
        """
        budget = self._ledger.snapshot()
        compactor_freed = sum(e.tokens_freed for e in self._events)
        masker_freed = self._tokens_masked
        total_freed = compactor_freed + masker_freed

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
            tokens_before_mgmt=self._tokens_before_mgmt,
            tokens_after_mgmt=self._tokens_after_mgmt,
            tokens_freed_total=total_freed,
            compactor_tokens_freed=compactor_freed,
            masker_tokens_freed=masker_freed,
            artifacts_offloaded=self._store.size,
            region_breakdown=budget.by_region,
            context_limit=budget.max_tokens,
            response_reserve=budget.response_reserve,
            warnings=tuple(self._warnings),
            observations_masked=self._observations_masked,
            tokens_masked=self._tokens_masked,
            masker_triggered_count=self._masker_triggered_count,
            masker_skipped_count=self._masker_skipped_count,
            optimal_budget=self._resolved_optimal,
            estimated_cost_without_mgmt=round(cost_without, 6),
            estimated_cost_with_mgmt=round(cost_with, 6),
            estimated_savings_pct=round(savings_pct, 2),
            recall_count=self._recall_tracker.recall_count,
            recall_tokens=self._recall_tracker.total_recalled_tokens,
            synthesis_rehydrated_count=self._synthesis_rehydrated_count,
            synthesis_rehydrated_tokens=self._synthesis_rehydrated_tokens,
            synthesis_refs_selected=self._synthesis_refs_selected,
            synthesis_refs_skipped=self._synthesis_refs_skipped,
            policy_breakdown=dict(self._policy_breakdown),
            policy_source_breakdown=dict(self._policy_source_breakdown),
        )
