"""Context Mgmt v2 — Step 2: tool-result policy classification.

A tool result is one of three things in v2:

* **EVIDENCE**: large, citation-worthy, expensive to recover.  When
  evicted under memory pressure it is *offloaded* — full content kept
  in :class:`~nucleusiq.agents.context.store.ContentStore` and a
  marker (with a ``ref`` key) takes its place in the conversation.
  The Generator can recover the content via the
  ``recall_tool_result(ref=...)`` tool.
* **EPHEMERAL**: small, throwaway facts (current time, formatter
  output, boolean validators).  When evicted it is *dropped* — a slim
  marker is left behind purely for trace shape; the content itself
  is lost intentionally because re-running the tool is cheaper than
  storing it.
* **AUTO**: a *deferred* decision.  AUTO is never a terminal state —
  every result classified AUTO is resolved to EVIDENCE or EPHEMERAL
  by :class:`PolicyClassifier` at ingestion time.

The two-source resolution rule (see ``CONTEXT_MANAGEMENT_V2_REDESIGN.md``
§3.2) is the single source of truth for *who* decides:

1. **Tool decoration** (3A) — ``@tool(context_policy=...)``.  Author
   knows best.  EVIDENCE / EPHEMERAL short-circuit the heuristic.
2. **Heuristic classifier** (3C) — ``classify_result(...)``.  Runs when
   the tool is decorated AUTO or has no decoration.  Looks at name
   patterns first, then size, then defaults to EVIDENCE.

The framework default is intentionally **EVIDENCE** (fail-safe):
losing evidence is expensive in dollars and quality; keeping a
20-token timestamp around for an extra turn is free.

Module layout (kept tiny on purpose — no compaction logic here):

* :class:`ContextPolicy` — the enum.
* :class:`ResolvedPolicy` — frozen result + provenance.
* :class:`PolicyClassifier` — the heuristic; pure function wrapped
  in a class only so the patterns/threshold come from
  :class:`~nucleusiq.agents.context.config.ContextConfig` instead of
  module globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from nucleusiq.agents.context.config import ContextConfig


__all__ = [
    "ContextPolicy",
    "ResolvedPolicy",
    "PolicyClassifier",
    "DEFAULT_EVIDENCE_NAME_PATTERNS",
    "DEFAULT_EPHEMERAL_NAME_PATTERNS",
    "DEFAULT_EPHEMERAL_SIZE_THRESHOLD",
]


# ====================================================================== #
# Enum                                                                     #
# ====================================================================== #


class ContextPolicy(str, Enum):
    """How a tool result should be treated under memory pressure.

    String-valued for clean serialisation in telemetry / config.
    """

    #: Preserve under pressure: offload to ContentStore, replace with a
    #: marker that carries a ``ref`` the model can pass to
    #: ``recall_tool_result``.
    EVIDENCE = "evidence"

    #: Drop under pressure: replace with a slim marker.  Content is
    #: not recoverable.  Use for tiny throwaway results where rerunning
    #: the tool is cheaper than persisting the bytes.
    EPHEMERAL = "ephemeral"

    #: Decide at runtime via the heuristic classifier.  Never a
    #: terminal state — every AUTO result is resolved at ingestion
    #: time.  Equivalent to "no decoration".
    AUTO = "auto"


# ====================================================================== #
# Defaults — exposed so tests and config defaults reference one place    #
# ====================================================================== #

#: Tool-name fragments that indicate evidence-shaped output (read,
#: search, fetch, query, retrieve, ...).  Match is *case-insensitive
#: substring* against the tool name.
DEFAULT_EVIDENCE_NAME_PATTERNS: Final[tuple[str, ...]] = (
    "read_",
    "search_",
    "fetch_",
    "pdf_",
    "get_document",
    "query_",
    "retrieve_",
    "load_",
    "download_",
    "annual_report",
    "_excerpt",
)

#: Tool-name fragments that indicate ephemeral output (current time,
#: formatters, validators, predicates).  Match is *case-insensitive
#: substring* against the tool name.
DEFAULT_EPHEMERAL_NAME_PATTERNS: Final[tuple[str, ...]] = (
    "get_time",
    "current_time",
    "now_",
    "calc_",
    "format_",
    "validate_",
    "is_",
    "has_",
    "check_",
)

#: Below this token count an un-pinned, un-named-pattern result is
#: classified EPHEMERAL.  Roughly ``500 tokens ≈ 2000 chars`` of
#: English text — small enough that storing it costs more than
#: re-running the tool would.
DEFAULT_EPHEMERAL_SIZE_THRESHOLD: Final[int] = 500


# ====================================================================== #
# ResolvedPolicy                                                            #
# ====================================================================== #


@dataclass(frozen=True, slots=True)
class ResolvedPolicy:
    """Outcome of policy resolution for a single tool result.

    Attributes:
        policy: Always ``EVIDENCE`` or ``EPHEMERAL`` — never ``AUTO``.
        source: How we decided.  One of ``"tool_decoration"`` (author
            declared explicitly), ``"name_pattern"`` (heuristic
            matched a name fragment), ``"size"`` (heuristic fell
            through to size gate), or ``"default"`` (heuristic
            uncertain → conservative EVIDENCE default).
        confidence: ``0.0 - 1.0`` — informational only.  Telemetry
            surfaces this so users can audit *why* a given result
            was classified the way it was.

    The class is deliberately small and ``frozen``: a ``ResolvedPolicy``
    is a value, not an object with behaviour.  The framework stores
    one of these per ``tool_call_id`` and never mutates it.
    """

    policy: ContextPolicy
    source: str
    confidence: float

    def __post_init__(self) -> None:
        if self.policy is ContextPolicy.AUTO:
            raise ValueError(
                "ResolvedPolicy.policy must be EVIDENCE or EPHEMERAL — "
                "AUTO is a deferred decision, not a terminal state. "
                "Run the classifier before constructing ResolvedPolicy."
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"ResolvedPolicy.confidence must be in [0.0, 1.0], got "
                f"{self.confidence!r}"
            )


# ====================================================================== #
# Classifier                                                                #
# ====================================================================== #


class PolicyClassifier:
    """Resolve a tool result to ``ResolvedPolicy(EVIDENCE|EPHEMERAL)``.

    Inputs are always *one tool result*: name + content + content
    token count.  The classifier is *stateless* — repeated calls with
    the same inputs produce the same output, which keeps telemetry
    explainable and tests trivial.

    The two-source rule (§3.2):

    1.  If the tool was decorated with an explicit policy
        (``EVIDENCE`` / ``EPHEMERAL``), pass it through unchanged.
    2.  Otherwise (decorated ``AUTO`` or undecorated), run the
        heuristic.

    The heuristic itself (§3.3):

    1.  Name pattern match — strongest signal.
    2.  Size gate — anything under the threshold is EPHEMERAL.
    3.  Conservative default — EVIDENCE.

    Why not LLM-based classification?  Because policy decisions run
    on every tool result — running an LLM call on every step would
    cost more than the eviction itself saves.  The heuristic is good
    enough for the common cases and the override hatch (``@tool``
    decoration) covers the rest.
    """

    __slots__ = ("_config",)

    def __init__(self, config: ContextConfig) -> None:
        self._config = config

    def classify(
        self,
        *,
        tool_name: str,
        content_tokens: int,
        declared_policy: ContextPolicy | None = None,
    ) -> ResolvedPolicy:
        """Resolve the policy for one tool result.

        Args:
            tool_name: The tool's registered name.  Heuristic name
                patterns are matched (case-insensitively) against this.
            content_tokens: Token count of the result content.  Used
                by the size gate.  Pass an integer ``≥ 0``.
            declared_policy: The policy declared on the tool itself
                (via ``@tool(context_policy=...)``).  ``None`` is
                treated as ``AUTO`` for symmetry.

        Returns:
            A ``ResolvedPolicy`` whose ``policy`` is always
            ``EVIDENCE`` or ``EPHEMERAL`` — never ``AUTO``.
        """
        # Step 1 — Author-declared override (3A short-circuits the heuristic).
        if declared_policy is ContextPolicy.EVIDENCE:
            return ResolvedPolicy(
                policy=ContextPolicy.EVIDENCE,
                source="tool_decoration",
                confidence=1.0,
            )
        if declared_policy is ContextPolicy.EPHEMERAL:
            return ResolvedPolicy(
                policy=ContextPolicy.EPHEMERAL,
                source="tool_decoration",
                confidence=1.0,
            )

        # Step 2 — Heuristic.  AUTO and None both fall through here.
        evidence_patterns = self._evidence_patterns()
        ephemeral_patterns = self._ephemeral_patterns()
        size_threshold = self._size_threshold()

        name_lc = tool_name.lower() if tool_name else ""

        for pat in evidence_patterns:
            if pat and pat in name_lc:
                return ResolvedPolicy(
                    policy=ContextPolicy.EVIDENCE,
                    source="name_pattern",
                    confidence=0.9,
                )

        for pat in ephemeral_patterns:
            if pat and pat in name_lc:
                return ResolvedPolicy(
                    policy=ContextPolicy.EPHEMERAL,
                    source="name_pattern",
                    confidence=0.9,
                )

        if content_tokens < size_threshold:
            return ResolvedPolicy(
                policy=ContextPolicy.EPHEMERAL,
                source="size",
                confidence=0.7,
            )

        return ResolvedPolicy(
            policy=ContextPolicy.EVIDENCE,
            source="default",
            confidence=0.5,
        )

    # ------------------------------------------------------------------ #
    # Config plumbing — kept private so we can move these knobs around   #
    # without touching call sites.                                       #
    # ------------------------------------------------------------------ #

    def _evidence_patterns(self) -> tuple[str, ...]:
        return tuple(
            getattr(
                self._config,
                "evidence_name_patterns",
                DEFAULT_EVIDENCE_NAME_PATTERNS,
            )
        )

    def _ephemeral_patterns(self) -> tuple[str, ...]:
        return tuple(
            getattr(
                self._config,
                "ephemeral_name_patterns",
                DEFAULT_EPHEMERAL_NAME_PATTERNS,
            )
        )

    def _size_threshold(self) -> int:
        return int(
            getattr(
                self._config,
                "ephemeral_size_threshold",
                DEFAULT_EPHEMERAL_SIZE_THRESHOLD,
            )
        )
