# src/nucleusiq/agents/config/agent_config.py
from __future__ import annotations

from enum import Enum
from typing import Literal

from nucleusiq.agents.config.observability_config import ObservabilityConfig
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.llms.llm_params import LLMParams
from pydantic import BaseModel, Field, field_validator, model_validator


class ExecutionMode(str, Enum):
    """Execution modes (Gearbox Strategy) for agent execution."""

    DIRECT = "direct"  # Gear 1: Fast, optional tools (max 25)
    STANDARD = "standard"  # Gear 2: Tool-enabled loop (max 80) — default
    AUTONOMOUS = "autonomous"  # Gear 3: Orchestration + Critic/Refiner (max 300)


class AgentConfig(BaseModel):
    """Configuration settings for agent behavior."""

    max_execution_time: int = Field(
        default=3600,
        description=(
            "Wall-clock budget for one ``execute()`` in seconds (0 = unlimited). "
            "Checked before every LLM round, tool batch and Autonomous attempt; "
            "when exceeded the loop stops with termination reason ``deadline`` "
            "and produces its best answer from the evidence gathered so far. "
            "Sub-agents receive the parent's *remaining* time."
        ),
    )
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    allow_code_execution: bool = Field(
        default=False, description="Enable code execution capabilities"
    )
    code_execution_mode: Literal["safe", "unsafe"] = Field(
        default="safe", description="Code execution safety mode"
    )
    respect_context_window: bool = Field(
        default=True, description="Maintain context within model's window"
    )
    context: ContextConfig | None = Field(
        default=None,
        description=(
            "Context window management configuration. "
            "None = uses respect_context_window flag (legacy). "
            "ContextConfig() = auto-detect with defaults. "
            "ContextConfig(max_context_tokens=50000) = explicit override."
        ),
    )
    verbose: bool = Field(
        default=False,
        description="Enable detailed logging (legacy — prefer observability)",
    )
    enable_tracing: bool = Field(
        default=False,
        description=(
            "Populate AgentResult with execution trace data "
            "(llm_calls, tool_calls, warnings). Off by default for zero overhead. "
            "Legacy — prefer observability.tracing."
        ),
    )
    observability: ObservabilityConfig | None = Field(
        default=None,
        description=(
            "Unified observability config. When set, takes precedence over "
            "verbose and enable_tracing. None = use legacy fields."
        ),
    )
    # Gearbox Strategy: Execution Modes
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.STANDARD,
        description="Execution mode (gear): DIRECT (fast, optional tools), STANDARD (tool-enabled loop), AUTONOMOUS (orchestration + verification)",
    )
    require_quality_check: bool = Field(
        default=False,
        description="Require quality check before returning (autonomous mode only). Uses Critic component to review output.",
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum iterations for iterative agents (autonomous mode, ReAct, etc.)",
    )

    # LLM call budgets (model-agnostic knobs)
    llm_max_output_tokens: int = Field(
        default=2048,
        description="Token budget for normal LLM calls.",
    )
    step_inference_max_tokens: int = Field(
        default=2048,
        description="Token budget for per-step tool-argument inference.",
    )

    # Timeout settings (in seconds)
    step_timeout: int = Field(
        default=60,
        description=(
            "Timeout in seconds for one tool execution inside the tool loop. "
            "Enforced only when set explicitly (the default is documentation "
            "only, so long-running tools keep working). On expiry the tool "
            "result becomes an error message and the loop continues."
        ),
    )
    llm_call_timeout: int = Field(
        default=90,
        description=(
            "Timeout in seconds for one LLM API call. Enforced only when set "
            "explicitly — reasoning models routinely exceed 90 s, so the "
            "default is not applied. On expiry the call raises "
            "``LLMTimeoutError``."
        ),
    )
    step_max_retries: int = Field(
        default=2,
        description="Maximum retries for a failed step before giving up (0 = no retries).",
    )

    # Tool limits
    max_tool_calls: int | None = Field(
        default=None,
        description=(
            "Maximum tool calls per execution. If None, uses mode defaults: "
            "DIRECT=25, STANDARD=80, AUTONOMOUS=300."
        ),
    )
    max_context_tool_calls: int | None = Field(
        default=None,
        description=(
            "Separate cap for framework context-management tool calls "
            "(recall_tool_result, list_recalled_evidence, workspace / "
            "evidence / corpus tools). These never count toward "
            "``max_tool_calls`` — that quota is for the user's external "
            "actions — but without their own cap a model stuck retrying a "
            "recall could loop indefinitely. None = 2 × effective "
            "``max_tool_calls``. When exhausted the loop stops with "
            "termination reason ``context_tool_budget``."
        ),
    )

    # Synthesis pass (breaks mode inertia after heavy tool use)
    enable_synthesis: bool = Field(
        default=True,
        description=(
            "After multiple rounds of tool calls, make one final LLM call "
            "without tools to produce the synthesized output. Prevents "
            "mode inertia where the model stays in tool-calling behaviour "
            "and returns a terse summary instead of the full deliverable. "
            "Automatically skipped when response_format is set — the schema "
            "is the deliverable; a prose synthesis pass would overwrite it."
        ),
    )
    synthesis_word_threshold: int = Field(
        default=500,
        description=(
            "Minimum word count below which the synthesis pass fires. "
            "If the model already produced content above this threshold, "
            "synthesis is skipped (the output is already substantial). "
            "Set to 0 to always synthesize when enable_synthesis is True."
        ),
    )

    # Autonomous mode
    critique_rounds: int = Field(
        default=3,
        description=(
            "[DEPRECATED] Use max_retries instead. Kept for backward compatibility. "
            "In the new architecture, max_retries controls validation retry cycles."
        ),
    )
    enable_decomposition: bool = Field(
        default=True,
        description=(
            "Autonomous mode only. When False the Decomposer classifier is "
            "never called: the task runs as a single agent with validation, "
            "Critic and Refiner intact and no sub-agents are spawned. Use "
            "for jobs where every part depends on the same documents or "
            "fills one output record — splitting those loses coverage. "
            "``max_sub_agents=1`` has the same routing effect but still "
            "pays the classifier LLM call."
        ),
    )
    max_sub_agents: int = Field(
        default=5,
        description=(
            "Maximum parallel sub-agents for complex task decomposition "
            "in autonomous mode."
        ),
    )
    decomposition_max_owners_per_resource: int = Field(
        default=1,
        ge=1,
        description=(
            "Coverage contract for COMPLEX splits when ``Task.resources`` is "
            "set: every resource must be claimed by at least one sub-task and "
            "by at most this many. A split that shares a source between "
            "children (each re-reads the same documents) is downgraded to "
            "SIMPLE with the reason in the run report."
        ),
    )
    decomposition_gather_first: bool = Field(
        default=False,
        description=(
            "Opt-in for COMPLEX runs with ``Task.resources``: before the "
            "analysis children start, one read-only 'gather' child fetches "
            "every resource (idempotent tools only, at most 2 × len(resources) "
            "tool calls) into the parent's shared evidence stores. Children "
            "then search / recall that material instead of each re-reading "
            "the same documents. Requires tools declared ``idempotent=True``."
        ),
    )
    coverage_followup: bool = Field(
        default=True,
        description=(
            "Only when ``Task.resources`` is set. After the sub-tasks (COMPLEX) "
            "or the first accepted-by-validation answer (SIMPLE), resources no "
            "tool touched and the answer does not name get exactly one bounded "
            "follow-up: a child capped at 2 × len(unprocessed) tool calls, or "
            "one retry attempt telling the model which resources to process. "
            "Coverage is always recorded in ``result.metadata['coverage']`` / "
            "``diagnostics.coverage``; ``evidence_gate_enforce=True`` turns a "
            "residual gap into an ABSTAINED result."
        ),
    )

    # Preflight fitness (Autonomous mode).  ``working`` = window −
    # response reserve − system prompt − tool schemas, measured at setup.
    preflight_min_working_tokens: int = Field(
        default=16_000,
        ge=0,
        description=(
            "Autonomous mode needs at least this many working tokens per "
            "call (window minus reply reserve, system prompt and tool "
            "schemas). Below it the run is downgraded to STANDARD (see "
            "``preflight_downgrade``) because Critic/Refiner hand-offs and "
            "sub-agent synthesis cannot fit. 0 disables the check."
        ),
    )
    preflight_marginal_working_tokens: int = Field(
        default=32_000,
        ge=0,
        description=(
            "Below this many working tokens Autonomous mode runs but the "
            "run report marks fitness ``marginal`` and logs the breakdown."
        ),
    )
    preflight_standard_min_working_tokens: int = Field(
        default=8_000,
        ge=0,
        description=(
            "STANDARD mode logs a warning (never downgrades) below this many "
            "working tokens."
        ),
    )
    preflight_downgrade: bool = Field(
        default=True,
        description=(
            "When the Autonomous preflight finds fewer than "
            "``preflight_min_working_tokens`` working tokens, run the task in "
            "STANDARD mode instead (recorded as decision ``preflight`` and "
            "event ``preflight_downgraded``). Set False to force Autonomous "
            "anyway; the warning is still logged."
        ),
    )
    sub_agent_context: ContextConfig | None = Field(
        default=None,
        description=(
            "Context budget for Autonomous COMPLEX sub-agents. "
            "None (default) copies this agent's ``context`` so a 65K "
            "parent does not spawn an 8K child. Set explicitly only "
            "when sub-agents should use a different window or strategy "
            "than the parent. Sub-agents always run in STANDARD mode."
        ),
    )
    n_parallel_attempts: int = Field(
        default=1,
        ge=1,
        le=5,
        description=(
            "F4 — number of independent Best-of-N attempts per run in "
            "autonomous mode. ``1`` (default) runs one attempt with zero "
            "overhead. ``2`` to ``5`` run that many independent Generator "
            "→ Verifier → Reviser loops in parallel (different LLM seeds / "
            "temperatures); the best-scoring PASS / UNCERTAIN candidate is "
            "returned, otherwise the run abstains. Capped at 5 — beyond "
            "that the cost/quality curve flattens (per Aletheia data)."
        ),
    )
    llm_review: bool = Field(
        default=False,
        description=(
            "Enable LLM-based review as validation Layer 3 (autonomous mode). "
            "Off by default — use Critic component for independent verification."
        ),
    )

    # Type-safe LLM parameter overrides for this agent.
    # Accepts LLMParams (base) or any provider subclass (OpenAILLMParams, etc.)
    # These are merged into every llm.call() this agent makes, overriding
    # the LLM-level defaults set in BaseOpenAI.__init__().
    llm_params: LLMParams | None = Field(
        default=None,
        description=(
            "Type-safe LLM call parameter overrides for this agent. "
            "Accepts LLMParams or any provider-specific subclass "
            "(e.g. OpenAILLMParams). Only non-None fields are merged."
        ),
    )

    evidence_gate_required_tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Optional evidence tags the framework checks before package synthesis. "
            "Callers (tasks, rubrics) supply tag names; empty = no gate."
        ),
    )
    evidence_gate_enforce: bool = Field(
        default=False,
        description=(
            "When True, treat missing required evidence tags as a blocked gate. "
            "When False, record coverage and optional gaps without blocking."
        ),
    )
    context_tool_result_corpus_max_chars: int = Field(
        default=500_000,
        ge=0,
        description=(
            "Max characters of each ingested business tool result text indexed into "
            "the run-local document corpus (L5). 0 disables automatic indexing."
        ),
    )
    context_activation_ingest_min_chars: int = Field(
        default=200,
        ge=0,
        description=(
            "Minimum inspected characters for a business tool result to qualify for "
            "L4.5 light ingest (workspace note + corpus index) when the output does "
            "not match strict evidence-shaped heuristics. 0 allows any non-empty text. "
            "Set very high to approximate legacy behavior (promotion/indexing only "
            "when evidence-shaped)."
        ),
    )

    _MODE_TOOL_DEFAULTS: dict = {"direct": 25, "standard": 80, "autonomous": 300}

    @field_validator("evidence_gate_required_tags", mode="before")
    @classmethod
    def _coerce_evidence_gate_tags(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            stripped = value.strip()
            return (stripped,) if stripped else ()
        if isinstance(value, (list, tuple)):
            out: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
            return tuple(out)
        raise TypeError(
            "evidence_gate_required_tags must be None, str, list[str], or tuple[str, ...]"
        )

    @model_validator(mode="after")
    def _validate_output_budget_fits_window(self) -> AgentConfig:
        """Fail at construction when the reply budget cannot fit the window.

        On servers with a shared input/output window (vLLM
        ``--max-model-len``) a request with ``prompt + max_tokens`` above
        the window is rejected with HTTP 400 — typically on tool round
        40, not at build time.  Only values the user set **explicitly**
        are checked here; defaults are derived by the context engine.
        """
        ctx = self.context
        if ctx is None:
            return self
        explicit = getattr(ctx, "model_fields_set", set())
        reserve_explicit = "response_reserve" in explicit
        window = ctx.max_context_tokens
        if reserve_explicit and window is not None and ctx.response_reserve >= window:
            raise ValueError(
                f"ContextConfig.response_reserve ({ctx.response_reserve}) must be "
                f"smaller than max_context_tokens ({window}); nothing would be "
                "left for the prompt."
            )
        if (
            reserve_explicit
            and "llm_max_output_tokens" in self.model_fields_set
            and self.llm_max_output_tokens > ctx.response_reserve
        ):
            raise ValueError(
                f"AgentConfig.llm_max_output_tokens ({self.llm_max_output_tokens}) "
                f"exceeds ContextConfig.response_reserve ({ctx.response_reserve}). "
                "The reply budget sent as max_tokens must fit inside the reserve, "
                "otherwise a full prompt plus the reply overflows the model window "
                f"({window if window is not None else 'auto-detected'}) and the "
                "server rejects the request."
            )
        return self

    def get_effective_max_context_tool_calls(self) -> int:
        """Cap on context-management tool calls per execution.

        Explicit ``max_context_tool_calls`` wins; otherwise twice the
        effective ``max_tool_calls`` so recall traffic is generous but
        finite.
        """
        if self.max_context_tool_calls is not None:
            return max(0, int(self.max_context_tool_calls))
        return 2 * self.get_effective_max_tool_calls()

    @property
    def effective_tracing(self) -> bool:
        """Resolve whether tracing is enabled (observability takes precedence)."""
        if self.observability is not None:
            return self.observability.tracing
        return self.enable_tracing

    @property
    def effective_verbose(self) -> bool:
        """Resolve whether verbose logging is enabled."""
        if self.observability is not None:
            return self.observability.verbose
        return self.verbose

    def get_effective_max_tool_calls(self) -> int:
        """Return the effective tool call limit for the current execution mode.

        If ``max_tool_calls`` is explicitly set, that value is used.
        Otherwise the mode default is returned (DIRECT=25, STANDARD=80,
        AUTONOMOUS=300).
        """
        if self.max_tool_calls is not None:
            return self.max_tool_calls
        mode_val = (
            self.execution_mode.value
            if hasattr(self.execution_mode, "value")
            else str(self.execution_mode)
        )
        return self._MODE_TOOL_DEFAULTS.get(mode_val, 80)


class AgentMetrics(BaseModel):
    """Tracks agent performance metrics."""

    tasks_completed: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_response_time: float = 0.0
    retry_count: int = 0
    error_count: int = 0


class AgentState(str, Enum):
    """Defines the possible states of an agent."""

    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_FOR_TOOLS = "waiting_for_tools"
    WAITING_FOR_HUMAN = "waiting_for_human"
    RETRYING = "retrying"
    BACKOFF = "backoff"
    ERROR = "error"
    COMPLETED = "completed"
