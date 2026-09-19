"""BudgetResolver — every hand-off cap derives from the one real number.

Design (docs/design/AUTONOMOUS_HARNESS_HARDENING.md, WS-1 / I-3).

Before this module the harness carried a dozen independent constants
(``[:2000]`` per sub-agent finding, ``8_000`` / ``4_000`` in the Refiner,
``[:2000]`` in LLM review, a ``2_000``-char gap summary, …).  They were
tuned for 128K cloud models and silently wrong everywhere else: on an
8K model they overflowed, on a 65K model with three children they threw
away most of each child's work while a 37K-token prompt budget sat idle.

:class:`BudgetResolver` is owned by :class:`~nucleusiq.agents.context.engine.ContextEngine`
(``engine.budgets``) and knows the window, the compaction working
budget, the resolved response reserve and the fixed per-call costs
(system prompt, tool schemas).  Roles ask it for a character cap::

    resolver.handoff_chars("synthesis_finding", items=len(findings))
    resolver.handoff_chars("refiner_candidate")

Each role has an overhead (tokens the role's own prompt framing and
reply need), a *share* of what is left, a floor (below which the
content is useless) and a ceiling (a safety rail against a single
pathological payload).  On a 128K model the ceilings dominate, so
today's behaviour is the *upper* bound; on smaller windows the caps
shrink in proportion instead of overflowing.

Everything here is pure integer math with no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nucleusiq.agents.context.config import ContextConfig

#: ~4 characters per token for English prose / JSON.  Matches
#: ``DefaultTokenCounter`` and ``context.store._CHARS_PER_TOKEN``.
CHARS_PER_TOKEN = 4

HandoffRole = Literal[
    "task_context",
    "synthesis_finding",
    "validation_result",
    "validation_task",
    "refiner_candidate",
    "refiner_tool_summary",
    "gap_summary",
    "critic_claimed_answer",
    "critic_evidence_total",
    "synthesis_package",
]


@dataclass(frozen=True)
class RoleSpec:
    """How one hand-off role carves its share of the prompt budget."""

    #: Tokens the role's own framing + reply need before any hand-off
    #: content.  Names refer to ``ContextConfig`` fields when they exist.
    overhead_tokens: int
    #: Fraction of the remaining prompt budget this payload may take.
    share: float
    #: Never go below this many characters (content becomes useless).
    floor_chars: int
    #: Never go above this many characters (safety rail).
    ceiling_chars: int


def _default_role_specs(config: ContextConfig | None) -> dict[str, RoleSpec]:
    critic_overhead = 13_000
    refiner_overhead = 24_000
    if config is not None:
        critic_overhead = int(
            getattr(config, "critic_prompt_overhead_tokens", 5_000)
            + getattr(config, "critic_response_reserve_tokens", 8_000)
        )
        refiner_overhead = int(
            getattr(config, "refiner_prompt_overhead_tokens", 8_000)
            + getattr(config, "refiner_response_reserve_tokens", 16_000)
        )
    return {
        # ``Task.context`` / ``Task.resources`` block ahead of the objective.
        "task_context": RoleSpec(2_000, 0.25, 1_000, 24_000),
        # COMPLEX synthesis: each child's result, shared across children.
        "synthesis_finding": RoleSpec(4_000, 1.0, 1_000, 40_000),
        # Optional L3 LLM review (task + candidate).
        "validation_result": RoleSpec(1_200, 0.8, 1_000, 20_000),
        "validation_task": RoleSpec(1_200, 0.2, 500, 4_000),
        # Refiner revision prompt: prior candidate + bounded tool summary.
        "refiner_candidate": RoleSpec(refiner_overhead, 0.5, 2_000, 32_000),
        "refiner_tool_summary": RoleSpec(refiner_overhead, 0.25, 1_000, 16_000),
        "gap_summary": RoleSpec(refiner_overhead, 0.1, 500, 4_000),
        # Critic: clamp-down only — presets stay the upper bound.
        "critic_claimed_answer": RoleSpec(critic_overhead, 0.5, 2_000, 50_000),
        "critic_evidence_total": RoleSpec(critic_overhead, 0.5, 2_000, 40_000),
        # Curated synthesis package fed to the tools=None synthesis pass.
        # The generator's own reply is the only other thing in that prompt,
        # so the package may take most of the budget.
        "synthesis_package": RoleSpec(3_000, 0.6, 4_000, 120_000),
    }


@dataclass(frozen=True)
class BudgetResolver:
    """Window-derived budgets for every role in the harness."""

    #: Model context window (hard ceiling) in tokens.
    window: int
    #: Compaction working budget (``optimal_budget``) in tokens.
    working_budget: int
    #: Tokens kept free for the model's reply.
    response_reserve: int
    #: Fixed per-call prompt costs measured at preflight.
    system_tokens: int = 0
    tool_schema_tokens: int = 0
    chars_per_token: int = CHARS_PER_TOKEN
    role_specs: dict[str, RoleSpec] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        config: ContextConfig | None,
        *,
        window: int,
        working_budget: int,
        response_reserve: int,
        system_tokens: int = 0,
        tool_schema_tokens: int = 0,
    ) -> BudgetResolver:
        return cls(
            window=max(0, int(window)),
            working_budget=max(0, int(working_budget)),
            response_reserve=max(0, int(response_reserve)),
            system_tokens=max(0, int(system_tokens)),
            tool_schema_tokens=max(0, int(tool_schema_tokens)),
            role_specs=_default_role_specs(config),
        )

    def with_fixed_costs(
        self, *, system_tokens: int, tool_schema_tokens: int
    ) -> BudgetResolver:
        return BudgetResolver(
            window=self.window,
            working_budget=self.working_budget,
            response_reserve=self.response_reserve,
            system_tokens=max(0, int(system_tokens)),
            tool_schema_tokens=max(0, int(tool_schema_tokens)),
            chars_per_token=self.chars_per_token,
            role_specs=self.role_specs,
        )

    # ------------------------------------------------------------------ #
    # Derived numbers                                                      #
    # ------------------------------------------------------------------ #

    @property
    def prompt_budget(self) -> int:
        """Tokens the prompt may occupy before compaction kicks in."""
        return max(0, min(self.working_budget, self.window) - self.response_reserve)

    @property
    def usable_tokens(self) -> int:
        """Preflight "working" tokens: window minus reply, system and tool schemas.

        This is the number the fitness check compares against the 16K /
        32K floors — what the model can actually spend on the task's
        evidence and reasoning per call.
        """
        return max(
            0,
            self.window
            - self.response_reserve
            - self.system_tokens
            - self.tool_schema_tokens,
        )

    def handoff_chars(self, role: HandoffRole | str, *, items: int = 1) -> int:
        """Characters one payload of ``role`` may take (``items`` share it)."""
        spec = self.role_specs.get(role)
        if spec is None:
            spec = _default_role_specs(None).get(role)
        if spec is None:
            raise KeyError(f"Unknown hand-off role: {role!r}")
        available_tokens = max(0, self.prompt_budget - spec.overhead_tokens)
        per_item_tokens = (available_tokens * spec.share) / max(1, int(items))
        chars = int(per_item_tokens * self.chars_per_token)
        return max(spec.floor_chars, min(spec.ceiling_chars, chars))

    def clamp(self, role: HandoffRole | str, current: int, *, items: int = 1) -> int:
        """``min(current, handoff_chars(role))`` — clamp-down only."""
        return min(int(current), self.handoff_chars(role, items=items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "working_budget": self.working_budget,
            "response_reserve": self.response_reserve,
            "system_tokens": self.system_tokens,
            "tool_schema_tokens": self.tool_schema_tokens,
            "prompt_budget": self.prompt_budget,
            "usable_tokens": self.usable_tokens,
        }


#: Window assumed when neither the user nor the provider declares one.
#: Mirrors ``BaseLLM.get_context_window()``'s default; the agent logs a
#: warning when this fallback is used so the report can say so.
FALLBACK_WINDOW = 128_000


def budgets_for(agent: Any) -> BudgetResolver:
    """Resolve budgets for ``agent`` — engine-backed when one exists.

    Without a context engine (``ContextStrategy.NONE`` or bare test
    doubles) a resolver is built from the LLM's declared window and the
    agent's ``ContextConfig`` (or defaults), so call sites never need a
    ``None`` branch.
    """
    engine = getattr(agent, "_context_engine", None)
    budgets = getattr(engine, "budgets", None)
    if isinstance(budgets, BudgetResolver):
        return budgets

    from nucleusiq.agents.context.config import ContextConfig

    config = getattr(getattr(agent, "config", None), "context", None)
    if not isinstance(config, ContextConfig):
        config = ContextConfig()
    window: int | None = config.max_context_tokens
    llm = getattr(agent, "llm", None)
    if window is None and llm is not None:
        try:
            raw = llm.get_context_window()
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                window = int(raw)
            else:
                # Test doubles may hand back a coroutine; never leave it pending.
                close = getattr(raw, "close", None)
                if callable(close):
                    close()
                window = None
        except Exception:
            window = None
    window = int(window or FALLBACK_WINDOW)
    max_out = getattr(getattr(agent, "config", None), "llm_max_output_tokens", None)
    reserve = ContextConfig.resolve_response_reserve(
        config, window, max_out if isinstance(max_out, int) else None
    )
    return BudgetResolver.from_config(
        config,
        window=window,
        working_budget=ContextConfig.resolve_optimal_budget(config, window),
        response_reserve=reserve,
    )


__all__ = [
    "CHARS_PER_TOKEN",
    "FALLBACK_WINDOW",
    "BudgetResolver",
    "HandoffRole",
    "RoleSpec",
    "budgets_for",
]
