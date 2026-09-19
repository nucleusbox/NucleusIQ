"""
``CriticRunner`` — encapsulates one Critic verification pass.

SRP: takes an agent, a prepared result and message list, drives the
``Critic``'s prompt builder + an LLM call, parses the response into a
``CritiqueResult``, and falls back to PASS on non-fatal Critic errors
(we never want a Critic bug to block the user's task).

Decoupling: does NOT import ``AutonomousMode`` — it only depends on the
``BaseExecutionMode.call_llm`` wrapping for usage-tracker hooks.  The
caller passes the mode instance in.

Design invariant I-10 — *the verifier sees at least what the generator
saw, or knows that it is seeing less.*  The runner therefore:

1. gives the Critic the curated package (budget-sized, item-complete or
   explicitly marked partial) **and** the raw tool trace rehydrated under
   the window-derived per-result cap — the package is a map, the trace
   is the territory the generator actually read;
2. computes whether that combined view is *complete* (every tool result
   shown whole, or every curated item present);
3. when the view is partial, tells the model so in the prompt and
   **downgrades a FAIL to UNCERTAIN** — a verifier that saw less than the
   generator may ask for another pass, but it may not, on its own,
   condemn an answer as ungrounded.  The downgrade is recorded on the
   critique (``original_verdict``) and in the run report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nucleusiq.agents.components.critic import (
    Critic,
    CritiqueResult,
    Verdict,
)
from nucleusiq.agents.usage.usage_tracker import CallPurpose

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent
    from nucleusiq.agents.chat_models import ChatMessage
    from nucleusiq.agents.modes.base_mode import BaseExecutionMode


#: Score floor for a FAIL downgraded to UNCERTAIN.  Below the acceptance
#: threshold (0.7) so the loop still refines, above 0.0 so a single
#: partially-sighted verdict does not read as "worthless" in trajectory
#: comparisons.
_DOWNGRADED_SCORE_FLOOR = 0.4


@dataclass(frozen=True)
class CriticView:
    """What the Critic was shown, and whether it is everything."""

    package_text: str
    package_complete: bool
    package_meta: dict[str, Any]
    raw_trace_included: bool
    raw_trace_complete: bool
    per_tool_cap: int | None
    tool_results: int
    longest_tool_result: int
    #: Material the generator read that is neither the task nor a tool
    #: result (COMPLEX: the sub-agent findings in the synthesis prompt).
    #: Shown to the Critic verbatim, so when present it is always complete.
    inputs_text: str = ""
    inputs_label: str = ""
    inputs_items: int = 0

    @property
    def inputs_included(self) -> bool:
        return bool(self.inputs_text)

    @property
    def complete(self) -> bool:
        """Parity with the generator.

        The Critic sees everything the generator saw when the tool trace is
        shown whole (or there was none) **and** any hand-off material is
        shown — or when the curated package is a complete map of the
        evidence.  A generator that used no tools and received no hand-off
        saw only the task; the Critic has that too.
        """
        territory_complete = (not self.raw_trace_included) or self.raw_trace_complete
        return territory_complete or self.package_complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "package_chars": len(self.package_text),
            "package_complete": self.package_complete,
            "package_omitted": {
                k: v
                for k, v in (self.package_meta.get("omitted_items") or {}).items()
                if v
            },
            "raw_trace_included": self.raw_trace_included,
            "raw_trace_complete": self.raw_trace_complete,
            "per_tool_cap": self.per_tool_cap,
            "tool_results": self.tool_results,
            "longest_tool_result": self.longest_tool_result,
            "inputs_included": self.inputs_included,
            "inputs_chars": len(self.inputs_text),
            "inputs_items": self.inputs_items,
        }

    def inputs_section(self) -> str:
        """Prompt block with the hand-off material, or ``""``."""
        if not self.inputs_text:
            return ""
        label = self.inputs_label or "material handed to the generator"
        return (
            f"## MATERIAL THE GENERATOR WAS GIVEN ({label})\n"
            "This is exactly what the generator read before writing the "
            "answer. Check the answer against it.\n"
            f"{self.inputs_text}"
        )

    def notice(self) -> str:
        """Prompt paragraph telling the model how much it is seeing."""
        if self.complete:
            return ""
        parts: list[str] = []
        if not self.package_complete and self.package_text:
            omitted = {
                k: v
                for k, v in (self.package_meta.get("omitted_items") or {}).items()
                if v
            }
            if omitted:
                parts.append(
                    "the curated package dropped items ("
                    + ", ".join(f"{k}: {v}" for k, v in omitted.items())
                    + ")"
                )
            else:
                parts.append("the curated package is a preview, not the full results")
        if self.raw_trace_included and not self.raw_trace_complete:
            parts.append(
                f"the raw trace shows at most {self.per_tool_cap} chars of each of "
                f"{self.tool_results} tool results (longest is "
                f"{self.longest_tool_result} chars)"
            )
        detail = "; ".join(parts) if parts else "evidence was cut for space"
        return (
            "## EVIDENCE VISIBILITY\n"
            f"You are seeing LESS than the generator saw: {detail}. "
            "Everything listed under 'Resources Processed' WAS read by tools. "
            "A fact you cannot find here may simply be in the part you were "
            "not shown. Do NOT fail the answer for containing content you "
            "cannot verify — use UNCERTAIN and name what you could not check. "
            "FAIL only for an error you can point to in the material shown."
        )


class CriticRunner:
    """Runs the ``Critic`` role against a candidate answer.

    Args:
        mode: The enclosing execution mode (supplies ``call_llm`` for
            usage-tracker-aware LLM invocation).
        critic: The ``Critic`` component (owns prompt/parse logic).
    """

    def __init__(self, mode: BaseExecutionMode, critic: Critic) -> None:
        self._mode = mode
        self._critic = critic

    async def run(
        self,
        agent: Agent,
        task_objective: str,
        result: Any,
        messages: list[ChatMessage],
    ) -> CritiqueResult:
        """Build the verification prompt, call the LLM, parse the result.

        Single LLM call, no tool access — we use the reasoning-verification
        prompt (no "call tools" instruction) to avoid confusing models
        that take instructions literally.  Token budget is the agent's
        configured ``llm_max_output_tokens``.

        On *any* exception the Critic is treated as non-fatal: we log a
        warning and return ``UNCERTAIN`` with score ``0.0`` so the
        orchestrator can retry or abstain instead of falsely passing.
        """
        phase_controller = getattr(agent, "_phase_controller", None)
        if phase_controller is not None:
            phase_controller.enter("VALIDATE")
        try:
            engine = getattr(agent, "_context_engine", None)
            content_store = getattr(engine, "store", None) if engine else None
            view = self._build_view(
                agent, task_objective, messages, content_store=content_store
            )
            per_tool_cap = view.per_tool_cap
            self._record_view(agent, view)

            task_for_critic = task_objective
            notice = view.notice()
            if notice:
                task_for_critic += f"\n\n{notice}"
            if view.package_text:
                task_for_critic += (
                    "\n\n## CURATED SYNTHESIS PACKAGE FOR VERIFICATION\n"
                    f"{view.package_text}"
                )
            inputs_section = view.inputs_section()
            if inputs_section:
                task_for_critic += f"\n\n{inputs_section}"

            contract = self._mode.structured_contract(agent)
            verification_prompt = self._critic.build_verification_prompt(
                task_objective=task_for_critic,
                final_result=result,
                generator_messages=messages if view.raw_trace_included else None,
                allow_tool_instructions=False,
                content_store=content_store,
                per_tool_char_cap=per_tool_cap,
                output_contract=(
                    contract.critic_criterion() if contract is not None else None
                ),
            )

            model_name = getattr(agent.llm, "model_name", "default")
            token_budget = agent.config.llm_max_output_tokens

            call_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": verification_prompt}],
                "max_output_tokens": token_budget,
            }
            call_kwargs.update(getattr(agent, "_current_llm_overrides", {}))

            response = await self._mode.call_llm(
                agent, call_kwargs, purpose=CallPurpose.CRITIC
            )

            text = ""
            if hasattr(response, "choices") and response.choices:
                msg = response.choices[0].message
                text = getattr(msg, "content", "") or ""
            elif isinstance(response, str):
                text = response

            critique = self._critic.parse_result_text(text)
            return self._apply_view(agent, critique, view)

        except Exception as e:
            agent._logger.warning("Critic failed (non-fatal, uncertain verdict): %s", e)
            return CritiqueResult(
                verdict=Verdict.UNCERTAIN,
                score=0.0,
                feedback=f"Critic infrastructure error: {e}",
            )

    # ------------------------------------------------------------------ #
    # I-10 — evidence view                                                 #
    # ------------------------------------------------------------------ #

    def _build_view(
        self,
        agent: Agent,
        task_objective: str,
        messages: list[ChatMessage],
        *,
        content_store: Any = None,
    ) -> CriticView:
        package_text = ""
        package_meta: dict[str, Any] = {}
        package_complete = False
        build_package_messages = getattr(
            agent, "_build_synthesis_messages_from_context", None
        )
        if build_package_messages is not None:
            package_messages = build_package_messages(
                task=task_objective,
                output_shape=(
                    "Use this curated package together with the raw trace "
                    "below it; the package tells you what was read and "
                    "recorded, the trace shows the actual tool output."
                ),
                role="critic_evidence_total",
            )
            if package_messages and package_messages[0].content:
                package_text = str(package_messages[0].content)
            last_pkg = getattr(agent, "_last_synthesis_package", None)
            if last_pkg is not None:
                package_meta = dict(getattr(last_pkg, "metadata", {}) or {})
                # The package is complete as a *map* only when nothing was
                # dropped AND no note preview was cut — a cut preview hides
                # facts the generator saw.
                package_complete = bool(package_meta.get("complete")) and not any(
                    (package_meta.get("cut_items") or {}).values()
                )

        inputs_text = ""
        inputs_label = ""
        inputs_items = 0
        inputs = getattr(agent, "_generator_inputs", None)
        if isinstance(inputs, dict) and str(inputs.get("text") or "").strip():
            inputs_text = str(inputs.get("text"))
            inputs_label = str(inputs.get("label") or "")
            inputs_items = int(inputs.get("items") or 0)

        # The per-result cap must leave room for the package and the
        # hand-off material, which sit in the same prompt; otherwise a large
        # trace plus a full package can exceed the window the cap was
        # derived from.
        per_tool_cap = _compute_critic_per_tool_cap(
            agent,
            messages,
            extra_prompt_chars=len(package_text) + len(inputs_text),
        )

        # Measure the *rehydrated* results — a masked or offloaded receipt
        # is a few hundred chars, the payload behind it may be 20K.
        trace_msgs = messages
        if content_store is not None:
            from nucleusiq.agents.context.store import extract_raw_trace

            trace_msgs = extract_raw_trace(
                messages, content_store, max_chars_per_result=10**9
            )
        tool_lengths = [
            len(str(getattr(m, "content", "") or ""))
            for m in trace_msgs
            if getattr(m, "role", None) == "tool"
        ]
        longest = max(tool_lengths, default=0)
        limits = getattr(self._critic, "_limits", None)
        trace_lines_cap = getattr(limits, "trace_lines", 0) or 0
        # Mirror ``Critic._extract_reasoning_trace``: one line per tool
        # result, per assistant text, and per tool call.
        trace_lines = 0
        for m in messages:
            role = getattr(m, "role", None)
            if role == "tool":
                trace_lines += 1
            elif role == "assistant":
                if str(getattr(m, "content", "") or "").strip():
                    trace_lines += 1
                trace_lines += len(getattr(m, "tool_calls", None) or [])
        raw_included = bool(tool_lengths)
        effective_cap = (
            per_tool_cap
            if per_tool_cap is not None
            else int(getattr(limits, "tool_result", 0) or 0)
        )
        raw_complete = raw_included and (
            (effective_cap <= 0 or longest <= effective_cap)
            and (trace_lines_cap <= 0 or trace_lines <= trace_lines_cap)
        )

        activator = getattr(agent, "_context_state_activator", None)
        phase_controller = getattr(agent, "_phase_controller", None)
        if package_text:
            if activator is not None:
                activator.metrics.critic_used_package = True
            if phase_controller is not None:
                phase_controller.critic_used_package = True
        if raw_included and activator is not None and not package_complete:
            activator.metrics.raw_trace_fallback_used = True

        return CriticView(
            package_text=package_text,
            package_complete=package_complete,
            package_meta=package_meta,
            raw_trace_included=raw_included,
            raw_trace_complete=raw_complete,
            per_tool_cap=per_tool_cap,
            tool_results=len(tool_lengths),
            longest_tool_result=longest,
            inputs_text=inputs_text,
            inputs_label=inputs_label,
            inputs_items=inputs_items,
        )

    @staticmethod
    def _record_view(agent: Agent, view: CriticView) -> None:
        recorder = getattr(agent, "_run_recorder", None)
        if recorder is None:
            return
        try:
            views = list(recorder.decisions.get("critic_views") or [])
            views.append(view.to_dict())
            recorder.record_decision("critic_views", views)
            if not view.complete:
                recorder.counters.critic_partial_views += 1
                recorder.record_event(
                    "critic_partial_view",
                    f"package_complete={view.package_complete} "
                    f"raw_complete={view.raw_trace_complete} "
                    f"per_tool_cap={view.per_tool_cap} longest={view.longest_tool_result}",
                )
        except Exception:
            pass

    @staticmethod
    def _apply_view(
        agent: Agent, critique: CritiqueResult, view: CriticView
    ) -> CritiqueResult:
        """Stamp the view on the critique; downgrade FAIL when it was partial."""
        update: dict[str, Any] = {
            "evidence_view": "complete" if view.complete else "partial",
        }
        if not view.complete and critique.verdict == Verdict.FAIL:
            update.update(
                verdict=Verdict.UNCERTAIN,
                original_verdict=Verdict.FAIL,
                score=max(critique.score, _DOWNGRADED_SCORE_FLOOR),
                feedback=(
                    "[Verifier saw PARTIAL evidence — FAIL downgraded to UNCERTAIN; "
                    "treat the issues below as points to re-check, not as proven "
                    f"errors] {critique.feedback}"
                ).strip(),
            )
            agent._logger.info(
                "Critic FAIL downgraded to UNCERTAIN: verifier saw partial evidence "
                "(package_complete=%s, raw_complete=%s, per_tool_cap=%s)",
                view.package_complete,
                view.raw_trace_complete,
                view.per_tool_cap,
            )
            recorder = getattr(agent, "_run_recorder", None)
            if recorder is not None:
                try:
                    recorder.counters.critic_fail_downgraded += 1
                    recorder.record_event(
                        "critic_fail_downgraded",
                        (critique.feedback or "")[:160],
                    )
                except Exception:
                    pass
        return critique.model_copy(update=update)


def _compute_critic_per_tool_cap(
    agent: Agent,
    messages: list[ChatMessage],
    *,
    extra_prompt_chars: int = 0,
) -> int | None:
    """Resolve the adaptive per-tool-result char cap for the Critic.

    v0.7.8 — the Critic's rehydrated tool-result trace is capped by a
    **runtime-computed** budget, not the legacy fixed
    ``CriticLimits.tool_result`` (3K/5K chars).  The budget shrinks as
    the trace grows and scales with the LLM's actual context window,
    so the same framework works for 32K open-source models and 2M
    Gemini without any model-specific tuning.

    Returns ``None`` when the agent has no context config or no LLM,
    in which case the Critic falls back to the legacy fixed limits
    for full backward compatibility.
    """
    cfg = getattr(getattr(agent, "config", None), "context", None)
    if cfg is None or agent.llm is None:
        return None

    from nucleusiq.agents.context.budgets import budgets_for
    from nucleusiq.agents.context.store import compute_per_tool_cap

    # The *resolved* window (explicit ContextConfig → engine → provider →
    # fallback), never the provider default alone.  A provider that reports
    # a flat 8K default while the user configured 65K used to starve the
    # Critic to 500 chars per result — it then "saw" three of nine invoices,
    # failed a correct answer as ungrounded, and the Refiner deleted the
    # other six.
    budgets = budgets_for(agent)
    context_window = budgets.window
    num_tool_results = sum(1 for m in messages if getattr(m, "role", None) == "tool")
    extra_tokens = max(0, extra_prompt_chars) // max(1, budgets.chars_per_token)

    return compute_per_tool_cap(
        context_window=context_window,
        prompt_overhead_tokens=cfg.critic_prompt_overhead_tokens + extra_tokens,
        response_reserve_tokens=cfg.critic_response_reserve_tokens,
        num_tool_results=num_tool_results,
        min_chars=cfg.tool_result_per_call_min_chars,
        max_chars=cfg.tool_result_per_call_max_chars,
        purpose="critic",
    )


__all__ = ["CriticRunner", "CriticView"]
