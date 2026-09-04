"""Preflight validation for a configured endpoint.

Layer-1 validation (offline argument checking) lives in
:mod:`~nucleusiq_openai_compatible.config` and runs unconditionally at
construction.  This module implements **Layer 2**: the network checks, which
are explicit and opt-in because a constructor should not perform I/O.

    report = await llm.validate()
    if not report.ok:
        print(report.render())

The value is in the failure messages.  A wrong ``model=`` normally surfaces
as a bare ``404`` somewhere inside an agent run; here it becomes *"server
serves: gemma-4-27b-it, qwen3-32b"*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ResolvedConfig

__all__ = ["ValidationReport", "build_report"]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Structured outcome of :meth:`OpenAICompatibleLLM.validate`."""

    ok: bool
    reachable: bool
    model_found: bool
    served_models: tuple[str, ...]
    context_window: int
    context_window_source: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    config: dict[str, object] = field(default_factory=dict)

    def render(self) -> str:
        """Return a human-readable multi-line summary."""
        status = "OK" if self.ok else "FAILED"
        lines = [
            f"OpenAI-compatible endpoint validation: {status}",
            f"  reachable          : {self.reachable}",
            f"  model found        : {self.model_found}",
            f"  context window     : {self.context_window} (source: {self.context_window_source})",
        ]
        if self.served_models:
            lines.append(f"  served models      : {', '.join(self.served_models)}")
        for label, items in (
            ("error", self.errors),
            ("warning", self.warnings),
            ("note", self.notes),
        ):
            for item in items:
                lines.append(f"  {label:<18}: {item}")
        return "\n".join(lines)


def build_report(
    *,
    config: ResolvedConfig,
    reachable: bool,
    served_models: tuple[str, ...],
    probe_error: str | None,
    model_found: bool,
) -> ValidationReport:
    """Assemble a :class:`ValidationReport` from probe results and config."""
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    if not reachable:
        errors.append(
            f"Could not list models at {config.base_url}/models"
            + (f": {probe_error}" if probe_error else "")
            + ". Check that the server is running and base_url is correct "
            "(it must include the /v1 prefix)."
        )
    elif not model_found:
        served = ", ".join(served_models) if served_models else "(none reported)"
        errors.append(
            f"Model {config.model!r} is not served by {config.base_url}. "
            f"This server serves: {served}."
        )

    if config.context_window_source == "default":
        warnings.append(
            f"Context window fell back to {config.context_window} tokens "
            "because it was neither declared nor discoverable. Pass "
            "context_window=<tokens> so the context engine budgets against "
            "the real limit."
        )

    if not config.supports_tools:
        warnings.append(
            f"Engine {config.engine!r} is configured without tool support; "
            "agents that rely on tools will not work."
        )

    if config.structured_output_suppresses_tools and config.supports_tools:
        notes.append(
            "On this engine, sending response_format together with tools "
            "suppresses tool calls; policy "
            f"structured_output_with_tools={config.structured_output_with_tools!r} "
            "is in effect."
        )

    if config.is_reasoning_model and not config.supports_reasoning:
        warnings.append(
            f"is_reasoning_model=True but engine {config.engine!r} cannot "
            "separate thinking from the answer, so thinking text will appear "
            "inline in the content. Start vLLM/SGLang with "
            "--reasoning-parser <parser>."
        )
    elif config.supports_reasoning and not config.is_reasoning_model:
        notes.append(
            "If this model thinks before answering, pass "
            "is_reasoning_model=True so internal agent calls get a token "
            "budget covering both thinking and the answer."
        )

    if config.is_reasoning_model and not config.chat_template_kwargs:
        notes.append(
            "No chat_template_kwargs set. Thinking is off by default for "
            "Gemma, Granite and DeepSeek-V3.1 (pass "
            "{'enable_thinking': True} or {'thinking': True}) and on by "
            "default for Qwen3 (pass {'enable_thinking': False} to disable)."
        )

    if config.token_count_method == "heuristic":
        notes.append(
            "Token counts use the ~4 chars/token heuristic. Pass "
            "tokenizer='<hf-repo-id>' with the [tokenizer] extra for exact "
            "budgeting."
        )

    if config.engine_notes:
        notes.append(config.engine_notes)

    return ValidationReport(
        ok=not errors,
        reachable=reachable,
        model_found=model_found,
        served_models=served_models,
        context_window=config.context_window,
        context_window_source=config.context_window_source,
        errors=tuple(errors),
        warnings=tuple(warnings),
        notes=tuple(notes),
        config=config.summary(),
    )
