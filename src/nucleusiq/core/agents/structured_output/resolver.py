# src/nucleusiq/agents/structured_output/resolver.py
"""
Resolver for NucleusIQ Structured Output.

Answers two questions for the structured-output pipeline: which provider is
behind an LLM adapter (so ``OutputSchema.for_provider`` picks the right wire
format), and what ``OutputMode.AUTO`` resolves to.

Identity comes from what the adapter **declares about itself** —
:attr:`BaseLLM.PROVIDER_NAME`.  Matching on model names is deliberately absent:
vendors add and rename models continuously, so any list of model IDs kept here
would go stale within weeks, and it would fail *silently* by mode-switching a
model the framework simply had not heard of yet.

Whether a given backend can enforce a schema server-side is deliberately *not*
consulted here; see :func:`_auto_select_mode` for why that belongs to the
adapter.
"""

from __future__ import annotations

from typing import Any

from .config import OutputSchema
from .errors import StructuredOutputError
from .types import OutputMode

_CLASS_NAME_HINTS: tuple[tuple[str, str], ...] = (
    # "compatible" precedes "openai": OpenAICompatibleLLM contains both, and
    # an OpenAI-compatible server is not OpenAI.
    ("compatible", "openai_compatible"),
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("claude", "anthropic"),
    ("gemini", "google"),
    ("google", "google"),
    ("ollama", "ollama"),
    ("groq", "groq"),
)


def _provider_from_class_name(llm: Any) -> str | None:
    """Guess the provider from the adapter's class name.

    A compatibility shim for adapters written before ``PROVIDER_NAME`` existed,
    and it is only ever a guess: a third-party subclass named
    ``MyOpenAIWrapper`` is indistinguishable here from the real OpenAI adapter.
    Every first-party adapter declares ``PROVIDER_NAME`` and never reaches this
    path.
    """
    class_name = type(llm).__name__.lower()
    for needle, provider in _CLASS_NAME_HINTS:
        if needle in class_name:
            return provider
    return None


def resolve_output_config(
    response_format: Any,
    *,
    model_name: str | None = None,
    provider: str | None = None,
) -> OutputSchema | None:
    """
    Resolve response_format to an OutputSchema configuration.

    This is the main entry point for handling response_format.
    It converts various input types to a standardized OutputSchema.

    Args:
        response_format: User-provided response format, can be:
            - OutputSchema: Already configured, return as-is
            - Type (Pydantic, dataclass, TypedDict): Wrap in OutputSchema
            - Dict: JSON Schema, wrap in OutputSchema
            - None: No structured output

        model_name: Model name; its presence signals a configured LLM
        provider: Provider name, used by callers to pick a wire format

    Returns:
        OutputSchema configuration or None

    Example:
        # From Agent._run_direct()
        config = resolve_output_config(
            self.response_format,
            model_name=self.llm.model_name,
            provider="openai",
        )
    """
    if response_format is None:
        return None

    # Already an OutputSchema
    if isinstance(response_format, OutputSchema):
        config = response_format

    # Schema type (Pydantic, dataclass, TypedDict, dict)
    elif isinstance(response_format, (type, dict)):
        config = OutputSchema(schema=response_format)

    else:
        raise StructuredOutputError(
            f"Invalid response_format type: {type(response_format)}. "
            "Expected OutputSchema, Pydantic model, dataclass, TypedDict, or dict.",
            retryable=False,
        )

    # Resolve AUTO mode to concrete mode
    if config.mode == OutputMode.AUTO:
        config._resolved_mode = _auto_select_mode(model_name=model_name)
    else:
        config._resolved_mode = config.mode

    return config


def _auto_select_mode(model_name: str | None = None) -> OutputMode:
    """
    Resolve ``OutputMode.AUTO`` to a concrete mode.

    Any configured ``model_name`` maps to **NATIVE**; without one there is no
    adapter to hand a schema to, so it falls back to **PROMPT**.

    **NATIVE** means "hand the schema to the adapter", *not* "the server
    implements ``response_format={"type": "json_schema"}``.  Those are
    different claims, and conflating them is a trap worth naming: an adapter
    knows things core cannot, so it owns the degradation.  The
    OpenAI-compatible provider, for instance, drops to ``json_object`` plus a
    prompt-injected schema when its engine lacks schema support, which
    preserves structured output. Re-routing such a case to ``OutputMode.PROMPT``
    here would instead raise, because :meth:`OutputMode.implemented_modes`
    covers only ``AUTO`` and ``NATIVE`` — so mode selection deliberately does
    **not** consult adapter capabilities.
    """
    if not model_name:
        return OutputMode.PROMPT
    return OutputMode.NATIVE


def get_provider_from_llm(llm: Any) -> str | None:
    """
    Identify the provider behind an LLM adapter.

    Prefers the adapter's own :attr:`BaseLLM.PROVIDER_NAME` declaration.  Falls
    back to class-name matching only for adapters that predate the declaration
    or come from outside this repo.

    Args:
        llm: LLM instance

    Returns:
        Provider name or None
    """
    if llm is None:
        return None

    declared = getattr(llm, "PROVIDER_NAME", None)
    if isinstance(declared, str) and declared.strip():
        return declared.strip().lower()

    return _provider_from_class_name(llm)
