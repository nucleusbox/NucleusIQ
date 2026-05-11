# src/nucleusiq/agents/structured_output/resolver.py
"""
Resolver for NucleusIQ Structured Output.

Resolves ``OutputMode.AUTO``, builds provider hints for ``OutputSchema.for_provider``,
and exposes optional model-prefix helpers (:func:`supports_native_output`).
"""

from __future__ import annotations

from typing import Any

from .config import OutputSchema
from .errors import StructuredOutputError
from .types import OutputMode


def supports_native_output(model_name: str, provider: str | None = None) -> bool:
    """
    Best-effort hint whether *native* structured output is plausible for this pair.

    There is **no** maintainable global model ID list (OpenAI, Google, Groq, and
    Ollama add or rename models often). Strategy:

    * If *provider* is set (as from :func:`get_provider_from_llm`), use
      **provider-specific** rules: Groq and Ollama NucleusIQ adapters always wire
      ``response_format`` into the vendor API where supported; OpenAI / Anthropic /
      Google use coarse **name shape** checks so we do not claim e.g. ``gpt-4o`` on
      an Anthropic-backed stack.
    * If *provider* is unknown, use only **weak** name substrings (GPT, Claude,
      Gemini) — custom or local models return ``False``.

    This does **not** drive :func:`_auto_select_mode`; see that function for
    ``OutputMode.AUTO`` behavior.
    """
    m = (model_name or "").strip().lower()
    if not m:
        return False

    if provider:
        p = provider.lower()
        if p == "openai":
            return m.startswith(
                ("gpt-3", "gpt-4", "gpt-5", "o1", "o3", "o4", "chatgpt")
            )
        if p == "anthropic":
            return m.startswith("claude")
        if p == "google":
            return "gemini" in m or m.startswith("gemini")
        if p in ("groq", "ollama"):
            # Adapters map Agent response_format to vendor structured output;
            # model-level eligibility is enforced by the API, not this helper.
            return True

    if m.startswith(("gpt-3", "gpt-4", "gpt-5", "o1", "o3", "o4")):
        return True
    if m.startswith("claude"):
        return True
    if "gemini" in m:
        return True
    return False


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

        model_name: Model name for capability detection
        provider: Provider name for optimization

    Returns:
        OutputSchema configuration or None

    Example:
        # From Agent._run_direct()
        config = resolve_output_config(
            self.response_format,
            model_name=self.llm.model_name,
            provider="openai"
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
        config._resolved_mode = _auto_select_mode(
            model_name=model_name,
            provider=provider,
        )
    else:
        config._resolved_mode = config.mode

    return config


def _auto_select_mode(
    model_name: str | None = None,
    provider: str | None = None,
) -> OutputMode:
    """
    Resolve ``OutputMode.AUTO`` to a concrete mode.

    Today: any configured ``model_name`` maps to **NATIVE** (all first-party LLM
    adapters wire ``response_format`` / provider ``format``). Without a model
    name, fall back to **PROMPT**.

    ``supports_native_output`` is **not** used here; see that function’s docstring.
    It is a separate hint for tests or out-of-band callers.
    """
    _ = provider  # Reserved for future capability-aware AUTO (TOOL/PROMPT).
    if not model_name:
        return OutputMode.PROMPT
    return OutputMode.NATIVE


def get_provider_from_llm(llm: Any) -> str | None:
    """
    Detect provider from LLM instance.

    Args:
        llm: LLM instance

    Returns:
        Provider name or None
    """
    if llm is None:
        return None

    class_name = type(llm).__name__.lower()

    if "openai" in class_name:
        return "openai"
    if "anthropic" in class_name or "claude" in class_name:
        return "anthropic"
    if "google" in class_name or "gemini" in class_name:
        return "google"
    if "ollama" in class_name:
        return "ollama"
    if "groq" in class_name:
        return "groq"

    return None
