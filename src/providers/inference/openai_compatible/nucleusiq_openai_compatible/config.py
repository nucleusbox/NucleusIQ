"""Configuration resolution — one precedence chain, one immutable result.

This module is the reason the provider needs no user-facing ``ModelProfile``.
The constructor takes flat keyword arguments; :class:`ConfigResolver` folds
them together with the engine preset and (optionally) a live ``/v1/models``
probe into a single frozen :class:`ResolvedConfig` that the rest of the
package reads from.

The chain, applied per field and documented in one place so no caller has to
guess which source won::

    explicit constructor arg  ->  /v1/models probe  ->  EngineProfile  ->  default

``ResolvedConfig`` is exposed read-only as ``llm.capabilities`` for
introspection and telemetry, and records *where* the context window came
from, which is the single most common source of budgeting confusion on
self-hosted deployments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from nucleusiq.llms.errors import InvalidRequestError

from .capabilities import (
    DEFAULT_CONTEXT_WINDOW,
    MAX_TOKENS_FIELDS,
    EngineProfile,
    get_engine_profile,
)

__all__ = [
    "MAX_REASONABLE_CONTEXT",
    "MIN_REASONABLE_CONTEXT",
    "ConfigResolver",
    "ContextWindowSource",
    "ResolvedConfig",
    "StructuredOutputMode",
    "normalize_base_url",
]

_logger = logging.getLogger(__name__)
_PROVIDER = "openai_compatible"

ContextWindowSource = Literal["explicit", "probe", "engine", "default"]
StructuredOutputMode = Literal["prompt", "drop", "error"]
TokenCountMethod = Literal["tokenizer", "heuristic"]

MIN_REASONABLE_CONTEXT = 256
"""Below this a context window is certainly a mistake (e.g. a value in K)."""

MAX_REASONABLE_CONTEXT = 100_000_000
"""Above this a context window is certainly a mistake (e.g. bytes, not tokens)."""

_STRUCTURED_OUTPUT_MODES = ("prompt", "drop", "error")


def normalize_base_url(base_url: str) -> str:
    """Validate and normalize an OpenAI-compatible base URL.

    Appends the ``/v1`` path segment when omitted — the single most common
    configuration slip, since ``http://host:8000`` looks complete but every
    OpenAI-compatible route lives under ``/v1``.

    Args:
        base_url: A URL such as ``http://gpu-node-1:8000`` or
            ``https://resource.openai.azure.com/openai/v1``.

    Returns:
        The normalized URL, without a trailing slash.

    Raises:
        InvalidRequestError: The value is empty, unparseable, has a scheme
            other than http/https, or carries no host.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=(
                "base_url is required. Pass base_url='http://<host>:<port>/v1' "
                "or set OPENAI_COMPATIBLE_BASE_URL."
            ),
        )

    raw = base_url.strip().rstrip("/")
    parsed = urlparse(raw)

    if parsed.scheme not in ("http", "https"):
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=(
                f"base_url must start with http:// or https://, got {base_url!r}. "
                "Example: base_url='http://gpu-node-1:8000/v1'"
            ),
        )
    if not parsed.netloc:
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=(
                f"base_url is missing a host: {base_url!r}. "
                "Example: base_url='http://gpu-node-1:8000/v1'"
            ),
        )

    path = parsed.path
    if not path or path == "/":
        path = "/v1"
    elif not any(seg == "v1" for seg in path.split("/")):
        path = f"{path}/v1"

    return urlunparse(
        (parsed.scheme, parsed.netloc, path, "", parsed.query, "")
    ).rstrip("/")


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Immutable, fully-resolved capability record for one model+endpoint.

    Exposed as ``llm.capabilities``.  Every consumer in the package reads
    capability facts from here rather than from scattered attributes, so
    there is exactly one place where "what can this server do" is decided.
    """

    base_url: str
    model: str
    engine: str

    context_window: int
    context_window_source: ContextWindowSource
    max_output_tokens: int | None

    supports_tools: bool
    supports_json_schema: bool
    supports_parallel_tool_calls: bool
    supports_stream_usage: bool
    structured_output_suppresses_tools: bool
    extra_body_allowed: bool

    max_tokens_field: str
    structured_output_with_tools: StructuredOutputMode
    strict_capabilities: bool
    token_count_method: TokenCountMethod
    tokenizer: str | None
    engine_notes: str

    supports_reasoning: bool = False
    supports_reasoning_effort: bool = False
    is_reasoning_model: bool = False
    chat_template_kwargs: dict[str, Any] | None = None

    def with_context_window(
        self, window: int, source: ContextWindowSource
    ) -> ResolvedConfig:
        """Return a copy carrying a newly discovered context window.

        Used after a lazy ``/v1/models`` probe, which cannot run during
        ``__init__`` because it needs the event loop.
        """
        return replace(self, context_window=window, context_window_source=source)

    def summary(self) -> dict[str, object]:
        """Return a JSON-friendly summary for logs and validation reports."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "engine": self.engine,
            "context_window": self.context_window,
            "context_window_source": self.context_window_source,
            "supports_tools": self.supports_tools,
            "supports_json_schema": self.supports_json_schema,
            "supports_reasoning": self.supports_reasoning,
            "is_reasoning_model": self.is_reasoning_model,
            "token_count_method": self.token_count_method,
        }


class ConfigResolver:
    """Builds a :class:`ResolvedConfig` from constructor arguments.

    Pure with respect to I/O — no network, no environment reads — so the
    whole precedence chain is unit-testable without fakes.  The probe result
    is passed in by the caller rather than fetched here, keeping this class
    free of transport concerns.
    """

    @staticmethod
    def resolve(
        *,
        base_url: str,
        model: str,
        engine: str = "generic",
        context_window: int | None = None,
        probed_context_window: int | None = None,
        max_output_tokens: int | None = None,
        supports_tools: bool | None = None,
        supports_json_schema: bool | None = None,
        supports_parallel_tool_calls: bool | None = None,
        max_tokens_field: str | None = None,
        structured_output_with_tools: str = "prompt",
        strict_capabilities: bool = False,
        tokenizer: str | None = None,
        has_tokenizer_backend: bool = False,
        is_reasoning_model: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> ResolvedConfig:
        """Fold every configuration source into one immutable record.

        Raises:
            InvalidRequestError: Any Layer-1 validation rule fails (see the
                provider design doc §9).  Messages name the offending value
                and the corrective action.
        """
        profile: EngineProfile = _resolve_engine(engine)
        normalized_url = normalize_base_url(base_url)
        resolved_model = _validate_model(model)

        window, source = ConfigResolver._resolve_window(
            explicit=context_window,
            probed=probed_context_window,
            profile=profile,
            model=resolved_model,
        )

        mode = _validate_structured_output_mode(structured_output_with_tools)
        tokens_field = _validate_max_tokens_field(
            max_tokens_field or profile.max_tokens_field
        )
        _validate_max_output_tokens(max_output_tokens, window)

        method: TokenCountMethod = (
            "tokenizer" if tokenizer and has_tokenizer_backend else "heuristic"
        )

        return ResolvedConfig(
            base_url=normalized_url,
            model=resolved_model,
            engine=profile.name,
            context_window=window,
            context_window_source=source,
            max_output_tokens=max_output_tokens,
            supports_tools=_pick(supports_tools, profile.supports_tools),
            supports_json_schema=_pick(
                supports_json_schema, profile.supports_json_schema
            ),
            supports_parallel_tool_calls=_pick(
                supports_parallel_tool_calls, profile.supports_parallel_tool_calls
            ),
            supports_stream_usage=profile.supports_stream_usage,
            structured_output_suppresses_tools=profile.structured_output_suppresses_tools,
            extra_body_allowed=profile.extra_body_allowed,
            max_tokens_field=tokens_field,
            structured_output_with_tools=mode,
            strict_capabilities=strict_capabilities,
            token_count_method=method,
            tokenizer=tokenizer,
            engine_notes=profile.notes,
            supports_reasoning=profile.supports_reasoning,
            supports_reasoning_effort=profile.supports_reasoning_effort,
            is_reasoning_model=_validate_reasoning(
                is_reasoning_model, profile, strict_capabilities
            ),
            chat_template_kwargs=(
                dict(chat_template_kwargs) if chat_template_kwargs else None
            ),
        )

    @staticmethod
    def _resolve_window(
        *,
        explicit: int | None,
        probed: int | None,
        profile: EngineProfile,
        model: str,
    ) -> tuple[int, ContextWindowSource]:
        """Apply the context-window precedence chain."""
        if explicit is not None:
            return _validate_context_window(explicit, "context_window"), "explicit"
        if probed is not None:
            return _validate_context_window(probed, "probed context window"), "probe"
        if profile.default_context_window is not None:
            return profile.default_context_window, "engine"

        _logger.warning(
            "No context window declared for model %r on engine %r and none "
            "could be discovered; falling back to %d tokens. Pass "
            "context_window=<tokens> to size the context budget correctly — "
            "an under-estimate only costs extra compaction, but a wrong "
            "over-estimate causes server-side context-overflow errors.",
            model,
            profile.name,
            DEFAULT_CONTEXT_WINDOW,
        )
        return DEFAULT_CONTEXT_WINDOW, "default"


# ---------------------------------------------------------------------- #
# Layer-1 validation helpers                                              #
# ---------------------------------------------------------------------- #


def _pick(override: bool | None, preset: bool) -> bool:
    """Explicit override wins over the engine preset."""
    return preset if override is None else override


def _resolve_engine(engine: str) -> EngineProfile:
    try:
        return get_engine_profile(engine)
    except ValueError as exc:
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER, message=str(exc), original_error=exc
        ) from exc


def _validate_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=(
                "model is required and must be the name your server serves "
                "(vLLM's --served-model-name, or the deployment name on "
                "Azure). Set OPENAI_COMPATIBLE_MODEL or pass model=..."
            ),
        )
    return model.strip()


def _validate_context_window(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=f"{label} must be an integer number of tokens, got {value!r}",
        )
    if not MIN_REASONABLE_CONTEXT <= value <= MAX_REASONABLE_CONTEXT:
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=(
                f"{label}={value} is outside the plausible range "
                f"[{MIN_REASONABLE_CONTEXT}, {MAX_REASONABLE_CONTEXT}]. "
                "Pass the window in tokens, e.g. context_window=32768."
            ),
        )
    return value


def _validate_structured_output_mode(mode: str) -> StructuredOutputMode:
    if mode not in _STRUCTURED_OUTPUT_MODES:
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=(
                f"structured_output_with_tools must be one of "
                f"{', '.join(_STRUCTURED_OUTPUT_MODES)}; got {mode!r}"
            ),
        )
    return mode  # type: ignore[return-value]


def _validate_max_tokens_field(field: str) -> str:
    if field not in MAX_TOKENS_FIELDS:
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=(
                f"max_tokens_field must be one of {', '.join(MAX_TOKENS_FIELDS)}; "
                f"got {field!r}"
            ),
        )
    return field


def _validate_reasoning(
    is_reasoning_model: bool, profile: EngineProfile, strict: bool
) -> bool:
    """Warn when thinking is declared on an engine that cannot surface it.

    Declaring ``is_reasoning_model=True`` is still honoured — it only widens
    internal token budgets, which is harmless — but on an engine with no
    reasoning parser the thinking text will arrive inline in ``content``
    rather than in its own field, and the caller should know.
    """
    if not is_reasoning_model or profile.supports_reasoning:
        return is_reasoning_model

    message = (
        f"is_reasoning_model=True was set but engine {profile.name!r} does "
        "not declare support for separating thinking from the answer. "
        "Thinking tokens will appear inline in the message content. For "
        "vLLM or SGLang, start the server with --reasoning-parser <parser>."
    )
    if strict:
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER, message=message
        )
    _logger.warning(message)
    return is_reasoning_model


def _validate_max_output_tokens(value: int | None, window: int) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=f"max_output_tokens must be a positive integer, got {value!r}",
        )
    if value > window:
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=(
                f"max_output_tokens={value} exceeds the context window "
                f"({window}); no tokens would remain for the prompt."
            ),
        )
