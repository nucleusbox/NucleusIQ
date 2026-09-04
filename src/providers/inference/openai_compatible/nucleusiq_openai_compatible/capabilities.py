"""Engine presets — capability defaults per inference server.

An :class:`EngineProfile` is **data, not behavior**.  Selecting
``engine="vllm"`` only changes defaults; every field remains overridable from
the :class:`~nucleusiq_openai_compatible.OpenAICompatibleLLM` constructor,
because a server's real capabilities depend on the flags it was started with
(a vLLM instance without ``--enable-auto-tool-choice`` cannot call tools no
matter what the preset claims).

Adding support for a new engine means adding a row to :data:`ENGINE_PRESETS`.
No existing code changes — the open/closed principle applied to a registry.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "ENGINE_PRESETS",
    "EngineProfile",
    "check_parallel_tool_calls",
    "get_engine_profile",
    "known_engines",
]

_logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW = 8_192
"""Conservative floor when no window is declared or discoverable.

Deliberately *not* OpenAI's 128K.  For a context-management framework
over-reporting is the dangerous direction: the engine skips compaction and
the server rejects the request.  Under-reporting only costs extra compaction.
"""

MAX_TOKENS_FIELDS = ("max_tokens", "max_completion_tokens")
"""The only legal wire names for the output-length parameter."""


@dataclass(frozen=True, slots=True)
class EngineProfile:
    """Capability defaults for one class of OpenAI-compatible server.

    Attributes:
        name: Preset key, e.g. ``"vllm"``.
        supports_tools: Server can accept ``tools`` and return ``tool_calls``.
        supports_json_schema: Server implements
            ``response_format={"type": "json_schema"}``.
        supports_parallel_tool_calls: Server honours ``parallel_tool_calls``.
        supports_stream_usage: Server returns usage on streams when asked via
            ``stream_options={"include_usage": True}``.
        max_tokens_field: Wire name for the output-length parameter.
        context_probe_field: Field on the ``/v1/models`` model card carrying
            the context length, or ``None`` if the engine does not publish it.
        default_context_window: Preset fallback window, or ``None`` to fall
            through to :data:`DEFAULT_CONTEXT_WINDOW`.
        extra_body_allowed: Server tolerates non-OpenAI keys in the body
            (vLLM's ``top_k``, ``guided_json``, ``chat_template_kwargs``, …).
        structured_output_suppresses_tools: Sending ``response_format``
            together with ``tools`` silently prevents tool calls.  True for
            vLLM-family constrained decoding; see the provider design doc.
        supports_reasoning: Server can split thinking from the answer into a
            separate response field, given a ``--reasoning-parser``.
        supports_reasoning_effort: Server accepts a top-level
            ``reasoning_effort``.  vLLM maps it onto the chat template's
            thinking switch, so it must not be stripped there even though it
            originates in the OpenAI schema.
        notes: Operator-facing guidance surfaced in validation reports.
    """

    name: str
    supports_tools: bool
    supports_json_schema: bool
    supports_parallel_tool_calls: bool
    supports_stream_usage: bool
    max_tokens_field: str = "max_tokens"
    context_probe_field: str | None = None
    default_context_window: int | None = None
    extra_body_allowed: bool = False
    structured_output_suppresses_tools: bool = False
    supports_reasoning: bool = False
    supports_reasoning_effort: bool = False
    notes: str = ""


_VLLM_NOTE = (
    "Tool calling requires --enable-auto-tool-choice and --tool-call-parser. "
    "Thinking output requires --reasoning-parser. Sending response_format "
    "together with tools suppresses tool calls."
)

REASONING_FIELDS: tuple[str, ...] = ("reasoning", "reasoning_content")
"""Response field names carrying thinking output, newest name first.

vLLM emitted ``reasoning_content`` historically and renamed it to
``reasoning``; both are still in the wild depending on server version, so
every read tries them in order.
"""

THINKING_TEMPLATE_KWARGS: dict[str, str] = {
    "qwen3": "enable_thinking",
    "gemma4": "enable_thinking",
    "granite": "thinking",
    "deepseek_v3": "thinking",
    "holo2": "thinking",
}
"""Which ``chat_template_kwargs`` key toggles thinking, per model family.

Documentation only — the framework never infers a family from a model name.
Callers pass ``chat_template_kwargs=`` explicitly; this mapping exists so the
README and error messages can tell them which key their model wants.
"""

ENGINE_PRESETS: dict[str, EngineProfile] = {
    "vllm": EngineProfile(
        name="vllm",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=True,
        supports_stream_usage=True,
        context_probe_field="max_model_len",
        extra_body_allowed=True,
        structured_output_suppresses_tools=True,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        notes=_VLLM_NOTE,
    ),
    "sglang": EngineProfile(
        name="sglang",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=True,
        supports_stream_usage=True,
        context_probe_field="max_model_len",
        extra_body_allowed=True,
        structured_output_suppresses_tools=True,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        notes=(
            "Tool calling requires --tool-call-parser; thinking output "
            "requires --reasoning-parser."
        ),
    ),
    "tgi": EngineProfile(
        name="tgi",
        supports_tools=True,
        supports_json_schema=False,
        supports_parallel_tool_calls=False,
        supports_stream_usage=False,
        notes="JSON-schema support varies by TGI version; verify before relying on it.",
    ),
    "llamacpp": EngineProfile(
        name="llamacpp",
        supports_tools=True,
        supports_json_schema=False,
        supports_parallel_tool_calls=False,
        supports_stream_usage=True,
        extra_body_allowed=True,
        supports_reasoning=True,
        notes=(
            "Grammar, tool-parser and reasoning support all depend on the "
            "build and the model's chat template."
        ),
    ),
    "lmstudio": EngineProfile(
        name="lmstudio",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=False,
        supports_stream_usage=True,
        supports_reasoning=True,
        notes="Desktop development target; unauthenticated by default.",
    ),
    "ollama": EngineProfile(
        name="ollama",
        supports_tools=True,
        # Measured, not assumed: the /v1 shim accepts both json_schema and
        # json_object and honours neither, returning fenced markdown with
        # whatever keys the model chose (checked on gemma4:31b and
        # gpt-oss:120b against https://ollama.com/v1). Claiming support here
        # would send a schema the server quietly discards, so the agent would
        # believe it had a validated object and get prose. Declaring False
        # instead routes through json_object plus a prompt-injected schema,
        # which does return the requested shape. Ollama's *native* API does
        # support schemas via `format` — nucleusiq-ollama uses it.
        supports_json_schema=False,
        supports_parallel_tool_calls=False,
        supports_stream_usage=True,
        supports_reasoning=True,
        notes=(
            "OpenAI compatibility shim; response_format is ignored, so "
            "structured output travels in the prompt. Prefer nucleusiq-ollama "
            "for the native API."
        ),
    ),
    "nim": EngineProfile(
        name="nim",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=True,
        supports_stream_usage=True,
        extra_body_allowed=True,
        supports_reasoning=True,
        notes="NVIDIA NIM containers; Bearer NVIDIA_API_KEY when hosted.",
    ),
    "openrouter": EngineProfile(
        name="openrouter",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=True,
        supports_stream_usage=True,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        notes=(
            "Capabilities vary per upstream model. Set default_headers with "
            "HTTP-Referer and X-Title for attribution."
        ),
    ),
    "together": EngineProfile(
        name="together",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=True,
        supports_stream_usage=True,
        notes="Capabilities vary per model; rate limits are real.",
    ),
    "fireworks": EngineProfile(
        name="fireworks",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=True,
        supports_stream_usage=True,
        notes="Capabilities vary per model; rate limits are real.",
    ),
    "deepinfra": EngineProfile(
        name="deepinfra",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=False,
        supports_stream_usage=True,
        notes="Capabilities vary per model.",
    ),
    "databricks": EngineProfile(
        name="databricks",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=True,
        supports_stream_usage=True,
        notes="Model Serving endpoints; Bearer personal access token.",
    ),
    "litellm": EngineProfile(
        name="litellm",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=True,
        supports_stream_usage=True,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        notes="Proxy fronting many backends; capabilities follow the routed model.",
    ),
    "azure": EngineProfile(
        name="azure",
        supports_tools=True,
        supports_json_schema=True,
        supports_parallel_tool_calls=True,
        supports_stream_usage=True,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        notes=(
            "base_url must end with /openai/v1 and model= is the deployment "
            "name. Use auth=HeaderAuth('api-key', ...) for key auth, or "
            "api_key= for a Microsoft Entra bearer token. Preview features "
            "need default_query={'api-version': 'preview'}."
        ),
    ),
    "generic": EngineProfile(
        name="generic",
        supports_tools=True,
        supports_json_schema=False,
        supports_parallel_tool_calls=False,
        supports_stream_usage=False,
        notes=(
            "Conservative defaults for an unidentified server. Declare "
            "capabilities explicitly once you know what it supports."
        ),
    ),
}


def known_engines() -> tuple[str, ...]:
    """Return the sorted tuple of valid ``engine=`` values."""
    return tuple(sorted(ENGINE_PRESETS))


def get_engine_profile(engine: str) -> EngineProfile:
    """Look up a preset by name.

    Raises:
        ValueError: Unknown engine name.  The message lists every valid
            preset, so a typo is self-correcting rather than mysterious.
    """
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError(
            f"engine must be a non-empty string; valid values: {', '.join(known_engines())}"
        )
    key = engine.strip().lower()
    profile = ENGINE_PRESETS.get(key)
    if profile is None:
        raise ValueError(
            f"Unknown engine {engine!r}. Valid values: {', '.join(known_engines())}. "
            "Use engine='generic' for a server not listed."
        )
    return profile


def check_parallel_tool_calls(
    *,
    requested: bool,
    supported: bool,
    engine: str,
    strict: bool,
) -> bool:
    """Gate ``parallel_tool_calls`` against declared capability.

    Mirrors ``nucleusiq_groq.capabilities.check_parallel_tool_calls_capability``:
    warn by default so a capable-but-undeclared server still works, raise
    under ``strict_capabilities=True`` for callers who want certainty.

    Returns:
        Whether the parameter should be forwarded on the wire.

    Raises:
        ValueError: *strict* is set and the capability is not declared.
    """
    if not requested or supported:
        return requested

    message = (
        f"parallel_tool_calls=True requested but engine {engine!r} does not "
        "declare support for it. Pass supports_parallel_tool_calls=True if "
        "your server handles it."
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, UserWarning, stacklevel=2)
    _logger.warning("%s Dropping the parameter.", message)
    return False
