"""Wire payload construction for Chat Completions.

Three responsibilities, all narrow:

* translate NucleusIQ's provider-neutral call arguments into an
  OpenAI-compatible request body;
* translate the framework's **canonical message shape** into the OpenAI
  dialect, which for assistant ``tool_calls`` is not the same thing (see
  :func:`sanitize_messages`);
* **strip anything a generic server will reject.**  OpenAI-cloud-only fields
  reach this layer through ``**kwargs`` (an agent configured for OpenAI, a
  shared `AgentConfig`, a plugin), and strict servers answer ``400 unknown
  parameter`` rather than ignoring them.  Dropping is logged at debug so the
  behavior is discoverable without being noisy.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from ..capabilities import check_parallel_tool_calls
from ..config import ResolvedConfig

__all__ = [
    "OPENAI_ONLY_PARAMS",
    "PayloadBuilder",
    "sanitize_messages",
]

_logger = logging.getLogger(__name__)

OPENAI_ONLY_PARAMS: frozenset[str] = frozenset(
    {
        "service_tier",
        "store",
        "prompt_cache_key",
        "prompt_cache_retention",
        "modalities",
        "audio",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "truncation",
        "safety_identifier",
        "max_tool_calls",
        "web_search_options",
        "metadata",
        "prediction",
        "organization",
    }
)
"""Parameters no generic OpenAI-compatible server can be relied on to accept."""

REASONING_PARAMS: frozenset[str] = frozenset(
    {"reasoning_effort", "reasoning", "include_reasoning"}
)
"""Thinking-mode controls.

These originate in the OpenAI schema but are **not** OpenAI-only: vLLM and
SGLang accept ``reasoning_effort`` and map it onto the chat template's
thinking switch, and several gateways accept ``include_reasoning``.  They are
forwarded when the engine declares reasoning support and stripped otherwise,
rather than being dropped unconditionally.
"""

# Framework-internal kwargs that must never reach the wire.
_INTERNAL_PARAMS: frozenset[str] = frozenset(
    {
        "api_key",
        "auth",
        "response_schema",
        "response_schema_name",
        "structured_output",
        "llm_params",
        "strict_capabilities",
    }
)

_SAMPLING_DEFAULTS: dict[str, float] = {
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


def _normalize_tool_call(tc: Any) -> Any:
    """Coerce one tool call into the OpenAI ``type``/``function`` shape.

    Entries already in the nested shape pass through untouched, so a caller
    that hand-builds OpenAI-shaped history is never rewritten.
    """
    if not isinstance(tc, dict):
        return tc
    if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
        return tc

    fn = tc.get("function")
    if isinstance(fn, dict):
        name = fn.get("name", "")
        arguments = fn.get("arguments", tc.get("arguments", "{}"))
    else:
        name = tc.get("name", "")
        arguments = tc.get("arguments", "{}")

    out: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    if tc.get("id") is not None:
        out["id"] = tc["id"]
    return out


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return *messages* with assistant tool calls in the OpenAI dialect.

    ``ToolCallRequest.to_dict`` serialises a tool call flat —
    ``{"id", "name", "arguments"}`` — and its docstring makes translating to
    the wire format the provider's job.  Chat Completions instead requires
    ``{"id", "type": "function", "function": {"name", "arguments"}}``.

    Sending the flat form costs nothing on the *first* request, because there
    are no tool calls in history yet; it breaks the request that echoes the
    call back alongside its result.  So the whole tool loop fails on round two
    with a message that names neither the field nor the shape — Ollama answers
    ``400 invalid tool call arguments``.  Mocked tests cannot catch it: a fake
    client returns a scripted reply whatever it is sent.

    Messages are copied only when a rewrite is needed, keeping the common
    tool-free path allocation-free.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if msg.get("role") != "assistant" or not isinstance(calls, list):
            out.append(msg)
            continue
        normalized = [_normalize_tool_call(tc) for tc in calls]
        if normalized == calls:
            out.append(msg)
            continue
        rewritten = copy.copy(msg)
        rewritten["tool_calls"] = normalized
        out.append(rewritten)
    return out


class PayloadBuilder:
    """Builds Chat Completions request bodies for one resolved configuration.

    Stateless with respect to a call; holds only the immutable
    :class:`~nucleusiq_openai_compatible.config.ResolvedConfig` so capability
    decisions come from a single source of truth.
    """

    __slots__ = ("_config",)

    def __init__(self, config: ResolvedConfig) -> None:
        self._config = config

    def build(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the request body.

        Args:
            messages: Chat messages, already provider-shaped.
            tools: Converted function-tool specs, or ``None``.
            tool_choice: ``"auto"``, ``"none"``, ``"required"`` or a specific
                tool selector. Omitted when no tools are present.
            max_output_tokens: Provider-neutral output cap, emitted under the
                configured wire name (``max_tokens`` or
                ``max_completion_tokens``).
            temperature: Sampling temperature.
            stop: Stop sequences.
            response_format: Already decided by the structured-output policy;
                passed through untouched.
            stream: Whether to request a streamed response.
            extra: Remaining call kwargs, filtered here.

        Returns:
            A request body ready to hand to ``client.chat.completions.create``.
        """
        cfg = self._config
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": sanitize_messages(messages),
        }

        limit = (
            max_output_tokens
            if max_output_tokens is not None
            else cfg.max_output_tokens
        )
        if limit is not None:
            payload[cfg.max_tokens_field] = limit
        if temperature is not None:
            payload["temperature"] = temperature
        if stop:
            payload["stop"] = stop
        if response_format is not None:
            payload["response_format"] = response_format

        if tools:
            if not cfg.supports_tools:
                self._reject_tools()
            else:
                payload["tools"] = tools
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice

        if stream:
            payload["stream"] = True
            if cfg.supports_stream_usage:
                # Many servers omit usage on streams unless asked, which
                # would leave call records without token counts.
                payload["stream_options"] = {"include_usage": True}

        payload.update(self._filter_extra(extra or {}, has_tools=bool(tools)))
        self._apply_chat_template_kwargs(payload)
        return payload

    def _apply_chat_template_kwargs(self, payload: dict[str, Any]) -> None:
        """Merge configured ``chat_template_kwargs`` into ``extra_body``.

        This is where a thinking toggle lands — ``enable_thinking`` for the
        Qwen3 and Gemma families, ``thinking`` for Granite and DeepSeek.  It
        is configured once per deployment rather than per call, but has to
        travel on every request because the template is applied server-side
        at request time.  Per-call ``extra_body`` wins on key conflicts.
        """
        configured = self._config.chat_template_kwargs
        if not configured or not self._config.extra_body_allowed:
            return
        body = payload.setdefault("extra_body", {})
        existing = body.get("chat_template_kwargs")
        body["chat_template_kwargs"] = (
            {**configured, **existing}
            if isinstance(existing, dict)
            else dict(configured)
        )

    # ------------------------------------------------------------------ #
    # Filtering                                                           #
    # ------------------------------------------------------------------ #

    def _filter_extra(
        self, extra: dict[str, Any], *, has_tools: bool
    ) -> dict[str, Any]:
        """Drop unsupported, internal and redundant keys from *extra*."""
        cfg = self._config
        out: dict[str, Any] = {}
        dropped: list[str] = []

        for key, value in extra.items():
            if value is None or key in _INTERNAL_PARAMS:
                continue
            if key in OPENAI_ONLY_PARAMS:
                dropped.append(key)
                continue

            if key in REASONING_PARAMS:
                if key == "reasoning_effort" and not cfg.supports_reasoning_effort:
                    dropped.append(key)
                    continue
                if key != "reasoning_effort" and not cfg.supports_reasoning:
                    dropped.append(key)
                    continue
                out[key] = value
                continue

            if key == "extra_body":
                if not isinstance(value, dict) or not value:
                    continue
                if not cfg.extra_body_allowed:
                    dropped.append("extra_body")
                    continue
                out["extra_body"] = dict(value)
                continue

            if key == "parallel_tool_calls":
                forward = check_parallel_tool_calls(
                    requested=bool(value),
                    supported=cfg.supports_parallel_tool_calls,
                    engine=cfg.engine,
                    strict=cfg.strict_capabilities,
                )
                if forward and has_tools:
                    out[key] = True
                continue

            # Sampling knobs left at their neutral defaults add nothing and
            # are rejected outright by some stricter servers.
            if key in _SAMPLING_DEFAULTS and value == _SAMPLING_DEFAULTS[key]:
                continue

            out[key] = value

        if dropped:
            _logger.debug(
                "Dropped %d parameter(s) unsupported by engine %r: %s",
                len(dropped),
                cfg.engine,
                ", ".join(sorted(dropped)),
            )
        return out

    def _reject_tools(self) -> None:
        """Handle tools being supplied to a server that cannot use them."""
        message = (
            f"Tools were supplied but engine {self._config.engine!r} is not "
            "configured for tool calling. For vLLM/SGLang, start the server "
            "with --enable-auto-tool-choice and --tool-call-parser <parser>, "
            "then pass supports_tools=True."
        )
        if self._config.strict_capabilities:
            from nucleusiq.llms.errors import InvalidRequestError

            raise InvalidRequestError.from_provider_error(
                provider="openai_compatible", message=message
            )
        import warnings

        warnings.warn(message, UserWarning, stacklevel=3)
        _logger.warning("%s Dropping tools from the request.", message)
