"""``BaseAnthropic`` — Claude adapter over the Messages API."""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, cast

import anthropic
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.streaming.events import StreamEvent
from pydantic import BaseModel

from nucleusiq_anthropic.llm_params import AnthropicLLMParams
from nucleusiq_anthropic.nb_anthropic.messages import create_message_response
from nucleusiq_anthropic.nb_anthropic.stream_adapter import stream_messages
from nucleusiq_anthropic.structured_output import (
    build_anthropic_output_config,
    parse_anthropic_response,
)
from nucleusiq_anthropic.tools import NATIVE_TOOL_TYPES, to_anthropic_tool_definition

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT = 128_000

_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-7-sonnet-20250219": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-opus-4-20250507": 200_000,
    "claude-haiku-4-20250415": 200_000,
    "claude-3-haiku-20240307": 200_000,
}


def _merge_output_config_wire(
    prev: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Shallow-merge ``output_config`` dicts; nested ``format`` keys merge one level."""
    out = copy.deepcopy(prev)
    for key, val in incoming.items():
        if key == "format" and isinstance(out.get(key), dict) and isinstance(val, dict):
            merged_fmt = {**dict(cast(dict[str, Any], out[key])), **val}
            out[key] = merged_fmt
        else:
            out[key] = copy.deepcopy(val)
    return out


def _coerce_stop_sequences(raw: Any) -> list[str] | None:
    """Normalize heterogeneous stop-sequence wire values → ``list[str]``."""

    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [s for s in raw if isinstance(s, str) and s]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return None


class BaseAnthropic(BaseLLM):
    """Anthropic Claude client implementing :class:`~nucleusiq.llms.base_llm.BaseLLM`.

    Modular collaborators:

    * :mod:`~nucleusiq_anthropic._shared.wire` — canonical chat dicts → Messages payloads
    * :mod:`~nucleusiq_anthropic.nb_anthropic.messages` — non-stream + normalisation
    * :mod:`~nucleusiq_anthropic.nb_anthropic.stream_adapter` — ``StreamEvent``
    * :mod:`~nucleusiq_anthropic.structured_output` — ``output_config.format`` (JSON Schema)
    * :mod:`~nucleusiq_anthropic._shared.retry` — provider-wide backoff policy
    """

    NATIVE_TOOL_TYPES: frozenset[str] = NATIVE_TOOL_TYPES

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        *,
        async_mode: bool = True,
        temperature: float = 1.0,
        max_retries: int = 3,
        llm_params: AnthropicLLMParams | None = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.async_mode = async_mode
        self.temperature = temperature
        self.max_retries = max_retries
        self._anthropic_llm_params = llm_params

        raw_key = (
            api_key if api_key not in ("", None) else os.getenv("ANTHROPIC_API_KEY")
        )

        if not isinstance(raw_key, str) or not raw_key.strip():
            from nucleusiq.llms.errors import AuthenticationError

            raise AuthenticationError.from_provider_error(
                provider="anthropic",
                message=(
                    "ANTHROPIC_API_KEY is required — set env var "
                    "or pass api_key= to the constructor."
                ),
                status_code=401,
                original_error=None,
            )

        key = raw_key.strip()
        self._client: anthropic.AsyncAnthropic | anthropic.Anthropic
        self._client = (
            anthropic.AsyncAnthropic(api_key=key)
            if async_mode
            else anthropic.Anthropic(api_key=key)
        )

    # --------------------------------------------------------------------- #
    # Tool conversion                                                          #
    # --------------------------------------------------------------------- #
    def _convert_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        return to_anthropic_tool_definition(spec)

    def get_context_window(self) -> int:
        """Return Claude context budgets for known SKU strings."""

        for prefix, tokens in sorted(
            _MODEL_CONTEXT_WINDOWS.items(),
            key=lambda kv: -len(kv[0]),
        ):
            if self.model_name == prefix:
                return tokens
            if self.model_name.startswith(prefix):
                return tokens

        name_l = self.model_name.lower()

        try:
            if name_l.startswith("claude-opus"):
                return 200_000
            if name_l.startswith("claude-sonnet"):
                return 200_000
            if name_l.startswith("claude-haiku"):
                return 200_000
        except Exception:
            pass

        return _DEFAULT_CONTEXT

    def _merge_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}

        xh_from_llm_params: dict[str, str] | None = None
        if self._anthropic_llm_params is not None:
            from_params = self._anthropic_llm_params.to_call_kwargs()
            mh = from_params.pop("_merged_extra_headers", None)
            if isinstance(mh, dict):
                xh_from_llm_params = cast(dict[str, str], mh)

            merged.update(from_params)

        merged.update(kwargs)

        call_headers = merged.pop("extra_headers", None)
        extra_from_params_kw = merged.pop("_merged_extra_headers", None)

        combined_headers: dict[str, str] = {}
        if isinstance(xh_from_llm_params, dict):
            combined_headers.update(dict(xh_from_llm_params))
        if isinstance(extra_from_params_kw, dict):
            combined_headers.update(dict(extra_from_params_kw))
        if isinstance(call_headers, dict):
            combined_headers.update(dict(call_headers))

        merged["_effective_extra_headers"] = (
            combined_headers if combined_headers else None
        )

        return merged

    def _strip_duplicate_wire_keys(self, extras: dict[str, Any]) -> dict[str, Any]:
        stripped = dict(extras)

        stripped.pop("_effective_extra_headers", None)

        for k in (
            "temperature",
            "top_p",
            "stop",
            "max_output_tokens",
            "tools",
            "tool_choice",
            "response_format",
        ):
            stripped.pop(k, None)

        return stripped

    def _resolve_response_format_for_output_config(
        self,
        response_format: Any,
        *,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any] | None, Any]:
        """Map ``response_format`` to ``output_config`` + optional Pydantic/dataclass type."""

        if response_format is None:
            return None, None

        if tools:
            logger.warning(
                "Claude structured outputs plus function tools in one request are "
                "often unsupported or model-dependent — dropping structured output."
            )
            return None, None

        output_schema_type: Any = None
        oc_frag: dict[str, Any] | None = None

        if isinstance(response_format, tuple) and len(response_format) == 2:
            hint, raw_type = response_format
            if isinstance(hint, dict):
                oc_frag = copy.deepcopy(hint)
            if isinstance(raw_type, type):
                output_schema_type = raw_type
        else:
            oc = build_anthropic_output_config(response_format)
            if oc:
                oc_frag = oc
                if isinstance(response_format, type):
                    output_schema_type = response_format

        return oc_frag, output_schema_type

    async def call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_output_tokens: int = 4096,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        response_format: type[BaseModel]
        | type
        | dict[str, Any]
        | tuple[Any, Any]
        | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke ``messages.create`` and normalize to ``AnthropicLLMResponse``.

        When *response_format* resolves to a typed schema, returns a validated
        instance (Pydantic/dataclass) instead of :class:`AnthropicLLMResponse`.
        """

        _ = frequency_penalty, presence_penalty

        merged = self._merge_call_kwargs(dict(kwargs))
        extra_headers = merged.pop("_effective_extra_headers", None)

        eff_temperature = (
            temperature if temperature is not None else merged.pop("temperature", None)
        )
        if eff_temperature is None:
            eff_temperature = self.temperature

        merged_top_p = merged.pop("top_p", None)
        eff_top_p: float | None
        if merged_top_p is not None:
            eff_top_p = merged_top_p
        elif top_p is not None:
            eff_top_p = top_p
        else:
            eff_top_p = None

        eff_stop_raw = merged.pop("stop", None)
        eff_stop_sequences: list[str] | None
        if stop is not None:
            eff_stop_sequences = _coerce_stop_sequences(stop)
        else:
            eff_stop_sequences = _coerce_stop_sequences(eff_stop_raw)

        max_override = merged.pop("max_output_tokens", None)

        merged_tool_choice = merged.pop("tool_choice", tool_choice)

        effective_tokens = (
            max_override if max_override is not None else max_output_tokens
        )

        rf_merged = merged.pop("response_format", None)
        rf_eff = response_format if response_format is not None else rf_merged

        oc_built, output_schema_type = self._resolve_response_format_for_output_config(
            rf_eff,
            tools=tools,
        )

        user_oc_raw = merged.pop("output_config", None)
        merged_oc: dict[str, Any] | None = None
        if isinstance(user_oc_raw, dict) or oc_built is not None:
            merged_oc = {}
            if isinstance(user_oc_raw, dict):
                merged_oc = _merge_output_config_wire(merged_oc, user_oc_raw)
            if oc_built is not None:
                merged_oc = _merge_output_config_wire(merged_oc, oc_built)

        extras = self._strip_duplicate_wire_keys(merged)
        if merged_oc:
            extras["output_config"] = merged_oc

        xh = dict(extra_headers) if isinstance(extra_headers, dict) else None

        result = await create_message_response(
            self._client,
            async_mode=self.async_mode,
            max_retries=self.max_retries,
            model=model,
            messages=messages,
            max_output_tokens=int(effective_tokens),
            temperature=eff_temperature,
            top_p=eff_top_p,
            stop=eff_stop_sequences,
            tools=tools,
            tool_choice=merged_tool_choice,
            merged_extras=extras,
            extra_headers=xh,
        )

        if (
            output_schema_type is not None
            and hasattr(result, "choices")
            and result.choices
        ):
            msg = result.choices[0].message
            content = getattr(msg, "content", None)
            if content:
                from nucleusiq_anthropic._shared.response_models import AssistantMessage

                if isinstance(msg, AssistantMessage):
                    return parse_anthropic_response(
                        msg.to_dict(),
                        output_schema_type,
                    )
                if isinstance(msg, dict):
                    return parse_anthropic_response(msg, output_schema_type)

        return result

    async def call_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_output_tokens: int = 4096,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Incremental Claude tokens + terminal ``COMPLETE`` metadata."""

        _ = frequency_penalty, presence_penalty

        merged = self._merge_call_kwargs(dict(kwargs))
        xh_src = merged.pop("_effective_extra_headers", None)

        rf_merged = merged.pop("response_format", None)
        if rf_merged is not None:
            logger.warning(
                "Ignoring response_format for streaming; use non-stream "
                "`call()` for structured Claude outputs."
            )

        eff_temperature = (
            temperature if temperature is not None else merged.pop("temperature", None)
        )
        if eff_temperature is None:
            eff_temperature = self.temperature

        merged_top_p = merged.pop("top_p", None)
        eff_top_p: float | None
        if merged_top_p is not None:
            eff_top_p = merged_top_p
        elif top_p is not None:
            eff_top_p = top_p
        else:
            eff_top_p = None

        eff_stop_raw = merged.pop("stop", None)
        if stop is not None:
            eff_stop_sequences = _coerce_stop_sequences(stop)
        else:
            eff_stop_sequences = _coerce_stop_sequences(eff_stop_raw)

        merged_tool_choice = merged.pop("tool_choice", tool_choice)
        max_override = merged.pop("max_output_tokens", None)
        effective_tokens = (
            max_override if max_override is not None else max_output_tokens
        )

        extras = self._strip_duplicate_wire_keys(merged)
        xh_dict = dict(xh_src) if isinstance(xh_src, dict) else None

        async for ev in stream_messages(
            self._client,
            async_mode=self.async_mode,
            max_retries=self.max_retries,
            model=model,
            messages=messages,
            max_output_tokens=int(effective_tokens),
            temperature=eff_temperature,
            top_p=eff_top_p,
            stop=eff_stop_sequences,
            tools=tools,
            tool_choice=merged_tool_choice,
            merged_extras=extras,
            extra_headers=xh_dict,
        ):
            yield ev


__all__ = ["BaseAnthropic"]
