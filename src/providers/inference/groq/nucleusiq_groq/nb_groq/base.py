"""``BaseGroq`` — Groq LLM adapter (official ``groq`` SDK, Chat Completions)."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import groq
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.streaming.events import StreamEvent
from pydantic import BaseModel

from nucleusiq_groq._shared.wire import build_chat_completion_payload
from nucleusiq_groq.capabilities import check_parallel_tool_calls_capability
from nucleusiq_groq.llm_params import GroqLLMParams
from nucleusiq_groq.nb_groq.chat import create_chat_completion
from nucleusiq_groq.nb_groq.stream_adapter import stream_chat_completions
from nucleusiq_groq.structured_output import build_response_format, parse_response
from nucleusiq_groq.tools import NATIVE_TOOL_TYPES
from nucleusiq_groq.tools.converter import to_openai_function_tool

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT = 128_000

# Keys handled explicitly on the Chat Completions wire; strip from ``extra`` merge.
_MERGED_WIRE_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "max_output_tokens",
        "seed",
        "user",
        "parallel_tool_calls",
    }
)
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "llama-3.3-70b-versatile": 131_072,
    "llama-3.1-8b-instant": 131_072,
    "meta-llama/llama-4-scout-17b-16e-instruct": 10_000_000,
    "openai/gpt-oss-20b": 131_072,
    "openai/gpt-oss-120b": 131_072,
}


class BaseGroq(BaseLLM):
    """Groq inference via the official ``groq`` Python SDK."""

    NATIVE_TOOL_TYPES: frozenset = NATIVE_TOOL_TYPES

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        *,
        async_mode: bool = True,
        temperature: float = 0.7,
        max_retries: int = 3,
        llm_params: GroqLLMParams | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.async_mode = async_mode
        self.temperature = temperature
        self.max_retries = max_retries
        self._groq_llm_params = llm_params

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            from nucleusiq.llms.errors import AuthenticationError

            raise AuthenticationError.from_provider_error(
                provider="groq",
                message=(
                    "GROQ_API_KEY is required. Set the environment variable "
                    "or pass api_key= to the constructor."
                ),
                status_code=401,
                original_error=None,
            )

        if async_mode:
            self._client: groq.AsyncGroq | groq.Groq = groq.AsyncGroq(api_key=key)
        else:
            self._client = groq.Groq(api_key=key)

    def _convert_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        return to_openai_function_tool(spec)

    def get_context_window(self) -> int:
        return _MODEL_CONTEXT_WINDOWS.get(self.model_name, _DEFAULT_CONTEXT)

    def _merge_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if self._groq_llm_params is not None:
            merged.update(self._groq_llm_params.to_call_kwargs())
        merged.update(kwargs)
        return merged

    def _resolve_wire_params(
        self,
        merged: dict[str, Any],
        *,
        temperature: float | None,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        stop: list[str] | None,
        max_output_tokens: int,
    ) -> tuple[
        float | None,
        float,
        float,
        float,
        list[str] | None,
        int,
        int | None,
        bool | None,
        str | None,
        dict[str, Any],
    ]:
        """Pop wire fields from *merged* so they are not duplicated in ``extra``.

        Explicit ``call()`` parameters (sampling, max tokens, stop, etc.) are
        forwarded as-is — they already reflect agent-level ``LLMParams`` merges.
        Only ``temperature`` falls back to :pyattr:`GroqLLMParams.temperature`
        then :pyattr:`self.temperature` when the call passes ``None``.
        """
        extra = dict(merged)
        p_seed = extra.pop("seed", None)
        p_user = extra.pop("user", None)
        p_parallel = extra.pop("parallel_tool_calls", None)
        p_temp = extra.pop("temperature", None)
        extra.pop("top_p", None)
        extra.pop("frequency_penalty", None)
        extra.pop("presence_penalty", None)
        extra.pop("stop", None)
        extra.pop("max_output_tokens", None)

        for k in _MERGED_WIRE_KEYS:
            extra.pop(k, None)

        eff_temperature = temperature if temperature is not None else p_temp
        if eff_temperature is None:
            eff_temperature = self.temperature

        return (
            eff_temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            stop,
            max_output_tokens,
            p_seed,
            p_parallel,
            p_user,
            extra,
        )

    async def call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        response_format: type[BaseModel] | type | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        merged = self._merge_call_kwargs(dict(kwargs))
        (
            eff_temperature,
            eff_top_p,
            eff_fp,
            eff_pp,
            eff_stop,
            eff_max_out,
            seed,
            parallel_tool_calls,
            user,
            extra,
        ) = self._resolve_wire_params(
            merged,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            max_output_tokens=max_output_tokens,
        )

        strict_caps = (
            self._groq_llm_params.strict_model_capabilities
            if self._groq_llm_params is not None
            else False
        )
        check_parallel_tool_calls_capability(
            model,
            parallel_tool_calls,
            strict=strict_caps,
            logger=logger,
        )

        output_schema_type: type[BaseModel] | None = None
        api_response_format: dict[str, Any] | None = None

        if response_format is not None:
            if tools:
                logger.warning(
                    "Groq may reject combining response_format with tools; "
                    "dropping response_format for this request."
                )
            else:
                if isinstance(response_format, tuple) and len(response_format) == 2:
                    api_response_format, schema_type = response_format
                    if isinstance(schema_type, type) and issubclass(
                        schema_type, BaseModel
                    ):
                        output_schema_type = schema_type
                else:
                    built = build_response_format(response_format)
                    if built:
                        api_response_format = built
                    if isinstance(response_format, type) and issubclass(
                        response_format, BaseModel
                    ):
                        output_schema_type = response_format

        result = await create_chat_completion(
            self._client,
            async_mode=self.async_mode,
            max_retries=self.max_retries,
            model=model,
            messages=messages,
            max_output_tokens=eff_max_out,
            temperature=eff_temperature,
            top_p=eff_top_p,
            frequency_penalty=eff_fp,
            presence_penalty=eff_pp,
            stop=eff_stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=api_response_format if not tools else None,
            parallel_tool_calls=parallel_tool_calls,
            seed=seed,
            user=user,
            extra=extra,
        )

        if (
            output_schema_type is not None
            and result.choices
            and not tools
            and api_response_format is not None
        ):
            msg = result.choices[0].message
            if msg.content:
                return parse_response(msg.to_dict(), output_schema_type)

        return result

    async def call_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        merged = self._merge_call_kwargs(dict(kwargs))
        (
            eff_temperature,
            eff_top_p,
            eff_fp,
            eff_pp,
            eff_stop,
            eff_max_out,
            seed,
            parallel_tool_calls,
            user,
            extra,
        ) = self._resolve_wire_params(
            merged,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            max_output_tokens=max_output_tokens,
        )

        strict_caps = (
            self._groq_llm_params.strict_model_capabilities
            if self._groq_llm_params is not None
            else False
        )
        check_parallel_tool_calls_capability(
            model,
            parallel_tool_calls,
            strict=strict_caps,
            logger=logger,
        )

        payload = build_chat_completion_payload(
            model=model,
            messages=messages,
            max_tokens=eff_max_out,
            temperature=eff_temperature,
            top_p=eff_top_p,
            frequency_penalty=eff_fp,
            presence_penalty=eff_pp,
            stop=eff_stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=None,
            parallel_tool_calls=parallel_tool_calls,
            seed=seed,
            user=user,
            extra=extra,
        )

        async for event in stream_chat_completions(
            self._client,
            payload,
            async_mode=self.async_mode,
            max_retries=self.max_retries,
        ):
            yield event
