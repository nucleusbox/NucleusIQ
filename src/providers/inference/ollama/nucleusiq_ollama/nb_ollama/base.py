"""``BaseOllama`` — Ollama LLM adapter (official ``ollama`` Python SDK)."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import ollama
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.streaming.events import StreamEvent
from pydantic import BaseModel

from nucleusiq_ollama._shared.wire import ThinkLevel
from nucleusiq_ollama.llm_params import OllamaLLMParams
from nucleusiq_ollama.nb_ollama.chat import create_ollama_chat
from nucleusiq_ollama.nb_ollama.stream_adapter import stream_ollama_chat
from nucleusiq_ollama.structured_output import build_ollama_format, parse_response
from nucleusiq_ollama.tools import NATIVE_TOOL_TYPES
from nucleusiq_ollama.tools.converter import to_ollama_function_tool

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT = 8192

_MERGED_WIRE_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "max_output_tokens",
        "seed",
        "think",
        "keep_alive",
    }
)


class BaseOllama(BaseLLM):
    """Run chat models via a local or remote Ollama server."""

    NATIVE_TOOL_TYPES: frozenset = NATIVE_TOOL_TYPES

    def __init__(
        self,
        model_name: str = "llama3.2",
        *,
        host: str | None = None,
        api_key: str | None = None,
        async_mode: bool = True,
        temperature: float = 0.7,
        max_retries: int = 3,
        llm_params: OllamaLLMParams | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.async_mode = async_mode
        self.temperature = temperature
        self.max_retries = max_retries
        self._ollama_llm_params = llm_params

        client_kw: dict[str, Any] = {}
        resolved_host = host or os.getenv("OLLAMA_HOST")
        if resolved_host:
            client_kw["host"] = resolved_host
        key = api_key or os.getenv("OLLAMA_API_KEY")
        if key:
            client_kw["headers"] = {"Authorization": f"Bearer {key}"}

        if async_mode:
            self._client: ollama.AsyncClient | ollama.Client = ollama.AsyncClient(
                **client_kw
            )
        else:
            self._client = ollama.Client(**client_kw)

    def _convert_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        return to_ollama_function_tool(spec)

    def get_context_window(self) -> int:
        return _DEFAULT_CONTEXT

    def _merge_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if self._ollama_llm_params is not None:
            merged.update(self._ollama_llm_params.to_call_kwargs())
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
        bool | ThinkLevel | None,
        float | str | None,
        dict[str, Any],
    ]:
        """Pop wire fields from *merged* so they are not duplicated elsewhere."""
        extra = dict(merged)
        p_seed = extra.pop("seed", None)
        p_think = extra.pop("think", None)
        p_keep = extra.pop("keep_alive", None)
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
            p_think,
            p_keep,
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
            think,
            keep_alive,
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
        if extra:
            logger.debug("Ignoring unsupported Ollama call kwargs: %s", sorted(extra))

        output_schema_type: type[BaseModel] | None = None
        format_payload: str | dict[str, Any] | None = None

        if response_format is not None:
            if tools:
                logger.warning(
                    "Ollama may reject combining structured format with tools; "
                    "dropping format for this request."
                )
            else:
                if isinstance(response_format, tuple) and len(response_format) == 2:
                    raw_fmt, schema_type = response_format
                    # Agent/OutputSchema may supply an OpenAI-style wrapper dict; always
                    # normalize for Ollama ``format`` (JSON schema or ``\"json\"``).
                    format_payload = build_ollama_format(raw_fmt)
                    if isinstance(schema_type, type) and issubclass(
                        schema_type, BaseModel
                    ):
                        output_schema_type = schema_type
                else:
                    built = build_ollama_format(response_format)
                    if built is not None:
                        format_payload = built
                    if isinstance(response_format, type) and issubclass(
                        response_format, BaseModel
                    ):
                        output_schema_type = response_format

        result = await create_ollama_chat(
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
            format_payload=format_payload if not tools else None,
            think=think,
            keep_alive=keep_alive,
            seed=seed,
        )

        if (
            output_schema_type is not None
            and result.choices
            and not tools
            and format_payload is not None
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
            think,
            keep_alive,
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
        if extra:
            logger.debug("Ignoring unsupported Ollama stream kwargs: %s", sorted(extra))

        async for event in stream_ollama_chat(
            self._client,
            async_mode=self.async_mode,
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
            think=think,
            keep_alive=keep_alive,
            seed=seed,
        ):
            yield event
