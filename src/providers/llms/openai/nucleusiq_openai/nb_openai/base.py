"""
OpenAI provider for NucleusIQ.

This module provides the OpenAI LLM client that supports both async and sync
modes.  The mode is determined by the ``async_mode`` parameter at init time.

**API Backend Routing (transparent to callers)**

``call()`` automatically selects the right OpenAI API:

* **Chat Completions API** — used when all tools are custom function-calling
  tools (or when there are no tools at all).
* **Responses API** — used when *any* tool is a native OpenAI tool
  (web_search, code_interpreter, file_search, image_generation, mcp,
  computer_use).  The Responses API also supports function-calling, so
  mixed tool lists work seamlessly.

Callers (Agents) never need to choose — the routing is fully internal.

Structured Output Support::

    response = await llm.call(
        model="gpt-4o",
        messages=[...],
        response_format=MyPydanticModel,  # Returns validated MyPydanticModel instance
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any

import openai
from nucleusiq.llms.base_llm import BaseLLM
from pydantic import BaseModel

from nucleusiq_openai._shared.model_config import (
    is_strict_defaults_model,
    uses_max_completion_tokens,
)
from nucleusiq_openai._shared.response_models import (
    AssistantMessage,
    _LLMResponse,
)
from nucleusiq_openai.nb_openai.chat_completions import call_chat_completions
from nucleusiq_openai.nb_openai.responses_api import (
    call_responses_api,
    responses_call_direct,
)
from nucleusiq_openai.structured_output import build_response_format, parse_response
from nucleusiq_openai.tools.openai_tool import NATIVE_TOOL_TYPES

logger = logging.getLogger(__name__)

__all__ = ["BaseOpenAI"]


class BaseOpenAI(BaseLLM):
    """
    OpenAI client for ChatCompletion **and** Responses API.

    Supports both async and sync modes based on ``async_mode`` parameter
    (default True).

    **Open/Closed Principle** — the public ``call()`` interface is stable;
    new API backends are added as separate modules without modifying the
    contract.

    **Adaptive** — tool type registries and version constants are configurable
    so API changes are absorbed in one place.
    """

    NATIVE_TOOL_TYPES: frozenset = NATIVE_TOOL_TYPES

    # ================================================================== #
    # Tool spec conversion                                                #
    # ================================================================== #

    def _convert_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        """
        Convert generic BaseTool spec to OpenAI format.

        * If spec already has a ``"type"`` key (native tool from
          ``OpenAITool``), it is returned **as-is** (pass-through).
        * Otherwise it is wrapped in the
          ``{"type": "function", "function": {...}}`` envelope that both
          the Chat Completions and Responses APIs accept.
        """
        if "type" in spec:
            return spec

        parameters = spec.get("parameters", {})
        if "additionalProperties" not in parameters:
            parameters = {**parameters, "additionalProperties": False}

        return {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": parameters,
            },
        }

    # ================================================================== #
    # __init__                                                            #
    # ================================================================== #

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.7,
        logit_bias: dict[str, float] | None = None,
        async_mode: bool = True,
    ) -> None:
        """
        Initialize OpenAI chat client with sensible defaults.

        Args:
            model_name: Model identifier (e.g. ``"gpt-4o"``).
            api_key: OpenAI API key (falls back to ``OPENAI_API_KEY`` env var).
            base_url: Custom base URL (falls back to ``OPENAI_API_BASE``).
            organization: Org ID (falls back to ``OPENAI_ORG_ID``).
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for transient errors.
            temperature: Default sampling temperature.
            logit_bias: Token-level logit bias dict.
            async_mode: ``True`` → ``AsyncOpenAI``, ``False`` → ``OpenAI``.
        """
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.logit_bias = logit_bias
        self.async_mode = async_mode
        self._logger = logging.getLogger("BaseOpenAI")

        self._last_response_id: str | None = None

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")

        if self.async_mode:
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        else:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

    # ================================================================== #
    # Public helpers                                                      #
    # ================================================================== #

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string."""
        import tiktoken

        enc = tiktoken.encoding_for_model(self.model_name)
        return len(enc.encode(text))

    # Delegate model quirk checks for external use
    def _uses_max_completion_tokens(self, model: str) -> bool:
        return uses_max_completion_tokens(model)

    def _is_strict_defaults_model(self, model: str) -> bool:
        return is_strict_defaults_model(model)

    # ================================================================== #
    # call() — unified entry point (smart routing)                        #
    # ================================================================== #

    async def call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        stream: bool = False,
        response_format: type[BaseModel] | type | dict[str, Any] | None = None,
        seed: int | None = None,
        n: int | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        modalities: list[str] | None = None,
        audio: dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
        store: bool | None = None,
        truncation: str | None = None,
        max_tool_calls: int | None = None,
        safety_identifier: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call OpenAI API with optional structured output.

        **Routing is automatic** — callers never specify which API to use.
        If *tools* contains any native OpenAI tool the Responses API is used;
        otherwise the Chat Completions API is used.
        """
        # ---- 1. Resolve structured output config ----
        output_schema_type = None
        api_response_format = None

        if response_format is not None:
            if isinstance(response_format, tuple) and len(response_format) == 2:
                api_response_format, schema_type = response_format
                if isinstance(schema_type, type):
                    output_schema_type = schema_type
            else:
                openai_response_format = build_response_format(response_format)
                if openai_response_format:
                    api_response_format = openai_response_format
                    if isinstance(response_format, type):
                        output_schema_type = response_format

        # ---- 1b. Collect extra params ----
        extra: dict[str, Any] = {}
        for _key, _val in [
            ("seed", seed),
            ("n", n),
            ("reasoning_effort", reasoning_effort),
            ("service_tier", service_tier),
            ("logprobs", logprobs),
            ("top_logprobs", top_logprobs),
            ("parallel_tool_calls", parallel_tool_calls),
            ("modalities", modalities),
            ("audio", audio),
            ("metadata", metadata),
            ("store", store),
            ("truncation", truncation),
            ("max_tool_calls", max_tool_calls),
            ("safety_identifier", safety_identifier),
            ("prompt_cache_key", prompt_cache_key),
            ("prompt_cache_retention", prompt_cache_retention),
        ]:
            if _val is not None:
                extra[_key] = _val
        extra.update(kwargs)

        # ---- 2. Route to appropriate backend ----
        if tools and self._has_native_tools(tools):
            result = await self._call_responses_api(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                response_format=api_response_format,
                **extra,
            )
        else:
            result = await call_chat_completions(
                self._client,
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                default_temperature=self.temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=stream,
                response_format=api_response_format,
                logit_bias=self.logit_bias,
                max_retries=self.max_retries,
                async_mode=self.async_mode,
                logger=self._logger,
                **extra,
            )

        # ---- 3. Parse structured output if requested ----
        if (
            output_schema_type is not None
            and hasattr(result, "choices")
            and result.choices
        ):
            msg = result.choices[0].message
            if isinstance(msg, AssistantMessage) and msg.content:
                return parse_response(msg.to_dict(), output_schema_type)
            elif isinstance(msg, dict) and msg.get("content"):
                return parse_response(msg, output_schema_type)

        return result

    # ================================================================== #
    # Responses API wrapper (manages conversation state)                  #
    # ================================================================== #

    async def _call_responses_api(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        **extra: Any,
    ) -> _LLMResponse:
        """Call Responses API and manage conversation continuity state."""
        resp_tuple = await call_responses_api(
            self._client,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            default_temperature=self.temperature,
            top_p=top_p,
            stream=stream,
            response_format=response_format,
            last_response_id=self._last_response_id,
            max_retries=self.max_retries,
            async_mode=self.async_mode,
            logger=self._logger,
            **extra,
        )

        # If None returned, SDK too old — fall back to Chat Completions
        if resp_tuple[0] is None:
            return await call_chat_completions(
                self._client,
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                default_temperature=self.temperature,
                top_p=top_p,
                stream=stream,
                response_format=response_format,
                logit_bias=self.logit_bias,
                max_retries=self.max_retries,
                async_mode=self.async_mode,
                logger=self._logger,
                **extra,
            )

        result, new_response_id = resp_tuple
        self._last_response_id = new_response_id
        return result

    # ================================================================== #
    # Public: responses_call() — direct Responses API access              #
    # ================================================================== #

    async def responses_call(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        instructions: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        previous_response_id: str | None = None,
        stream: bool = False,
        include: list[str] | None = None,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        **Advanced** — direct access to OpenAI's Responses API.

        Most users should use ``call()`` which routes automatically.
        Use this method when you need:

        * Explicit control over ``previous_response_id``
        * The ``include`` parameter for rich annotations / sources
        * Raw Responses API response (not normalised to ``_LLMResponse``)

        Returns:
            Raw ``openai.types.responses.Response`` object.
        """
        return await responses_call_direct(
            self._client,
            model=model,
            input=input,
            tools=tools,
            instructions=instructions,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            previous_response_id=previous_response_id,
            stream=stream,
            include=include,
            tool_choice=tool_choice,
            async_mode=self.async_mode,
            **kwargs,
        )

    # ================================================================== #
    # Routing helpers                                                     #
    # ================================================================== #

    def _has_native_tools(self, tools: list[dict[str, Any]] | None) -> bool:
        """Return ``True`` if *tools* contains at least one native OpenAI tool."""
        if not tools:
            return False
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type", "")
            if (
                tool_type
                and tool_type != "function"
                and tool_type in self.NATIVE_TOOL_TYPES
            ):
                return True
        return False
