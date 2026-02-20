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
        response_format=MyPydanticModel  # Returns validated MyPydanticModel instance
    )
"""

from __future__ import annotations

import os
import asyncio
import time
import logging
import httpx
import json
import dataclasses
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints

from pydantic import BaseModel
import openai

from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq_openai.tools.openai_tool import NATIVE_TOOL_TYPES

logger = logging.getLogger(__name__)


# ======================================================================== #
# Lightweight response wrappers — match BaseLLM contract                    #
# ======================================================================== #

class _Choice(BaseModel):
    """Minimal wrapper so we match BaseLLM expectation."""
    message: Dict[str, Any]


class _LLMResponse(BaseModel):
    choices: List[_Choice]


# ======================================================================== #
# BaseOpenAI                                                                #
# ======================================================================== #

class BaseOpenAI(BaseLLM):
    """
    OpenAI client for ChatCompletion **and** Responses API.

    Supports both async and sync modes based on ``async_mode`` parameter
    (default True).

    **Open/Closed Principle** — the public ``call()`` interface is stable;
    new API backends (Responses API, future APIs) are added as private
    ``_call_via_*`` methods without modifying the contract.

    **Adaptive** — tool type registries and version constants are configurable
    so API changes are absorbed in one place.
    """

    # ------------------------------------------------------------------ #
    # Native-tool registry — used by _has_native_tools() for routing.     #
    # Subclasses or users can extend this set for new tool types.         #
    # ------------------------------------------------------------------ #
    NATIVE_TOOL_TYPES: frozenset = NATIVE_TOOL_TYPES

    # ================================================================== #
    # Tool spec conversion                                                #
    # ================================================================== #

    def _convert_tool_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert generic BaseTool spec to OpenAI format.

        * If spec already has a ``"type"`` key (native tool from
          ``OpenAITool``), it is returned **as-is** (pass-through).
        * Otherwise it is wrapped in the
          ``{"type": "function", "function": {...}}`` envelope that both
          the Chat Completions and Responses APIs accept.
        """
        # Already in OpenAI format (from OpenAITool or raw dict)
        if "type" in spec:
            return spec

        # Convert generic spec → OpenAI function-calling format
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
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.7,
        logit_bias: Optional[Dict[str, float]] = None,
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

        # Conversation continuity for Responses API (multi-turn tool loops)
        self._last_response_id: Optional[str] = None

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")

        # Create appropriate client based on mode
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

    # ================================================================== #
    # Model quirks                                                        #
    # ================================================================== #

    def _uses_max_completion_tokens(self, model: str) -> bool:
        """
        Some newer models reject ``max_tokens`` and require
        ``max_completion_tokens``.
        """
        m = (model or "").lower()
        return m.startswith("gpt-5")

    def _is_strict_defaults_model(self, model: str) -> bool:
        """
        Some models only accept default sampling parameters.
        """
        m = (model or "").lower()
        return m.startswith("gpt-5")

    # ================================================================== #
    # call() — unified entry point (smart routing)                        #
    # ================================================================== #

    async def call(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        response_format: Optional[Union[Type[BaseModel], Type, Dict[str, Any]]] = None,
        # --- New params (passed via LLMParams / AgentConfig / per-execute) ---
        seed: Optional[int] = None,
        n: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        service_tier: Optional[str] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        modalities: Optional[List[str]] = None,
        audio: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        store: Optional[bool] = None,
        truncation: Optional[str] = None,
        max_tool_calls: Optional[int] = None,
        safety_identifier: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        prompt_cache_retention: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call OpenAI API with optional structured output.

        **Routing is automatic** — callers never specify which API to use.
        If *tools* contains any native OpenAI tool the Responses API is used;
        otherwise the Chat Completions API is used.  The return type is
        always ``_LLMResponse`` (or a validated Pydantic instance when
        *response_format* is a class).

        Args:
            model: Model name (e.g. ``"gpt-4o"``).
            messages: Chat messages list.
            tools: Tool definitions (function-calling and/or native).
            tool_choice: Tool choice strategy.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            frequency_penalty: Frequency penalty (Chat Completions only).
            presence_penalty: Presence penalty (Chat Completions only).
            stop: Stop sequences (Chat Completions only).
            stream: Enable streaming.
            response_format: Structured output schema — can be a Pydantic
                model class, dataclass, dict, or ``{"type": "json_object"}``.
            seed: Fixed seed for deterministic outputs.
            n: Number of completions to generate.
            reasoning_effort: Reasoning depth (o-series/gpt-5 models).
            service_tier: Processing tier (auto/default/flex/priority).
            logprobs: Return log probabilities of output tokens.
            top_logprobs: Number of top tokens to return (0-20).
            parallel_tool_calls: Allow parallel function calling.
            modalities: Output types (["text"] or ["text", "audio"]).
            audio: Audio output config dict (voice, format).
            metadata: Key-value pairs for tracking.
            store: Whether to store response for retrieval.
            truncation: Context overflow handling (Responses API).
            max_tool_calls: Max built-in tool calls (Responses API).
            safety_identifier: Hashed end-user identifier.
            prompt_cache_key: Cache optimisation identifier.
            prompt_cache_retention: Cache retention policy.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ``_LLMResponse`` or validated structured output instance.
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
                openai_response_format = self._build_response_format(response_format)
                if openai_response_format:
                    api_response_format = openai_response_format
                    if isinstance(response_format, type):
                        output_schema_type = response_format

        # ---- 1b. Collect extra params (from LLMParams merge chain) ----
        extra: Dict[str, Any] = {}
        for _key, _val in [
            ("seed", seed), ("n", n), ("reasoning_effort", reasoning_effort),
            ("service_tier", service_tier), ("logprobs", logprobs),
            ("top_logprobs", top_logprobs), ("parallel_tool_calls", parallel_tool_calls),
            ("modalities", modalities), ("audio", audio), ("metadata", metadata),
            ("store", store), ("truncation", truncation), ("max_tool_calls", max_tool_calls),
            ("safety_identifier", safety_identifier),
            ("prompt_cache_key", prompt_cache_key),
            ("prompt_cache_retention", prompt_cache_retention),
        ]:
            if _val is not None:
                extra[_key] = _val
        extra.update(kwargs)

        # ---- 2. Route to appropriate backend ----
        if tools and self._has_native_tools(tools):
            raw_response = await self._call_via_responses_api(
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
            raw_response = await self._call_via_chat_completions(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=stream,
                response_format=api_response_format,
                **extra,
            )

        # ---- 3. Parse structured output if requested ----
        if output_schema_type is not None:
            if hasattr(raw_response, "choices") and raw_response.choices:
                msg_dict = raw_response.choices[0].message
                if isinstance(msg_dict, dict) and msg_dict.get("content"):
                    return self._parse_structured_response(msg_dict, output_schema_type)

        return raw_response

    # ================================================================== #
    # Backend: Chat Completions API (existing logic, extracted)           #
    # ================================================================== #

    async def _call_via_chat_completions(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> _LLMResponse:
        """
        Call OpenAI **Chat Completions API** (``chat.completions.create``).

        This is the original backend — handles function-calling tools and
        standard chat.  Extracted from ``call()`` without logic changes.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Sampling params — some models only accept defaults
        if not self._is_strict_defaults_model(model):
            payload.update(
                {
                    "temperature": temperature if temperature is not None else self.temperature,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                }
            )

        # Token limit parameter name depends on model family
        if self._uses_max_completion_tokens(model):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

        # Optional parameters
        if self.logit_bias is not None:
            payload["logit_bias"] = self.logit_bias
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if stop is not None:
            payload["stop"] = stop

        # Structured output
        if response_format is not None:
            payload["response_format"] = response_format

        # ---- New: pass through extra params from LLMParams ----
        _chat_keys = {
            "seed", "n", "logprobs", "top_logprobs", "parallel_tool_calls",
            "modalities", "audio", "metadata", "store", "service_tier",
            "reasoning_effort", "safety_identifier", "prompt_cache_key",
            "prompt_cache_retention",
        }
        for k, v in extra.items():
            if k in _chat_keys and v is not None:
                payload[k] = v

        attempt = 0
        while True:
            try:
                if self.async_mode:
                    if stream:
                        async for chunk in self._client.chat.completions.create(**payload):
                            first = chunk.choices[0].delta
                            return _LLMResponse(
                                choices=[_Choice(message=first.model_dump())]
                            )
                    else:
                        resp = await self._client.chat.completions.create(**payload)
                        msg_dict = resp.choices[0].message.model_dump()
                        return _LLMResponse(choices=[_Choice(message=msg_dict)])
                else:
                    loop = asyncio.get_event_loop()
                    if stream:
                        chunks = []
                        for chunk in self._client.chat.completions.create(**payload):
                            chunks.append(chunk)
                        if chunks:
                            first = chunks[0].choices[0].delta
                            return _LLMResponse(
                                choices=[_Choice(message=first.model_dump())]
                            )
                    else:
                        resp = await loop.run_in_executor(
                            None,
                            lambda: self._client.chat.completions.create(**payload),
                        )
                        msg_dict = resp.choices[0].message.model_dump()
                        return _LLMResponse(choices=[_Choice(message=msg_dict)])
            except openai.RateLimitError as e:
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error("Rate limit exceeded after %d retries: %s", self.max_retries, e)
                    raise
                backoff = 2 ** attempt
                self._logger.warning("Rate limit hit (%s); retry %d/%d in %ds", e, attempt, self.max_retries, backoff)
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except openai.APIConnectionError as e:
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error("Connection error after %d retries: %s", self.max_retries, e)
                    raise
                backoff = 2 ** attempt
                self._logger.warning("Connection error (%s); retry %d/%d in %ds", e, attempt, self.max_retries, backoff)
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except openai.AuthenticationError as e:
                self._logger.error("Authentication failed: %s", e)
                raise ValueError(f"Invalid API key or authentication failed: {e}") from e
            except openai.PermissionDeniedError as e:
                self._logger.error("Permission denied: %s", e)
                raise ValueError(f"Permission denied: {e}") from e
            except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
                if tool_choice is not None and "tool_choice" in payload:
                    self._logger.warning("Invalid request (retrying without tool_choice): %s", e)
                    payload.pop("tool_choice", None)
                    tool_choice = None
                    continue
                self._logger.error("Invalid request: %s", e)
                raise ValueError(f"Invalid request parameters: {e}") from e
            except openai.APIError as e:
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error("API error after %d retries: %s", self.max_retries, e)
                    raise
                backoff = 2 ** attempt
                self._logger.warning("API error (%s); retry %d/%d in %ds", e, attempt, self.max_retries, backoff)
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except httpx.HTTPError as e:
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error("HTTP error after %d retries: %s", self.max_retries, e)
                    raise
                backoff = 2 ** attempt
                self._logger.warning("HTTP error (%s); retry %d/%d in %ds", e, attempt, self.max_retries, backoff)
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except Exception as e:
                self._logger.error("Unexpected error during OpenAI call: %s", e, exc_info=True)
                raise

    # ================================================================== #
    # Backend: Responses API (native tools)                               #
    # ================================================================== #

    async def _call_via_responses_api(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        stream: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> _LLMResponse:
        """
        Call OpenAI **Responses API** (``responses.create``).

        Used automatically when native tools (web_search, code_interpreter,
        etc.) are detected.  Handles:

        * Message format conversion (Chat → Responses input)
        * Conversation continuity via ``previous_response_id``
        * Response normalisation → same ``_LLMResponse`` as Chat Completions
        * Retry with exponential backoff

        The Responses API executes native tools server-side and may also
        return ``function_call`` items for custom tools — these are
        normalised into the ``tool_calls`` list so the Agent's existing
        function-calling loop works unchanged.
        """
        # --- Guard: SDK version check ---
        if not hasattr(self._client, "responses"):
            self._logger.warning(
                "Responses API not available (requires openai>=1.66). "
                "Falling back to Chat Completions — native tools may not work."
            )
            return await self._call_via_chat_completions(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                response_format=response_format,
                **extra,
            )

        # --- Convert messages → Responses API format ---
        instructions, input_items = self._messages_to_responses_input(messages)

        # --- Build payload ---
        payload: Dict[str, Any] = {
            "model": model,
            "input": input_items,
        }

        if tools:
            payload["tools"] = tools
        if instructions:
            payload["instructions"] = instructions
        if self._last_response_id:
            payload["previous_response_id"] = self._last_response_id

        # Sampling params
        if not self._is_strict_defaults_model(model):
            effective_temp = temperature if temperature is not None else self.temperature
            payload["temperature"] = effective_temp
            payload["top_p"] = top_p

        payload["max_output_tokens"] = max_tokens

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if stream:
            payload["stream"] = True

        # Structured output → Responses API text.format config
        if response_format is not None:
            text_config = self._build_responses_text_config(response_format)
            if text_config:
                payload["text"] = text_config

        # ---- New: pass through extra params from LLMParams ----
        _resp_keys = {
            "reasoning_effort", "service_tier", "metadata", "store",
            "truncation", "max_tool_calls", "parallel_tool_calls",
            "safety_identifier", "seed",
        }
        # Responses API nests reasoning_effort under "reasoning" key
        if "reasoning_effort" in extra and extra["reasoning_effort"] is not None:
            payload["reasoning"] = {"effort": extra.pop("reasoning_effort")}
        for k, v in extra.items():
            if k in _resp_keys and v is not None:
                payload[k] = v

        # --- API call with retry ---
        attempt = 0
        while True:
            try:
                if self.async_mode:
                    resp = await self._client.responses.create(**payload)
                else:
                    loop = asyncio.get_event_loop()
                    resp = await loop.run_in_executor(
                        None,
                        lambda: self._client.responses.create(**payload),
                    )

                # Store response ID for conversation continuity
                resp_id = getattr(resp, "id", None)

                # Normalise → _LLMResponse
                result = self._normalize_responses_output(resp)

                # Track conversation state
                has_pending_tool_calls = bool(
                    result.choices
                    and isinstance(result.choices[0].message, dict)
                    and result.choices[0].message.get("tool_calls")
                )
                if has_pending_tool_calls:
                    # Function calls pending — Agent will execute them and
                    # call us again.  Keep response_id for continuation.
                    self._last_response_id = resp_id
                else:
                    # Conversation complete — reset.
                    self._last_response_id = None

                return result

            except openai.RateLimitError as e:
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error("Rate limit exceeded after %d retries: %s", self.max_retries, e)
                    raise
                backoff = 2 ** attempt
                self._logger.warning("Rate limit hit (%s); retry %d/%d in %ds", e, attempt, self.max_retries, backoff)
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except openai.APIConnectionError as e:
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error("Connection error after %d retries: %s", self.max_retries, e)
                    raise
                backoff = 2 ** attempt
                self._logger.warning("Connection error (%s); retry %d/%d in %ds", e, attempt, self.max_retries, backoff)
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except openai.AuthenticationError as e:
                self._logger.error("Authentication failed: %s", e)
                raise ValueError(f"Invalid API key or authentication failed: {e}") from e
            except openai.PermissionDeniedError as e:
                self._logger.error("Permission denied: %s", e)
                raise ValueError(f"Permission denied: {e}") from e
            except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
                # Retry once without tool_choice if the API rejects it
                if tool_choice is not None and "tool_choice" in payload:
                    self._logger.warning("Invalid request (retrying without tool_choice): %s", e)
                    payload.pop("tool_choice", None)
                    tool_choice = None
                    continue
                self._logger.error("Invalid request: %s", e)
                raise ValueError(f"Invalid request parameters: {e}") from e
            except openai.APIError as e:
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error("API error after %d retries: %s", self.max_retries, e)
                    raise
                backoff = 2 ** attempt
                self._logger.warning("API error (%s); retry %d/%d in %ds", e, attempt, self.max_retries, backoff)
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except httpx.HTTPError as e:
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error("HTTP error after %d retries: %s", self.max_retries, e)
                    raise
                backoff = 2 ** attempt
                self._logger.warning("HTTP error (%s); retry %d/%d in %ds", e, attempt, self.max_retries, backoff)
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except Exception as e:
                self._logger.error("Unexpected error during Responses API call: %s", e, exc_info=True)
                raise

    # ================================================================== #
    # Public: responses_call() — direct Responses API access              #
    # ================================================================== #

    async def responses_call(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        tools: Optional[List[Dict[str, Any]]] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        previous_response_id: Optional[str] = None,
        stream: bool = False,
        include: Optional[List[str]] = None,
        tool_choice: Optional[Any] = None,
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

        Raises:
            AttributeError: If the ``openai`` SDK is too old.
        """
        if not hasattr(self._client, "responses"):
            raise AttributeError(
                "Responses API requires openai>=1.66.  "
                "Upgrade with: pip install --upgrade openai"
            )

        payload: Dict[str, Any] = {
            "model": model,
            "input": input,
        }
        if tools:
            payload["tools"] = tools
        if instructions:
            payload["instructions"] = instructions
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if stream:
            payload["stream"] = True
        if include:
            payload["include"] = include
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        payload.update(kwargs)

        if self.async_mode:
            return await self._client.responses.create(**payload)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._client.responses.create(**payload),
            )

    # ================================================================== #
    # Routing helpers                                                     #
    # ================================================================== #

    def _has_native_tools(self, tools: Optional[List[Dict[str, Any]]]) -> bool:
        """
        Return ``True`` if *tools* contains at least one native OpenAI tool.

        Native tools are identified by their ``"type"`` field matching
        a value in ``NATIVE_TOOL_TYPES``.  Function-calling tools have
        ``type="function"`` which is **not** in the registry.
        """
        if not tools:
            return False
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type", "")
            if tool_type and tool_type != "function" and tool_type in self.NATIVE_TOOL_TYPES:
                return True
        return False

    # ================================================================== #
    # Message / response format conversion                                #
    # ================================================================== #

    def _messages_to_responses_input(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert Chat Completions ``messages`` to Responses API format.

        Returns:
            ``(instructions, input_items)`` where *instructions* is the
            extracted system message (or ``None``) and *input_items* is the
            list suitable for ``responses.create(input=...)``.

        Conversion rules:

        * ``system`` messages → ``instructions`` string.
        * ``user`` / ``assistant`` messages → input items.
        * ``tool`` messages → ``function_call_output`` items (for
          Responses API continuation after local tool execution).
        """
        instructions: Optional[str] = None
        input_items: List[Dict[str, Any]] = []

        if self._last_response_id:
            # Continuation: only send function call outputs (the Responses API
            # already has the earlier conversation in server-side state).
            for msg in messages:
                if msg.get("role") == "tool":
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": msg.get("tool_call_id", ""),
                            "output": str(msg.get("content", "")),
                        }
                    )
            return instructions, input_items

        # First call: convert full message history
        system_parts: List[str] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(str(content) if content else "")
            elif role == "user":
                input_items.append({"role": "user", "content": str(content) if content else ""})
            elif role == "assistant":
                input_items.append({"role": "assistant", "content": str(content) if content else ""})
            elif role == "tool":
                # Tool results from a prior turn (unlikely on first call,
                # but handled for completeness).
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": str(content) if content else "",
                    }
                )

        if system_parts:
            instructions = "\n".join(system_parts)

        return instructions, input_items

    def _normalize_responses_output(self, response: Any) -> _LLMResponse:
        """
        Convert a Responses API response into the ``_LLMResponse`` format
        used throughout the framework.

        Mapping:

        * ``output[type="message"]`` → ``message.content``
        * ``output[type="function_call"]`` → ``message.tool_calls``
        * Other output types (``web_search_call``, ``code_interpreter_call``,
          etc.) are collected into ``message._native_outputs`` for optional
          downstream inspection.
        """
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        native_outputs: List[Dict[str, Any]] = []

        output_items = getattr(response, "output", []) or []
        for item in output_items:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                # Extract text from content blocks
                for content_block in getattr(item, "content", []) or []:
                    block_type = getattr(content_block, "type", None)
                    if block_type == "output_text":
                        text = getattr(content_block, "text", "")
                        if text:
                            content_parts.append(text)

            elif item_type == "function_call":
                # Custom tool call → normalise to Chat Completions format
                tool_calls.append(
                    {
                        "id": getattr(item, "call_id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": getattr(item, "arguments", "{}"),
                        },
                    }
                )

            else:
                # Native tool output (web_search_call, code_interpreter_call,
                # file_search_call, etc.) — keep as metadata.
                try:
                    native_outputs.append(
                        item.model_dump() if hasattr(item, "model_dump") else {"type": item_type}
                    )
                except Exception:
                    native_outputs.append({"type": str(item_type)})

        # Build normalised message dict
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": "\n\n".join(content_parts) if content_parts else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if native_outputs:
            message["_native_outputs"] = native_outputs

        return _LLMResponse(choices=[_Choice(message=message)])

    def _build_responses_text_config(
        self,
        response_format: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a Chat Completions ``response_format`` dict into the
        Responses API ``text`` parameter.

        Chat Completions format::

            {"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}

        Responses API format::

            {"format": {"type": "json_schema", "name": ..., "schema": ...}}
        """
        if not isinstance(response_format, dict):
            return None

        fmt_type = response_format.get("type")

        if fmt_type == "json_schema":
            json_schema = response_format.get("json_schema", {})
            return {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.get("name", "response"),
                    "strict": json_schema.get("strict", True),
                    "schema": json_schema.get("schema", {}),
                }
            }

        if fmt_type == "json_object":
            return {"format": {"type": "json_object"}}

        # Unknown format — skip
        return None

    # ================================================================== #
    # STRUCTURED OUTPUT HELPERS (unchanged)                               #
    # ================================================================== #

    def _build_response_format(
        self,
        schema: Union[Type[BaseModel], Type, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a schema type to OpenAI's response_format parameter.

        Supports:
        - Pydantic models (recommended)
        - Python dataclasses
        - Raw dict with JSON schema
        - Simple ``{"type": "json_object"}`` mode
        """
        # Already a raw response_format dict
        if isinstance(schema, dict):
            return schema

        # Pydantic model
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()
            clean_schema = self._clean_schema_for_openai(json_schema)
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": clean_schema,
                },
            }

        # Dataclass
        if dataclasses.is_dataclass(schema) and isinstance(schema, type):
            json_schema = self._dataclass_to_schema(schema)
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": json_schema,
                },
            }

        # TypedDict or other types
        if hasattr(schema, "__annotations__"):
            json_schema = self._annotations_to_schema(schema)
            name = getattr(schema, "__name__", "response")
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": json_schema,
                },
            }

        self._logger.warning("Unknown schema type: %s, skipping response_format", type(schema))
        return None

    def _clean_schema_for_openai(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean Pydantic schema for OpenAI structured outputs."""
        import copy

        schema = copy.deepcopy(schema)

        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        defs = schema.pop("$defs", {})
        if defs:
            schema = self._inline_refs(schema, defs)

        schema.pop("title", None)
        schema.pop("$schema", None)
        schema.pop("description", None)

        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
            for key, prop in schema["properties"].items():
                if isinstance(prop, dict):
                    schema["properties"][key] = self._clean_property(prop, defs)

        return schema

    def _clean_property(self, prop: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a property schema recursively."""
        import copy

        prop = copy.deepcopy(prop)
        for key in ("title", "default", "description", "minimum", "maximum",
                     "minLength", "maxLength", "ge", "le"):
            prop.pop(key, None)

        if "anyOf" in prop:
            prop["anyOf"] = [self._clean_property(opt, defs) for opt in prop["anyOf"]]

        if prop.get("type") == "object" and "properties" in prop:
            prop["additionalProperties"] = False
            prop["required"] = list(prop["properties"].keys())
            for key, nested in prop["properties"].items():
                if isinstance(nested, dict):
                    prop["properties"][key] = self._clean_property(nested, defs)

        if prop.get("type") == "array" and "items" in prop:
            prop["items"] = self._clean_property(prop["items"], defs)

        return prop

    def _inline_refs(self, obj: Any, defs: Dict[str, Any]) -> Any:
        """Recursively inline $ref references."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in defs:
                        return self._inline_refs(
                            self._clean_schema_for_openai(defs[def_name]),
                            defs,
                        )
            return {k: self._inline_refs(v, defs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._inline_refs(item, defs) for item in obj]
        return obj

    def _dataclass_to_schema(self, cls: Type) -> Dict[str, Any]:
        """Convert a dataclass to JSON Schema."""
        hints = get_type_hints(cls)
        fields = dataclasses.fields(cls)

        properties = {}
        required = []

        for field in fields:
            prop_schema = self._type_to_schema(hints.get(field.name, str))
            properties[field.name] = prop_schema
            if (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                required.append(field.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _annotations_to_schema(self, cls: Type) -> Dict[str, Any]:
        """Convert type annotations to JSON Schema."""
        hints = get_type_hints(cls)
        properties = {}
        required = list(hints.keys())

        for name, hint in hints.items():
            properties[name] = self._type_to_schema(hint)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _type_to_schema(self, type_hint: Type) -> Dict[str, Any]:
        """Convert Python type hint to JSON Schema."""
        from typing import get_origin, get_args, Union as UnionType

        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is UnionType:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                inner = self._type_to_schema(non_none[0])
                return {"anyOf": [inner, {"type": "null"}]}

        if origin is list:
            item_schema = self._type_to_schema(args[0]) if args else {"type": "string"}
            return {"type": "array", "items": item_schema}

        if origin is dict:
            return {"type": "object"}

        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
        }
        return type_map.get(type_hint, {"type": "string"})

    def _parse_structured_response(
        self,
        message: Dict[str, Any],
        schema_type: Union[Type[BaseModel], Type, Dict[str, Any]],
    ) -> Any:
        """
        Parse LLM response into the requested structured type.

        Returns:
        - Pydantic model instance if schema_type is Pydantic
        - Dataclass instance if schema_type is dataclass
        - Dict if schema_type is dict/TypedDict
        """
        content = message.get("content", "")
        if not content:
            raise ValueError("LLM returned empty content for structured output")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {e}")

        if isinstance(schema_type, type) and issubclass(schema_type, BaseModel):
            return schema_type.model_validate(data)

        if dataclasses.is_dataclass(schema_type) and isinstance(schema_type, type):
            return schema_type(**data)

        if isinstance(schema_type, dict) or hasattr(schema_type, "__annotations__"):
            return data

        return data
