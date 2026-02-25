"""Responses API backend.

Builds the payload, dispatches the call (sync or async), normalises
the response into ``_LLMResponse``, and manages conversation continuity
via ``previous_response_id``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from nucleusiq_openai._shared.model_config import is_strict_defaults_model
from nucleusiq_openai._shared.models import ResponsesFunctionTool
from nucleusiq_openai._shared.response_models import _LLMResponse
from nucleusiq_openai._shared.retry import call_with_retry
from nucleusiq_openai.nb_openai.response_normalizer import (
    build_responses_text_config,
    messages_to_responses_input,
    normalize_responses_output,
)


def _adapt_tools_for_responses(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Chat Completions tool format to Responses API format.

    Chat Completions nests function details::

        {
            "type": "function",
            "function": {"name": ..., "description": ..., "parameters": ...},
        }

    Responses API flattens them::

        {
            "type": "function",
            "name": ...,
            "description": ...,
            "parameters": ...,
            "strict": True,
        }

    Native tools (web_search_preview, code_interpreter, etc.) pass through unchanged.
    """
    adapted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            fn = tool["function"]
            flat_tool = ResponsesFunctionTool(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {}),
                strict=fn.get("strict", True),
            )
            adapted.append(flat_tool.model_dump())
        else:
            adapted.append(tool)
    return adapted


async def call_responses_api(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    default_temperature: float = 0.7,
    top_p: float = 1.0,
    stream: bool = False,
    response_format: dict[str, Any] | None = None,
    last_response_id: str | None = None,
    max_retries: int = 3,
    async_mode: bool = True,
    logger: logging.Logger | None = None,
    **extra: Any,
) -> tuple[_LLMResponse, str | None]:
    """Call OpenAI **Responses API** (``responses.create``).

    Returns:
        ``(response, new_response_id)`` — the normalised response and the
        response ID to use for conversation continuity (or ``None`` if the
        conversation is complete).
    """
    _log = logger or logging.getLogger(__name__)

    if not hasattr(client, "responses"):
        _log.warning(
            "Responses API not available (requires openai>=1.66). "
            "Falling back to Chat Completions — native tools may not work."
        )
        return None, None  # caller should fall back

    instructions, input_items = messages_to_responses_input(messages, last_response_id)
    serialized_input = [
        item.model_dump() if hasattr(item, "model_dump") else item
        for item in input_items
    ]

    payload: dict[str, Any] = {
        "model": model,
        "input": serialized_input,
    }

    if tools:
        payload["tools"] = _adapt_tools_for_responses(tools)
    if instructions:
        payload["instructions"] = instructions
    if last_response_id:
        payload["previous_response_id"] = last_response_id

    if not is_strict_defaults_model(model):
        effective_temp = temperature if temperature is not None else default_temperature
        payload["temperature"] = effective_temp
        payload["top_p"] = top_p

    payload["max_output_tokens"] = max_tokens

    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if stream:
        payload["stream"] = True

    if response_format is not None:
        text_config = build_responses_text_config(response_format)
        if text_config:
            payload["text"] = text_config

    if "reasoning_effort" in extra and extra["reasoning_effort"] is not None:
        payload["reasoning"] = {"effort": extra.pop("reasoning_effort")}
    _responses_extra_keys = {
        "service_tier",
        "metadata",
        "store",
        "truncation",
        "max_tool_calls",
        "parallel_tool_calls",
        "safety_identifier",
        "seed",
    }
    for k, v in extra.items():
        if k in _responses_extra_keys and v is not None:
            payload[k] = v

    _tool_choice_holder = {"value": tool_choice}

    def _on_bad_request() -> bool:
        if _tool_choice_holder["value"] is not None and "tool_choice" in payload:
            _log.warning("Invalid request — retrying without tool_choice")
            payload.pop("tool_choice", None)
            _tool_choice_holder["value"] = None
            return True
        return False

    if async_mode:

        async def _call() -> Any:
            return await client.responses.create(**payload)

        raw = await call_with_retry(
            _call,
            max_retries=max_retries,
            async_mode=True,
            logger=_log,
            on_bad_request=_on_bad_request,
        )
    else:

        def _call_sync() -> Any:
            return client.responses.create(**payload)

        raw = await call_with_retry(
            _call_sync,
            max_retries=max_retries,
            async_mode=False,
            logger=_log,
            on_bad_request=_on_bad_request,
        )

    resp_id = getattr(raw, "id", None)
    result = normalize_responses_output(raw)

    has_pending_tool_calls = bool(
        result.choices and result.choices[0].message.tool_calls
    )
    new_response_id = resp_id if has_pending_tool_calls else None

    return result, new_response_id


async def responses_call_direct(
    client: Any,
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
    async_mode: bool = True,
    **kwargs: Any,
) -> Any:
    """Direct access to OpenAI's Responses API (advanced usage).

    Most users should use ``BaseOpenAI.call()`` which routes automatically.

    Returns:
        Raw ``openai.types.responses.Response`` object.
    """
    if not hasattr(client, "responses"):
        raise AttributeError(
            "Responses API requires openai>=1.66.  "
            "Upgrade with: pip install --upgrade openai"
        )

    payload: dict[str, Any] = {
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

    if async_mode:
        return await client.responses.create(**payload)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: client.responses.create(**payload),
        )
