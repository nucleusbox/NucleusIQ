"""Request sanitization for Groq's OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import copy
from typing import Any

_UNSUPPORTED_CHAT_KEYS = frozenset({"logprobs", "logit_bias", "top_logprobs"})


def _normalize_tool_call_entry(tc: Any) -> dict[str, Any]:
    """Coerce one tool call to Groq's OpenAI-compatible shape (type + function)."""
    if not isinstance(tc, dict):
        return {"type": "function", "function": {"name": "", "arguments": "{}"}}
    if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
        return copy.deepcopy(tc)
    fn = tc.get("function")
    if isinstance(fn, dict) and "name" in fn:
        out: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": fn["name"],
                "arguments": fn.get("arguments", tc.get("arguments", "{}")),
            },
        }
        if tc.get("id") is not None:
            out["id"] = tc["id"]
        return out
    out = {
        "type": "function",
        "function": {
            "name": tc.get("name", ""),
            "arguments": tc.get("arguments", "{}"),
        },
    }
    if tc.get("id") is not None:
        out["id"] = tc["id"]
    return out


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deep-ish copy of *messages* with Groq-incompatible fields removed.

    Groq rejects ``messages[].name`` (OpenAI compat doc).

    NucleusIQ serialises assistant ``tool_calls`` in a flat canonical form
    ``{id, name, arguments}``; Groq requires ``type: function`` and a nested
    ``function`` object — we coerce that here.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        m = copy.deepcopy(msg)
        m.pop("name", None)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            raw_tcs = m["tool_calls"]
            if isinstance(raw_tcs, list):
                m["tool_calls"] = [_normalize_tool_call_entry(tc) for tc in raw_tcs]
        out.append(m)
    return out


def filter_unsupported_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs Groq documents as unsupported for Chat Completions."""
    return {k: v for k, v in kwargs.items() if k not in _UNSUPPORTED_CHAT_KEYS}


def validate_sampling_count(n: Any) -> None:
    """Groq requires ``n == 1`` when ``n`` is supplied."""
    from nucleusiq.llms.errors import InvalidRequestError

    if n is None:
        return
    if n != 1:
        raise InvalidRequestError.from_provider_error(
            provider="groq",
            message="Groq Chat Completions requires n=1 when n is supplied.",
            status_code=400,
            original_error=None,
        )


def build_chat_completion_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop: list[str] | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    response_format: dict[str, Any] | None,
    parallel_tool_calls: bool | None,
    seed: int | None,
    user: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Assemble kwargs for ``client.chat.completions.create``."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": sanitize_messages(messages),
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if stop:
        payload["stop"] = stop
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if response_format is not None:
        payload["response_format"] = response_format
    if parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = parallel_tool_calls
    if seed is not None:
        payload["seed"] = seed
    if user is not None:
        payload["user"] = user

    filtered_extra = filter_unsupported_kwargs(dict(extra))
    n_val = filtered_extra.pop("n", None)
    validate_sampling_count(n_val)
    payload.update(filtered_extra)
    return payload
