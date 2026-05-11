"""Build Ollama ``chat`` kwargs and normalise messages."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)


def _normalize_tool_call_entry(tc: Any) -> dict[str, Any]:
    """Coerce one tool call to OpenAI-style ``type`` + nested ``function``."""
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
    """Normalise assistant ``tool_calls`` for Ollama (OpenAI-compatible tool shape)."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        m = copy.deepcopy(msg)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            raw_tcs = m["tool_calls"]
            if isinstance(raw_tcs, list):
                m["tool_calls"] = [_normalize_tool_call_entry(tc) for tc in raw_tcs]
        out.append(m)
    return out


def build_options(
    *,
    max_output_tokens: int,
    temperature: float | None,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop: list[str] | None,
    seed: int | None,
) -> dict[str, Any]:
    """Map NucleusIQ-style sampling args to Ollama ``options``."""
    opts: dict[str, Any] = {"num_predict": max(1, max_output_tokens)}
    if temperature is not None:
        opts["temperature"] = float(temperature)
    opts["top_p"] = float(top_p)
    if frequency_penalty:
        opts["frequency_penalty"] = float(frequency_penalty)
    if presence_penalty:
        opts["presence_penalty"] = float(presence_penalty)
    if stop:
        opts["stop"] = stop if len(stop) > 1 else stop[0]
    if seed is not None:
        opts["seed"] = int(seed)
    return opts


ThinkLevel = Literal["low", "medium", "high"]


def build_chat_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    format_payload: str | dict[str, Any] | None,
    options: dict[str, Any],
    think: bool | ThinkLevel | None,
    keep_alive: float | str | None,
    stream: bool,
    tool_choice: Any,
) -> dict[str, Any]:
    """Assemble keyword args for ``Client.chat`` / ``AsyncClient.chat``."""
    if tool_choice is not None:
        logger.debug(
            "Ollama chat does not support tool_choice=%r; ignoring.", tool_choice
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": sanitize_messages(messages),
        "stream": stream,
        "options": options,
    }
    if tools:
        kwargs["tools"] = tools
    if format_payload is not None:
        kwargs["format"] = format_payload
    if think is not None:
        kwargs["think"] = think
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    return kwargs


def tool_arguments_to_json_string(arguments: Any) -> str:
    """Ollama tool calls use ``Mapping`` arguments; the framework expects JSON string."""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)
