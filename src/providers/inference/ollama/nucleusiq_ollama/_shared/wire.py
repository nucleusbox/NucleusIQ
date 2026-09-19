"""Build Ollama ``chat`` kwargs and normalise messages."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)


def _arguments_to_mapping(arguments: Any) -> dict[str, Any]:
    """Ollama's ``Message.ToolCall.Function.arguments`` is a ``Mapping``.

    The framework (and every OpenAI-shaped provider) stores tool-call
    arguments as a JSON *string*, so assistant turns replayed from history
    must be decoded before they go back over the wire.  Undecodable strings
    are preserved under ``_raw`` rather than dropped, so the model still
    sees what it asked for.
    """
    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except (ValueError, TypeError):
            return {"_raw": arguments}
        return decoded if isinstance(decoded, dict) else {"_raw": decoded}
    try:
        return dict(arguments)
    except (TypeError, ValueError):
        return {"_raw": str(arguments)}


def _normalize_tool_call_entry(tc: Any) -> dict[str, Any]:
    """Coerce one tool call to OpenAI-style ``type`` + nested ``function``.

    ``function.arguments`` always comes out as a mapping (see
    :func:`_arguments_to_mapping`).
    """
    if not isinstance(tc, dict):
        return {"type": "function", "function": {"name": "", "arguments": {}}}
    if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
        out = copy.deepcopy(tc)
        out["function"]["arguments"] = _arguments_to_mapping(
            out["function"].get("arguments")
        )
        return out
    fn = tc.get("function")
    if isinstance(fn, dict) and "name" in fn:
        out = {
            "type": "function",
            "function": {
                "name": fn["name"],
                "arguments": _arguments_to_mapping(
                    fn.get("arguments", tc.get("arguments"))
                ),
            },
        }
        if tc.get("id") is not None:
            out["id"] = tc["id"]
        return out
    out = {
        "type": "function",
        "function": {
            "name": tc.get("name", ""),
            "arguments": _arguments_to_mapping(tc.get("arguments")),
        },
    }
    if tc.get("id") is not None:
        out["id"] = tc["id"]
    return out


def _split_data_url(url: str) -> str | None:
    """Return the base64 payload from a ``data:image/...;base64,<data>`` URL.

    Returns ``None`` if the URL is not a base64-encoded data URL.  Ollama's
    chat API expects raw base64 strings (no ``data:`` prefix) in
    ``message.images``.
    """
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    comma = url.find(",", 5)
    if comma < 0:
        return None
    meta = url[5:comma].strip().lower()
    if ";base64" not in meta:
        return None
    return url[comma + 1 :].strip() or None


def _extract_text_and_images(
    content: Any,
) -> tuple[str | None, list[str]]:
    """Split OpenAI-style multimodal content into (text, base64_images).

    Accepts:

    * ``str`` — passed through as text, no images.
    * ``list`` of OpenAI-style parts (``{"type": "text", "text": ...}`` and
      ``{"type": "image_url", "image_url": {"url": ...}}``) — text parts
      are joined; image parts whose URL is a base64 data URL are decoded
      into raw base64 strings for Ollama's ``message.images`` field.
    * ``None`` — returns ``(None, [])``.

    HTTP(S) image URLs are skipped with a warning — Ollama needs the bytes
    inline (no implicit download).  Callers that want HTTP image support
    should pre-encode them as data URLs (the core attachment system already
    handles this for ``Attachment.IMAGE`` parts).
    """
    if content is None:
        return None, []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    text_parts: list[str] = []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        pt = part.get("type")
        if pt == "text":
            text_val = part.get("text")
            if isinstance(text_val, str) and text_val:
                text_parts.append(text_val)
        elif pt == "image_url":
            iu = part.get("image_url")
            url = iu.get("url", "") if isinstance(iu, dict) else str(iu or "")
            decoded = _split_data_url(url)
            if decoded:
                images.append(decoded)
            elif isinstance(url, str) and url.startswith(("http://", "https://")):
                logger.warning(
                    "Ollama wire: skipping HTTP image_url part (no inline "
                    "download support); pre-encode as a data: URL instead. "
                    "url=%s",
                    url,
                )
        elif pt == "image":
            # Raw passthrough: ``{"type": "image", "data": "<base64>"}``
            data = part.get("data")
            if isinstance(data, str) and data:
                images.append(data)

    joined = "\n".join(t for t in text_parts if t).strip() or None
    return joined, images


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise messages for Ollama.

    * Assistant ``tool_calls`` are converted to OpenAI-compatible shape.
    * User / system messages whose ``content`` is an OpenAI-style multimodal
      part list are split into a plain ``content`` string plus an ``images``
      list of base64 strings (Ollama's vision-message shape).  Multimodal
      messages without any text parts get an empty ``content`` (some models
      accept this; others ignore it — both behaviours are correct).
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        m = copy.deepcopy(msg)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            raw_tcs = m["tool_calls"]
            if isinstance(raw_tcs, list):
                m["tool_calls"] = [_normalize_tool_call_entry(tc) for tc in raw_tcs]

        role = m.get("role")
        if role in ("user", "system"):
            text, images = _extract_text_and_images(m.get("content"))
            if images or isinstance(m.get("content"), list):
                m["content"] = text or ""
                if images:
                    # Merge with any existing ``images`` already on the
                    # message (callers may pass them directly).
                    existing = m.get("images") or []
                    if isinstance(existing, list):
                        m["images"] = list(existing) + images
                    else:
                        m["images"] = images
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
