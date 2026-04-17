"""Message and response format conversion between Chat Completions and Responses API.

Pure functions — no dependency on BaseOpenAI state (except
``last_response_id`` which is passed as a parameter).
"""

from __future__ import annotations

from typing import Any

from nucleusiq_openai._shared.models import (
    FunctionCallInput,
    FunctionCallOutput,
    JsonSchemaFormat,
    MessageInputItem,
    TextFormatConfig,
)
from nucleusiq_openai._shared.response_models import (
    AssistantMessage,
    ToolCall,
    ToolCallFunction,
    _Choice,
    _LLMResponse,
)

InputItem = MessageInputItem | FunctionCallInput | FunctionCallOutput


def _convert_content_part_for_responses(part: dict[str, Any]) -> dict[str, Any]:
    """Convert a single Chat Completions content part to Responses API format.

    Mapping (per OpenAI API docs):
        ``text``      → ``input_text``
        ``file``      → ``input_file``  (flatten nested ``file`` dict)
        ``input_file`` → pass-through   (already Responses format, e.g. file_url)
        ``image_url`` → ``input_image`` (flatten nested ``image_url`` dict)
    """
    part_type = part.get("type", "")

    if part_type == "text":
        return {"type": "input_text", "text": part.get("text", "")}

    if part_type == "file":
        inner = part.get("file", {})
        result: dict[str, Any] = {"type": "input_file"}
        if "file_id" in inner:
            result["file_id"] = inner["file_id"]
        if "filename" in inner:
            result["filename"] = inner["filename"]
        if "file_data" in inner:
            result["file_data"] = inner["file_data"]
        return result

    if part_type == "input_file":
        return part

    if part_type == "image_url":
        img = part.get("image_url", {})
        url = img.get("url", "") if isinstance(img, dict) else str(img)
        result_img: dict[str, Any] = {"type": "input_image", "image_url": url}
        if isinstance(img, dict) and "detail" in img:
            result_img["detail"] = img["detail"]
        return result_img

    if part_type == "input_text":
        return part
    if part_type == "input_image":
        return part

    return part


def _convert_content_for_responses(
    content: str | list[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    """Convert multimodal content arrays from Chat Completions to Responses format."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""
    return [_convert_content_part_for_responses(p) for p in content]


def messages_to_responses_input(
    messages: list[dict[str, Any]],
    last_response_id: str | None,
) -> tuple[str | None, list[InputItem]]:
    """Convert Chat Completions ``messages`` to Responses API format.

    Returns:
        ``(instructions, input_items)`` where *instructions* is the
        extracted system message (or ``None``) and *input_items* is the
        list suitable for ``responses.create(input=...)``.

    Conversion rules:

    * ``system`` messages → ``instructions`` string.
    * ``user`` / ``assistant`` messages → input items.  Multimodal
      content arrays are converted (``text`` → ``input_text``,
      ``file`` → ``input_file``, ``image_url`` → ``input_image``).
    * ``tool`` messages → ``function_call_output`` items.
    """
    instructions: str | None = None
    input_items: list[InputItem] = []

    if last_response_id:
        # Chain mode: the Responses API looks up call_ids in the ancestral
        # chain of ``previous_response_id``.  Only send tool outputs whose
        # matching function_call belongs to the CURRENT head of the chain
        # (i.e. the most recent assistant's tool_calls).  Re-sending older
        # tool outputs from prior turns will fail with
        # ``No tool call found for function call output with call_id ...``
        # whenever the chain was reset (e.g. a retry that triggered a full
        # replay and started a fresh chain on the next response).
        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        valid_call_ids: set[str] = set()
        if last_assistant_idx >= 0:
            for tc in messages[last_assistant_idx].get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                cid = tc.get("id", "")
                if cid:
                    valid_call_ids.add(cid)

        for msg in messages[last_assistant_idx + 1 :]:
            if msg.get("role") != "tool":
                continue
            cid = msg.get("tool_call_id", "")
            if valid_call_ids and cid not in valid_call_ids:
                continue
            input_items.append(
                FunctionCallOutput(
                    call_id=cid,
                    output=str(msg.get("content", "")),
                )
            )
        return instructions, input_items

    system_parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            system_parts.append(str(content) if content else "")
        elif role in ("user", "assistant"):
            resolved_content = _convert_content_for_responses(content)
            input_items.append(MessageInputItem(role=role, content=resolved_content))
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        tc_name = fn.get("name", "")
                        tc_args = fn.get("arguments", "{}")
                    else:
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("arguments", "{}")
                    input_items.append(
                        FunctionCallInput(
                            call_id=tc.get("id", ""),
                            name=tc_name,
                            arguments=tc_args,
                        )
                    )
        elif role == "tool":
            input_items.append(
                FunctionCallOutput(
                    call_id=msg.get("tool_call_id", ""),
                    output=str(content) if content else "",
                )
            )

    if system_parts:
        instructions = "\n".join(system_parts)

    # Defensive: drop orphan FunctionCallOutput items whose call_id has no
    # matching FunctionCallInput in this request.  The framework's
    # compactors preserve assistant+tool atomic groups, so this should
    # never trigger in practice — but filtering here guarantees the
    # Responses API never rejects the request over an upstream history
    # anomaly (compaction bug, manual history edit, etc.).
    known_call_ids = {
        item.call_id for item in input_items if isinstance(item, FunctionCallInput)
    }
    input_items = [
        item
        for item in input_items
        if not (
            isinstance(item, FunctionCallOutput) and item.call_id not in known_call_ids
        )
    ]

    return instructions, input_items


def normalize_responses_output(response: Any) -> _LLMResponse:
    """Convert a Responses API response into the ``_LLMResponse`` format.

    Mapping:

    * ``output[type="message"]`` → ``message.content``
    * ``output[type="function_call"]`` → ``message.tool_calls``
    * Other output types → ``message._native_outputs``
    """
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    native_outputs: list[dict[str, Any]] = []

    output_items = getattr(response, "output", []) or []
    for item in output_items:
        item_type = getattr(item, "type", None)

        if item_type == "message":
            for content_block in getattr(item, "content", []) or []:
                block_type = getattr(content_block, "type", None)
                if block_type == "output_text":
                    text = getattr(content_block, "text", "")
                    if text:
                        content_parts.append(text)

        elif item_type == "function_call":
            tool_calls.append(
                ToolCall(
                    id=getattr(item, "call_id", ""),
                    function=ToolCallFunction(
                        name=getattr(item, "name", ""),
                        arguments=getattr(item, "arguments", "{}"),
                    ),
                )
            )

        else:
            try:
                native_outputs.append(
                    item.model_dump()
                    if hasattr(item, "model_dump")
                    else {"type": item_type}
                )
            except Exception:
                native_outputs.append({"type": str(item_type)})

    message = AssistantMessage(
        content="\n\n".join(content_parts) if content_parts else None,
        tool_calls=tool_calls or None,
        native_outputs=native_outputs or None,
    )
    return _LLMResponse(choices=[_Choice(message=message)])


def build_responses_text_config(
    response_format: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert a Chat Completions ``response_format`` to Responses API ``text`` param.

    Chat Completions::

        {"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}

    Responses API::

        {"format": {"type": "json_schema", "name": ..., "schema": ...}}
    """
    if not isinstance(response_format, dict):
        return None

    fmt_type = response_format.get("type")

    if fmt_type == "json_schema":
        json_schema = response_format.get("json_schema", {})
        config = TextFormatConfig(
            format=JsonSchemaFormat(
                name=json_schema.get("name", "response"),
                strict=json_schema.get("strict", True),
                schema=json_schema.get("schema", {}),
            )
        )
        return config.model_dump(by_alias=True)

    if fmt_type == "json_object":
        return {"format": {"type": "json_object"}}

    return None
