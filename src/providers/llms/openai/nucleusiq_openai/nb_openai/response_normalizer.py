"""Message and response format conversion between Chat Completions and Responses API.

Pure functions — no dependency on BaseOpenAI state (except
``last_response_id`` which is passed as a parameter).
"""

from __future__ import annotations

from typing import Any

from nucleusiq_openai._shared.models import (
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

InputItem = MessageInputItem | FunctionCallOutput


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
    * ``user`` / ``assistant`` messages → input items.
    * ``tool`` messages → ``function_call_output`` items.
    """
    instructions: str | None = None
    input_items: list[InputItem] = []

    if last_response_id:
        for msg in messages:
            if msg.get("role") == "tool":
                input_items.append(
                    FunctionCallOutput(
                        call_id=msg.get("tool_call_id", ""),
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
        elif role == "user":
            input_items.append(
                MessageInputItem(
                    role="user",
                    content=str(content) if content else "",
                )
            )
        elif role == "assistant":
            input_items.append(
                MessageInputItem(
                    role="assistant",
                    content=str(content) if content else "",
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
