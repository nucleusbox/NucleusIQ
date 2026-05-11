"""response_models helpers."""

from __future__ import annotations

from nucleusiq_ollama._shared.response_models import (
    AssistantMessage,
    ToolCall,
    ToolCallFunction,
)


def test_assistant_message_to_dict_includes_thinking_and_tools() -> None:
    m = AssistantMessage(
        content="c",
        thinking="why",
        tool_calls=[
            ToolCall(
                id="1",
                function=ToolCallFunction(name="f", arguments="{}"),
            )
        ],
    )
    d = m.to_dict()
    assert d["thinking"] == "why"
    assert len(d["tool_calls"]) == 1
