"""Coverage for structured response helpers."""

from __future__ import annotations

from nucleusiq_anthropic._shared.response_models import (
    AssistantMessage,
    ToolCall,
    ToolCallFunction,
)


def test_assistant_to_dict_without_tools() -> None:

    md = AssistantMessage(content="hey")
    d = md.to_dict()
    assert d == {"role": "assistant", "content": "hey"}


def test_assistant_to_dict_with_tools() -> None:

    msg = AssistantMessage(
        content=None,
        tool_calls=[
            ToolCall(
                id="1",
                function=ToolCallFunction(name="f", arguments="{}"),
            )
        ],
    )

    dumped = msg.to_dict()
    assert dumped["tool_calls"][0]["function"]["name"] == "f"
