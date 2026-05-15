"""Tests for :mod:`nucleusiq_anthropic._shared.wire`."""

from __future__ import annotations

import pytest
from nucleusiq_anthropic._shared.wire import (
    anthropic_tool_choice,
    flatten_tools,
    split_system,
    translate_messages,
)


def test_split_system_joins_chunks() -> None:
    msgs = [
        {"role": "system", "content": "A"},
        {"role": "system", "content": "B"},
        {"role": "user", "content": "Hello"},
    ]
    system, rest = split_system(msgs)

    assert system == "A\n\nB"

    assert rest == [{"role": "user", "content": "Hello"}]


@pytest.mark.parametrize(
    ("tc_raw", "expected"),
    [
        (None, None),
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
        (
            {"type": "function", "function": {"name": "w"}},
            {"type": "tool", "name": "w"},
        ),
    ],
)
def test_anthropic_tool_choice(tc_raw, expected):

    assert anthropic_tool_choice(tc_raw) == expected


def test_translate_assistant_tools_and_results() -> None:

    msgs = [
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [{"id": "t1", "name": "x", "arguments": '{"z": true}'}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": '{"ok":1}'},
    ]

    _, out = translate_messages(msgs)

    assistant = next(m for m in out if m["role"] == "assistant")

    assert any(b.get("type") == "tool_use" for b in assistant["content"])

    user_tr = next(
        m for m in out if m["role"] == "user" and isinstance(m["content"], list)
    )

    blocks = user_tr["content"]

    assert blocks[0]["type"] == "tool_result"

    assert blocks[0]["tool_use_id"] == "t1"


def test_flatten_tools_preserves_native_shape() -> None:

    defs = [{"type": "web_search"}]

    flat = flatten_tools(defs)

    assert flat == defs
