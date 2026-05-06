"""Tool result serialization for context / compaction."""

from __future__ import annotations

import json

from nucleusiq.agents.modes.tool_payload import tool_result_to_context_string


def test_string_tool_result_unchanged() -> None:
    assert tool_result_to_context_string("hello") == "hello"


def test_dict_tool_result_is_json() -> None:
    s = tool_result_to_context_string({"a": 1})
    assert json.loads(s) == {"a": 1}
