"""Anthropic tool **schema** shaping (parity with ``OpenAITool`` / converters).

OpenAI ships ``OpenAITool.web_search()`` etc.; this repo’s Anthropic alpha focuses on
**client-side** tools and ``to_anthropic_tool_definition`` for Messages API payloads.

Shows: OpenAI-shaped spec -> Claude ``tools[]`` entries (and optional native-type passthrough).

Run from ``src/providers/llms/anthropic``::

    uv run python examples/agents/08_anthropic_tool_schema.py

No API call — prints JSON-like dicts only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow ``from nucleusiq_anthropic …`` without installing wheel
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nucleusiq_anthropic import (  # noqa: E402
    NATIVE_TOOL_TYPES,
    to_anthropic_tool_definition,
)


def main() -> None:
    openai_style_fn = {
        "type": "function",
        "function": {
            "name": "lookup_capital",
            "description": "Return capital city.",
            "parameters": {
                "type": "object",
                "properties": {"country": {"type": "string"}},
                "required": ["country"],
            },
        },
    }

    anthropic_native_placeholder = {"type": "web_search"}

    print("=== OpenAI envelope -> Claude tool definition ===")
    print(json.dumps(to_anthropic_tool_definition(openai_style_fn), indent=2))

    print(
        "\n=== Non-'function' spec -> passthrough (Phase B/C fills NATIVE_TOOL_TYPES) ==="
    )
    print(
        json.dumps(to_anthropic_tool_definition(anthropic_native_placeholder), indent=2)
    )
    print(f"\n(NATIVE_TOOL_TYPES is currently empty): {sorted(NATIVE_TOOL_TYPES)}")


if __name__ == "__main__":
    main()
