"""Normalize tool return values for context / compaction (shared by modes)."""

from __future__ import annotations

import json
from typing import Any


def tool_result_to_context_string(tool_result: Any) -> str:
    """Match streaming/base_mode: keep str tools as-is; JSON-serialize others."""
    if isinstance(tool_result, str):
        return tool_result
    try:
        return json.dumps(tool_result, ensure_ascii=False)
    except TypeError:
        return str(tool_result)
