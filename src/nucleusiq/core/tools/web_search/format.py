"""Format search hits for the model (shared by every backend)."""

from __future__ import annotations

from typing import Any

from nucleusiq.tools.web_search.types import SearchHit

DEFAULT_MAX_RESULTS = 5
HARD_MAX_RESULTS = 10


def clamp_max_results(value: Any, default: int = DEFAULT_MAX_RESULTS) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, HARD_MAX_RESULTS))


def format_results(query: str, hits: list[SearchHit]) -> str:
    if not hits:
        return f'No web results for "{query}". Try a simpler or more specific query.'

    lines = [f'Search results for "{query}":', ""]
    for i, hit in enumerate(hits, start=1):
        title = (hit.title or "(untitled)").strip()
        lines.append(f"{i}. {title}")
        if hit.url:
            lines.append(f"   URL: {hit.url}")
        if hit.snippet:
            lines.append(f"   {hit.snippet.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip()
