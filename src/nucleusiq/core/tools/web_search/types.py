"""Shared types for web-search adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    """One web-search result, normalized across providers."""

    title: str
    url: str
    snippet: str
