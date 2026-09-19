"""Credential helpers for paid search adapters."""

from __future__ import annotations

import os

from nucleusiq.tools.errors import ToolValidationError


def env_or(explicit: str | None, *names: str) -> str | None:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def require_config(
    *,
    provider: str,
    fields: dict[str, str | None],
    how: str,
) -> dict[str, str]:
    """Return stripped required fields or raise a constructor-time error."""
    missing = [
        name for name, value in fields.items() if not (value and str(value).strip())
    ]
    if not missing:
        return {name: str(value).strip() for name, value in fields.items() if value}
    needed = ", ".join(missing)
    raise ToolValidationError(
        f"{provider} web search requires {needed}. {how}",
        tool_name="web_search",
    )
