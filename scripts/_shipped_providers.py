"""Shipped Hatch providers — the set both verify scripts must stay aligned with.

``src/providers/dbs/`` (chroma, pinecone) is Pre-Alpha and is not published
or CI-gated, so it is excluded until those packages ship.

Used by ``verify_core_package_layout.py`` and
``verify_dependency_completeness.py``. Adding a new first-party provider under
``llms/``, ``inference/``, or ``tools/`` without updating those scripts will
fail both of them, and the unit tests that assert the same set.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Roots that hold published (or release-gated) Hatch packages.
_SHIPPED_ROOTS = (
    REPO_ROOT / "src" / "providers" / "llms",
    REPO_ROOT / "src" / "providers" / "inference",
    REPO_ROOT / "src" / "providers" / "tools",
)


def discover_shipped_provider_dirs() -> list[Path]:
    """Directories that contain a ``pyproject.toml`` under shipped roots."""
    found: list[Path] = []
    for root in _SHIPPED_ROOTS:
        if not root.is_dir():
            continue
        for ppt in sorted(root.rglob("pyproject.toml")):
            if any(part.startswith(".") or part == "__pycache__" for part in ppt.parts):
                continue
            found.append(ppt.parent)
    return found


def discover_shipped_provider_relpaths() -> list[str]:
    """Repo-relative POSIX paths, e.g. ``src/providers/inference/openai_compatible``."""
    return [
        p.relative_to(REPO_ROOT).as_posix() for p in discover_shipped_provider_dirs()
    ]
