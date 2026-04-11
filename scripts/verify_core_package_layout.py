#!/usr/bin/env python3
"""
Fail if setuptools `packages` in src/nucleusiq/pyproject.toml does not list
every subpackage under core/ that contains __init__.py.

Prevents PyPI wheels from omitting subpackages (e.g. nucleusiq.tools.builtin)
while editable installs still work.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "src" / "nucleusiq" / "core"
PYPROJECT = REPO_ROOT / "src" / "nucleusiq" / "pyproject.toml"


def _discovered_packages() -> set[str]:
    found: set[str] = set()
    for init in CORE_ROOT.rglob("__init__.py"):
        rel_dir = init.parent.relative_to(CORE_ROOT)
        parts = rel_dir.parts
        if parts and parts[0] == "__pycache__":
            continue
        name = "nucleusiq" + ("" if not parts else "." + ".".join(parts))
        found.add(name)
    return found


def _declared_packages() -> set[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r"packages\s*=\s*\[(.*?)\]\s*\n", text, re.DOTALL)
    if not m:
        print("ERROR: could not find [tool.setuptools] packages = [...]", file=sys.stderr)
        sys.exit(1)
    block = m.group(1)
    return {p.strip().strip('"').strip("'") for p in re.findall(r'"([^"]+)"', block)}


def main() -> None:
    discovered = _discovered_packages()
    declared = _declared_packages()

    missing = sorted(discovered - declared)
    extra = sorted(declared - discovered)

    if missing:
        print("ERROR: pyproject.toml `packages` is missing subpackages that exist on disk:", file=sys.stderr)
        for p in missing:
            print(f"  + {p}", file=sys.stderr)
        print("\nAdd them under [tool.setuptools] packages = [...]", file=sys.stderr)
        sys.exit(1)

    if extra:
        print("WARNING: pyproject.toml lists packages not found under core/ (stale entries?):", file=sys.stderr)
        for p in extra:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {len(declared)} packages in pyproject.toml match {len(discovered)} on disk.")


if __name__ == "__main__":
    main()
