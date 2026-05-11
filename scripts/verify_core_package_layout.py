#!/usr/bin/env python3
"""
Layout checks for publishable Python packages in this monorepo.

1. **Core (`nucleusiq`)** — `tool.setuptools.packages` must list every subpackage
   under `src/nucleusiq/core/` that has `__init__.py`. Prevents wheels from
   omitting subpackages while editable installs still work.

2. **Hatch providers (`nucleusiq-openai`, `nucleusiq-gemini`, `nucleusiq-groq`, `nucleusiq-ollama`)** —
   `[tool.hatch.build.targets.wheel] packages = [...]` must name the on-disk
   import root (e.g. `nucleusiq_groq`). Every directory under that tree that
   contains `*.py` files must also contain `__init__.py` (classic packages only),
   so imports and wheels stay aligned.

Run: ``python scripts/verify_core_package_layout.py`` (no pip install required).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_TREE = REPO_ROOT / "src" / "nucleusiq" / "core"
CORE_PYPROJECT = REPO_ROOT / "src" / "nucleusiq" / "pyproject.toml"


@dataclass(frozen=True)
class HatchProvider:
    """One installable provider with a Hatch wheel `packages = [...]` stanza."""

    dist_name: str
    project_dir: Path  # directory containing pyproject.toml and the import root folder
    import_root: str  # top-level Python package name on disk (e.g. nucleusiq_groq)


# Keep in sync with RELEASE.md (wheel build list) and CI import-check.
HATCH_PROVIDERS: tuple[HatchProvider, ...] = (
    HatchProvider(
        "nucleusiq-openai",
        REPO_ROOT / "src" / "providers" / "llms" / "openai",
        "nucleusiq_openai",
    ),
    HatchProvider(
        "nucleusiq-gemini",
        REPO_ROOT / "src" / "providers" / "llms" / "gemini",
        "nucleusiq_gemini",
    ),
    HatchProvider(
        "nucleusiq-groq",
        REPO_ROOT / "src" / "providers" / "inference" / "groq",
        "nucleusiq_groq",
    ),
    HatchProvider(
        "nucleusiq-ollama",
        REPO_ROOT / "src" / "providers" / "inference" / "ollama",
        "nucleusiq_ollama",
    ),
)


def _discovered_core_packages() -> set[str]:
    found: set[str] = set()
    for init in CORE_TREE.rglob("__init__.py"):
        rel_dir = init.parent.relative_to(CORE_TREE)
        parts = rel_dir.parts
        if parts and parts[0] == "__pycache__":
            continue
        name = "nucleusiq" + ("" if not parts else "." + ".".join(parts))
        found.add(name)
    return found


def _declared_setuptools_packages() -> set[str]:
    text = CORE_PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r"packages\s*=\s*\[(.*?)\]\s*\n", text, re.DOTALL)
    if not m:
        print(
            "ERROR: could not find [tool.setuptools] packages = [...]",
            file=sys.stderr,
        )
        sys.exit(1)
    block = m.group(1)
    return {p.strip().strip('"').strip("'") for p in re.findall(r'"([^"]+)"', block)}


def _discovered_under(fs_root: Path, base: str) -> set[str]:
    """Dotted package names for every package directory under *fs_root*."""
    found: set[str] = set()
    if not (fs_root / "__init__.py").is_file():
        return found
    found.add(base)
    for init in fs_root.rglob("__init__.py"):
        rel_dir = init.parent.relative_to(fs_root)
        parts = rel_dir.parts
        if not parts:
            continue
        if parts[0] == "__pycache__":
            continue
        name = base + "." + ".".join(parts)
        found.add(name)
    return found


def _hatch_wheel_package_roots(pyproject: Path) -> list[str]:
    text = pyproject.read_text(encoding="utf-8")
    if "[tool.hatch.build.targets.wheel]" not in text:
        return []
    start = text.index("[tool.hatch.build.targets.wheel]")
    chunk = text[start : start + 800]
    m = re.search(r"packages\s*=\s*\[(.*?)\]", chunk, re.DOTALL)
    if not m:
        return []
    block = m.group(1)
    return [p for p in re.findall(r'"([^"]+)"', block) if p]


def _package_dirs_missing_init(fs_root: Path) -> list[str]:
    """Dirs that contain .py but lack __init__.py (classic layout only)."""
    bad: list[str] = []
    for subdir in fs_root.rglob("*"):
        if not subdir.is_dir() or "__pycache__" in subdir.parts:
            continue
        py_modules = [
            p
            for p in subdir.iterdir()
            if p.is_file() and p.suffix == ".py" and p.name != "__init__.py"
        ]
        if not py_modules:
            continue
        if not (subdir / "__init__.py").is_file():
            bad.append(str(subdir.relative_to(fs_root)))
    return sorted(bad)


def verify_core() -> None:
    discovered = _discovered_core_packages()
    declared = _declared_setuptools_packages()

    missing = sorted(discovered - declared)
    extra = sorted(declared - discovered)

    if missing:
        print(
            "ERROR: nucleusiq pyproject.toml `packages` is missing subpackages:",
            file=sys.stderr,
        )
        for p in missing:
            print(f"  + {p}", file=sys.stderr)
        print("\nAdd them under [tool.setuptools] packages = [...]", file=sys.stderr)
        sys.exit(1)

    if extra:
        print(
            "ERROR: nucleusiq pyproject lists packages not found under core/:",
            file=sys.stderr,
        )
        for p in extra:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    print(
        f"OK (core): {len(declared)} setuptools packages match {len(discovered)} on disk."
    )


def verify_hatch_providers() -> None:
    for hp in HATCH_PROVIDERS:
        ppt = hp.project_dir / "pyproject.toml"
        if not ppt.is_file():
            print(f"ERROR: missing pyproject.toml: {ppt}", file=sys.stderr)
            sys.exit(1)
        roots = _hatch_wheel_package_roots(ppt)
        if roots != [hp.import_root]:
            print(
                f"ERROR: {hp.dist_name}: expected hatch wheel packages = "
                f'["{hp.import_root}"], got {roots!r} ({ppt})',
                file=sys.stderr,
            )
            sys.exit(1)
        fs_root = hp.project_dir / hp.import_root
        if not fs_root.is_dir():
            print(
                f"ERROR: {hp.dist_name}: import root directory missing: {fs_root}",
                file=sys.stderr,
            )
            sys.exit(1)
        missing_init = _package_dirs_missing_init(fs_root)
        if missing_init:
            print(
                f"ERROR: {hp.dist_name}: package dirs with .py modules but no __init__.py:",
                file=sys.stderr,
            )
            for rel in missing_init:
                print(f"  - {hp.import_root}/{rel}", file=sys.stderr)
            sys.exit(1)
        discovered = _discovered_under(fs_root, hp.import_root)
        if len(discovered) < 1:
            print(
                f"ERROR: {hp.dist_name}: no packages discovered under {fs_root}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"OK ({hp.dist_name}): hatch wheel root `{hp.import_root}` — "
            f"{len(discovered)} subpackages, classic layout."
        )


def main() -> None:
    verify_core()
    verify_hatch_providers()


if __name__ == "__main__":
    main()
