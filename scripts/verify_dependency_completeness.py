#!/usr/bin/env python3
"""Verify every package can be imported from its declared dependencies alone.

Why this exists
---------------
``nucleusiq-openai`` imported ``httpx`` at module scope in
``_shared/retry.py`` while declaring only ``nucleusiq``, ``openai`` and
``tiktoken``.  That worked for as long as the ``openai`` SDK happened to pull
``httpx`` in transitively.  When ``openai`` 3.x stopped shipping it, ``import
nucleusiq_openai`` began raising ``ModuleNotFoundError`` on a fresh install —
and because the floor was an unbounded ``openai>=1.0``, every new install got
the broken combination.

No existing CI job could catch that.  ``test-*``, ``import-check`` and
``type-check`` all install siblings and dev tooling into a single environment,
and any one of those happens to supply the missing module.  An undeclared
import only fails in an environment that contains the package under test and
nothing else, so that is what this builds: one throwaway virtualenv per
package, holding that package plus its declared dependencies, then importing
the public API.

It is deliberately slow and deliberately not merged into another job — the
isolation *is* the test.

Usage
-----
    python scripts/verify_dependency_completeness.py            # all packages
    python scripts/verify_dependency_completeness.py openai     # subset match

Exit code is non-zero if any package cannot be imported from its own metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _shipped_providers import REPO_ROOT, discover_shipped_provider_relpaths

ROOT = REPO_ROOT
CORE = ROOT / "src" / "nucleusiq"

# Package directory -> statement importing that package's advertised surface.
# Keep these in step with the `import-check` job in .github/workflows/ci.yml;
# both answer "does the public API import?", just in different environments.
CASES: dict[str, str] = {
    "src/providers/llms/openai": (
        "from nucleusiq_openai import BaseOpenAI, OpenAITool, OpenAILLMParams"
    ),
    "src/providers/llms/gemini": (
        "from nucleusiq_gemini import BaseGemini, GeminiTool, GeminiLLMParams"
    ),
    "src/providers/llms/anthropic": (
        "from nucleusiq_anthropic import ("
        "    BaseAnthropic, AnthropicLLMParams, NATIVE_TOOL_TYPES,"
        "    to_anthropic_tool_definition, build_anthropic_output_config,"
        "    parse_anthropic_response,"
        ")"
    ),
    "src/providers/inference/groq": (
        "from nucleusiq_groq import BaseGroq, GroqLLMParams"
    ),
    "src/providers/inference/ollama": (
        "from nucleusiq_ollama import BaseOllama, OllamaLLMParams, ThinkLevel"
    ),
    "src/providers/inference/openai_compatible": (
        "from nucleusiq_openai_compatible import ("
        "    BaseOpenAICompatible, OpenAICompatibleLLM, OpenAICompatibleLLMParams,"
        "    AuthStrategy, BearerAuth, HeaderAuth, NoAuth, build_auth,"
        "    ENGINE_PRESETS, EngineProfile, known_engines,"
        "    DropPolicy, ErrorPolicy, PromptPolicy,"
        "    NATIVE_TOOL_TYPES, ResolvedConfig, ValidationReport,"
        ")"
    ),
    "src/providers/tools/mcp": (
        "from nucleusiq_mcp import ("
        "    MCPTool, MCPBoundTool, MCPSession, MCPServerConfig,"
        "    MCPSchemaAdapter, BearerAuth, OAuthAuth, EnvAuth, mcp_tool_filter,"
        ")"
    ),
}


def venv_python(venv: Path) -> Path:
    sub = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return venv / sub


def check(rel: str, smoke: str, tmp_root: Path) -> tuple[str, str]:
    """Install ``rel`` alone into a fresh venv and import its public API."""
    venv = tmp_root / rel.replace("/", "_")
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        check=True,
        capture_output=True,
    )
    py = venv_python(venv)

    # Core is the one sibling allowed: every provider declares it. Nothing
    # else goes in — no pytest, no other providers, no extras.
    install = subprocess.run(
        [str(py), "-m", "pip", "install", "-q", str(CORE), str(ROOT / rel)],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        return "INSTALL FAILED", install.stderr.strip().splitlines()[-1:][0][:200]

    run = subprocess.run(
        [str(py), "-c", f"{smoke}\nprint('ok')"],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        lines = [ln for ln in run.stderr.strip().splitlines() if ln.strip()]
        return "IMPORT FAILED", lines[-1][:200] if lines else "unknown error"

    listing = subprocess.run(
        [str(py), "-m", "pip", "list", "--format=json"],
        capture_output=True,
        text=True,
    )
    count = len(json.loads(listing.stdout)) if listing.returncode == 0 else -1
    return "OK", f"{count} packages resolved"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "filters",
        nargs="*",
        help="only check packages whose path contains one of these substrings",
    )
    args = parser.parse_args()

    discovered = set(discover_shipped_provider_relpaths())
    registered = set(CASES)
    if not args.filters:
        missing = sorted(discovered - registered)
        extra = sorted(registered - discovered)
        if missing or extra:
            if missing:
                print(
                    "ERROR: shipped provider(s) have no CASES smoke import:",
                    file=sys.stderr,
                )
                for rel in missing:
                    print(f"  + {rel}", file=sys.stderr)
            if extra:
                print(
                    "ERROR: CASES lists paths that are not shipped providers:",
                    file=sys.stderr,
                )
                for rel in extra:
                    print(f"  - {rel}", file=sys.stderr)
            print(
                "\nAdd the package to CASES in "
                "scripts/verify_dependency_completeness.py.",
                file=sys.stderr,
            )
            return 1

    selected = {
        rel: smoke
        for rel, smoke in CASES.items()
        if not args.filters or any(f in rel for f in args.filters)
    }
    if not selected:
        print(f"no packages matched {args.filters!r}", file=sys.stderr)
        return 2

    tmp_root = Path(tempfile.mkdtemp(prefix="nq-depcheck-"))
    results: list[tuple[str, str, str]] = []
    try:
        for rel, smoke in selected.items():
            print(f"... {rel}", flush=True)
            status, detail = check(rel, smoke, tmp_root)
            results.append((rel, status, detail))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print()
    failures = 0
    for rel, status, detail in results:
        failures += status != "OK"
        print(f"{status:<15} {rel:<46} {detail}")
    print()

    if failures:
        print(
            f"FAIL: {failures} package(s) import modules they do not declare.\n"
            "Add the missing distribution to that package's "
            "[project].dependencies — transitive availability is not a contract."
        )
        return 1
    print(f"OK: all {len(results)} packages import from declared dependencies alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
