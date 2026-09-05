"""Registry + layout checks for the monorepo verify scripts.

These scripts run as their own CI jobs.  The tests here are the cheap
half: they fail if a new provider is added on disk and nobody updates
``HATCH_PROVIDERS`` / ``CASES``.  They do *not* spawn the isolated
virtualenvs — that is ``dependency-completeness`` in CI, and it is
deliberately slow.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SCRIPTS = REPO / "scripts"

EXPECTED_DISTS = {
    "nucleusiq-openai",
    "nucleusiq-gemini",
    "nucleusiq-anthropic",
    "nucleusiq-groq",
    "nucleusiq-ollama",
    "nucleusiq-openai-compatible",
    "nucleusiq-mcp",
}

EXPECTED_RELS = {
    "src/providers/llms/openai",
    "src/providers/llms/gemini",
    "src/providers/llms/anthropic",
    "src/providers/inference/groq",
    "src/providers/inference/ollama",
    "src/providers/inference/openai_compatible",
    "src/providers/tools/mcp",
}


def _load(filename: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_layout_registry_includes_openai_compatible():
    layout = _load("verify_core_package_layout.py")
    names = {hp.dist_name for hp in layout.HATCH_PROVIDERS}
    assert "nucleusiq-openai-compatible" in names
    assert names == EXPECTED_DISTS


def test_layout_registry_matches_disk():
    layout = _load("verify_core_package_layout.py")
    layout.verify_registry_matches_disk()
    on_disk = {p.relative_to(REPO).as_posix() for p in layout.discover_shipped_provider_dirs()}
    assert on_disk == EXPECTED_RELS


def test_dependency_cases_include_openai_compatible():
    deps = _load("verify_dependency_completeness.py")
    assert "src/providers/inference/openai_compatible" in deps.CASES
    assert "OpenAICompatibleLLM" in deps.CASES["src/providers/inference/openai_compatible"]
    assert set(deps.CASES) == EXPECTED_RELS


def test_dependency_cases_match_disk():
    shipped = _load("_shipped_providers.py")
    deps = _load("verify_dependency_completeness.py")
    assert set(shipped.discover_shipped_provider_relpaths()) == set(deps.CASES)


def test_layout_script_exits_zero():
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "verify_core_package_layout.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "nucleusiq-openai-compatible" in result.stdout
