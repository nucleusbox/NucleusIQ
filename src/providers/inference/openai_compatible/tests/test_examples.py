"""Guards that keep ``examples/`` honest.

The examples are documentation people copy from, so a rename in the provider
must not leave them silently broken.  These checks are static — nothing here
contacts a server.
"""

from __future__ import annotations

import ast
from pathlib import Path

import nucleusiq_openai_compatible as package
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(EXAMPLES.rglob("*.py"))


def ids(paths: list[Path]) -> list[str]:
    return [p.relative_to(EXAMPLES).as_posix() for p in paths]


def test_examples_exist() -> None:
    assert SCRIPTS, "the examples directory should not be empty"


@pytest.mark.parametrize("script", SCRIPTS, ids=ids(SCRIPTS))
class TestEachScript:
    def test_parses(self, script: Path) -> None:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))

    def test_imports_only_real_provider_names(self, script: Path) -> None:
        tree = ast.parse(script.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "nucleusiq_openai_compatible":
                continue
            for alias in node.names:
                assert hasattr(package, alias.name), (
                    f"{script.name} imports {alias.name!r}, which the package "
                    "no longer exports"
                )

    def test_is_runnable_as_a_script(self, script: Path) -> None:
        source = script.read_text(encoding="utf-8")
        assert '__name__ == "__main__"' in source, (
            "importing an example must not fire off network calls"
        )

    def test_has_a_module_docstring(self, script: Path) -> None:
        tree = ast.parse(script.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), f"{script.name} needs a docstring"


class TestReadme:
    @pytest.fixture(scope="class")
    def readme(self) -> str:
        return (EXAMPLES / "README.md").read_text(encoding="utf-8")

    def test_exists(self, readme: str) -> None:
        assert readme.strip()

    @pytest.mark.parametrize("script", SCRIPTS, ids=ids(SCRIPTS))
    def test_every_script_is_listed(self, readme: str, script: Path) -> None:
        assert script.relative_to(EXAMPLES).as_posix() in readme, (
            "an unlisted example will not be found by anyone"
        )
