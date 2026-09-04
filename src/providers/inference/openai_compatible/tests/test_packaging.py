"""Build-artifact guards.

``tests/`` and ``examples/`` sit beside the import root, so the wheel excludes
them by construction.  An sdist, however, defaults to everything not
gitignored — these tests fail loudly if either exclusion is ever dropped.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOT = "nucleusiq_openai_compatible"


def read_pyproject() -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - 3.10 fallback
        import tomli as tomllib
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))


@pytest.fixture(scope="module")
def config() -> dict:
    return read_pyproject()


class TestSourceLayout:
    def test_tests_live_outside_the_import_root(self) -> None:
        assert not (PROJECT_ROOT / IMPORT_ROOT / "tests").exists(), (
            "tests inside the import root would be installed into every "
            "user's site-packages"
        )

    def test_examples_live_outside_the_import_root(self) -> None:
        assert not (PROJECT_ROOT / IMPORT_ROOT / "examples").exists()

    def test_both_folders_exist_at_project_level(self) -> None:
        assert (PROJECT_ROOT / "tests").is_dir()
        assert (PROJECT_ROOT / "examples").is_dir()


class TestBuildConfig:
    def test_wheel_scoped_to_the_import_root(self, config: dict) -> None:
        packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        assert packages == [IMPORT_ROOT]

    @pytest.mark.parametrize("folder", ["tests", "examples"])
    def test_sdist_excludes_the_folder(self, config: dict, folder: str) -> None:
        exclude = config["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
        assert folder in exclude, (
            f"{folder}/ would otherwise be published to PyPI in the sdist"
        )

    def test_pytest_not_a_runtime_dependency(self, config: dict) -> None:
        deps = " ".join(config["project"]["dependencies"]).lower()
        for package in ("pytest", "pytest-asyncio", "pytest-cov"):
            assert package not in deps

    def test_coverage_gate_configured(self, config: dict) -> None:
        addopts = config["tool"]["pytest"]["ini_options"]["addopts"]
        assert "--cov-fail-under=95" in addopts


@pytest.mark.slow
class TestBuiltArtifacts:
    """Actually build, then inspect. The only check that cannot be fooled."""

    @pytest.fixture(scope="class")
    def artifacts(self) -> tuple[list[str], list[str]]:
        if shutil.which("python") is None:  # pragma: no cover
            pytest.skip("no interpreter on PATH")
        with tempfile.TemporaryDirectory() as out:
            result = subprocess.run(
                [sys.executable, "-m", "hatchling", "build", "-d", out],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:  # pragma: no cover
                pytest.skip(f"hatchling unavailable: {result.stderr[-300:]}")

            wheels = list(Path(out).glob("*.whl"))
            sdists = list(Path(out).glob("*.tar.gz"))
            assert wheels and sdists
            with zipfile.ZipFile(wheels[0]) as zf:
                wheel_names = zf.namelist()
            with tarfile.open(sdists[0]) as tf:
                sdist_names = tf.getnames()
        return wheel_names, sdist_names

    def test_wheel_has_no_tests(self, artifacts) -> None:
        wheel_names, _ = artifacts
        assert not [n for n in wheel_names if "tests/" in n]

    def test_wheel_has_no_examples(self, artifacts) -> None:
        wheel_names, _ = artifacts
        assert not [n for n in wheel_names if "examples/" in n]

    def test_wheel_has_no_bytecode(self, artifacts) -> None:
        wheel_names, _ = artifacts
        assert not [n for n in wheel_names if "__pycache__" in n or n.endswith(".pyc")]

    def test_wheel_contains_the_package(self, artifacts) -> None:
        wheel_names, _ = artifacts
        assert f"{IMPORT_ROOT}/__init__.py" in wheel_names

    @pytest.mark.parametrize("folder", ["tests", "examples"])
    def test_sdist_excludes_the_folder(self, artifacts, folder: str) -> None:
        _, sdist_names = artifacts
        assert not [n for n in sdist_names if f"/{folder}/" in n]

    def test_sdist_contains_the_package_and_readme(self, artifacts) -> None:
        _, sdist_names = artifacts
        assert any(n.endswith("README.md") for n in sdist_names)
        assert any(f"/{IMPORT_ROOT}/__init__.py" in n for n in sdist_names)
