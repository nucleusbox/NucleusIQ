"""Extra coverage for package __init__ exception guard."""

from __future__ import annotations

import importlib


def test_init_handles_path_resolve_failure(monkeypatch):
    import pathlib

    import nucleusiq

    def _boom(self):
        raise RuntimeError("resolve failure")

    monkeypatch.setattr(pathlib.Path, "resolve", _boom)
    importlib.reload(nucleusiq)
    assert hasattr(nucleusiq, "__version__")

