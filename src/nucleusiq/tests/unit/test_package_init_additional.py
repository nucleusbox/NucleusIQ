"""Extra coverage for package __init__ exception guard."""

from __future__ import annotations

import importlib


def test_init_handles_dotenv_import_failure(monkeypatch):
    """If dotenv is somehow unimportable, import nucleusiq must still work."""
    import nucleusiq

    real_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def _patched_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("no dotenv")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _patched_import)
    importlib.reload(nucleusiq)
    assert hasattr(nucleusiq, "__version__")
