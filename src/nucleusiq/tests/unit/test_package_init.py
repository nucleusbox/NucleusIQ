"""Tests for package-level initialization behavior."""

from __future__ import annotations

import importlib


def test_init_loads_dotenv_when_env_exists(monkeypatch):
    """If .env exists, load_dotenv should be invoked."""
    from pathlib import Path

    import dotenv

    import nucleusiq

    loaded = {"called": False}

    def _fake_load_dotenv(_path, override=False):
        loaded["called"] = True
        assert override is False
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", _fake_load_dotenv)
    monkeypatch.setattr(Path, "exists", lambda self: True)

    importlib.reload(nucleusiq)
    assert loaded["called"] is True


def test_init_never_raises_when_dotenv_loading_fails(monkeypatch):
    """Any dotenv-loading exception should be swallowed."""
    from pathlib import Path

    import dotenv

    import nucleusiq

    def _boom(*_args, **_kwargs):
        raise RuntimeError("dotenv failure")

    monkeypatch.setattr(dotenv, "load_dotenv", _boom)
    monkeypatch.setattr(Path, "exists", lambda self: True)

    # Should not raise because module guards dotenv loading.
    importlib.reload(nucleusiq)

