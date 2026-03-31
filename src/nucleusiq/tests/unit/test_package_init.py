"""Tests for package-level initialization behavior."""

from __future__ import annotations

import importlib


def test_init_loads_dotenv(monkeypatch):
    """load_dotenv() should be called on import (no path arg)."""
    import dotenv
    import nucleusiq

    loaded = {"called": False, "override": None}

    def _fake_load_dotenv(override=False):
        loaded["called"] = True
        loaded["override"] = override
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", _fake_load_dotenv)

    importlib.reload(nucleusiq)
    assert loaded["called"] is True
    assert loaded["override"] is False


def test_init_never_raises_when_dotenv_loading_fails(monkeypatch):
    """Any dotenv-loading exception should be swallowed."""
    import dotenv
    import nucleusiq

    def _boom(*_args, **_kwargs):
        raise RuntimeError("dotenv failure")

    monkeypatch.setattr(dotenv, "load_dotenv", _boom)

    importlib.reload(nucleusiq)
