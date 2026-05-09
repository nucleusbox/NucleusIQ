"""Tests for Groq model capability helpers."""

from __future__ import annotations

import logging

import pytest
from nucleusiq.llms.errors import InvalidRequestError
from nucleusiq_groq.capabilities import (
    PARALLEL_TOOL_CALLS_DOCUMENTED_MODELS,
    check_parallel_tool_calls_capability,
)


def test_parallel_documented_models_non_empty() -> None:
    assert "llama-3.3-70b-versatile" in PARALLEL_TOOL_CALLS_DOCUMENTED_MODELS


def test_check_parallel_noop_when_disabled() -> None:
    log = logging.getLogger("test_cap")
    check_parallel_tool_calls_capability(
        "any",
        None,
        strict=True,
        logger=log,
    )


def test_check_parallel_known_model_strict() -> None:
    log = logging.getLogger("test_cap")
    check_parallel_tool_calls_capability(
        "llama-3.3-70b-versatile",
        True,
        strict=True,
        logger=log,
    )


def test_check_parallel_unknown_strict_raises() -> None:
    log = logging.getLogger("test_cap")
    with pytest.raises(InvalidRequestError):
        check_parallel_tool_calls_capability(
            "unknown-custom-model",
            True,
            strict=True,
            logger=log,
        )


def test_check_parallel_unknown_warn_only(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    check_parallel_tool_calls_capability(
        "unknown-custom-model",
        True,
        strict=False,
        logger=logging.getLogger(),
    )
    assert "capability allowlist" in caplog.text
