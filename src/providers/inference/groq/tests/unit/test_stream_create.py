"""Unit tests for ``nucleusiq_groq._shared.stream_create``."""

from __future__ import annotations

from nucleusiq_groq._shared.stream_create import apply_stream_options


def test_apply_stream_options_adds_stream_flags() -> None:
    base = {"model": "m", "messages": []}
    out = apply_stream_options(base)
    assert out["stream"] is True
    assert out["stream_options"] == {"include_usage": True}
    assert out["model"] == "m"
    assert base.get("stream") is None
