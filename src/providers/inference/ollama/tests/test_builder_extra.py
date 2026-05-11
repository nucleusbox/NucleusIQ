"""More build_ollama_format branches."""

from __future__ import annotations

import dataclasses

from nucleusiq_ollama.structured_output import build_ollama_format


@dataclasses.dataclass
class DC:
    name: str
    count: int = 0


class Proto:
    a: str
    b: int


def test_build_from_dataclass() -> None:
    fmt = build_ollama_format(DC)
    assert fmt.get("type") == "object"
    assert "name" in fmt["properties"]


def test_build_from_annotations_class() -> None:
    fmt = build_ollama_format(Proto)
    assert "a" in fmt["properties"]


def test_build_unknown_returns_none() -> None:
    assert build_ollama_format(42) is None  # type: ignore[arg-type]
