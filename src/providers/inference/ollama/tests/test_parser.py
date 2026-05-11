"""parse_response integration."""

from __future__ import annotations

import dataclasses

import pytest
from nucleusiq_ollama.structured_output.parser import parse_response
from pydantic import BaseModel


class Box(BaseModel):
    x: int


def test_parse_response_pydantic() -> None:
    msg = {"content": '{"x": 3}', "role": "assistant"}
    out = parse_response(msg, Box)
    assert out.x == 3


def test_parse_empty_content_raises() -> None:
    with pytest.raises(Exception):
        parse_response({"content": "", "role": "assistant"}, Box)


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(Exception):
        parse_response({"content": "not json", "role": "assistant"}, Box)


@dataclasses.dataclass
class DC:
    v: int


def test_parse_dataclass() -> None:
    out = parse_response({"content": '{"v": 9}', "role": "assistant"}, DC)
    assert out.v == 9


def test_parse_returns_raw_dict_schema() -> None:
    out = parse_response({"content": '{"z": true}', "role": "assistant"}, {"x": 1})
    assert out == {"z": True}
