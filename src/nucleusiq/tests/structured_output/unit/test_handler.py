"""Tests for StructuredOutputHandler."""

from __future__ import annotations

from types import SimpleNamespace

from nucleusiq.agents.structured_output import OutputMode, OutputSchema
from nucleusiq.agents.structured_output.handler import StructuredOutputHandler


class _Cfg:
    def __init__(self, mode, schema=None, schema_name="MySchema"):
        self._resolved_mode = mode
        self.schema = schema if schema is not None else {"type": "json_object"}
        self.schema_name = schema_name

    def for_provider(self, _provider):
        return {"type": "json_schema", "json_schema": {"name": self.schema_name}}


def test_get_call_kwargs_for_non_native_mode_returns_empty():
    h = StructuredOutputHandler()
    cfg = _Cfg(OutputMode.AUTO)
    assert h.get_call_kwargs(cfg, response_format=None, llm=None) == {}


def test_get_call_kwargs_with_output_schema_uses_provider_format():
    h = StructuredOutputHandler()
    cfg = _Cfg(OutputMode.NATIVE, schema={"type": "object"}, schema_name="Person")
    rf = OutputSchema(schema={"type": "object"})
    kwargs = h.get_call_kwargs(cfg, response_format=rf, llm=SimpleNamespace())
    assert "response_format" in kwargs
    provider_format, raw_schema = kwargs["response_format"]
    assert provider_format["type"] == "json_schema"
    assert raw_schema == {"type": "object"}


def test_get_call_kwargs_without_output_schema_returns_plain_schema():
    h = StructuredOutputHandler()
    cfg = _Cfg(OutputMode.NATIVE, schema={"type": "json_object"})
    kwargs = h.get_call_kwargs(cfg, response_format={"type": "json_object"}, llm=None)
    assert kwargs == {"response_format": {"type": "json_object"}}


def test_wrap_result_native_without_choices_returns_metadata_dict():
    h = StructuredOutputHandler()
    cfg = _Cfg(OutputMode.NATIVE, schema_name="MySchema")
    wrapped = h.wrap_result("raw text", cfg)
    assert wrapped["output"] == "raw text"
    assert wrapped["schema"] == "MySchema"
    assert wrapped["mode"] == "native"


def test_wrap_result_extracts_content_from_choice_message():
    h = StructuredOutputHandler()
    cfg = _Cfg(OutputMode.AUTO)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message={"content": "json content"})]
    )
    assert h.wrap_result(response, cfg) == "json content"


def test_wrap_result_returns_original_when_no_choices():
    h = StructuredOutputHandler()
    cfg = _Cfg(OutputMode.AUTO)
    response = SimpleNamespace(choices=[])
    assert h.wrap_result(response, cfg) is response


def test_resolve_response_format_none_returns_none():
    h = StructuredOutputHandler()
    assert h.resolve_response_format(None, llm=None) is None


def test_wrap_result_extracts_content_from_object_message():
    h = StructuredOutputHandler()
    cfg = _Cfg(OutputMode.AUTO)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="object-content"))]
    )
    assert h.wrap_result(response, cfg) == "object-content"
