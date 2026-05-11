"""Public package surface."""

from __future__ import annotations

import nucleusiq_ollama as no


def test_version_and_exports() -> None:
    assert isinstance(no.__version__, str)
    assert no.BaseOllama is not None
    assert no.OllamaLLMParams is not None
    assert len(no.NATIVE_TOOL_TYPES) == 0
    assert callable(no.build_ollama_format)
    assert callable(no.parse_response)
    assert callable(no.to_ollama_function_tool)
