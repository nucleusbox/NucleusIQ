"""Public package surface."""

from __future__ import annotations

import nucleusiq_groq as ng


def test_version_and_all_exports() -> None:
    assert isinstance(ng.__version__, str)
    assert ng.BaseGroq is not None
    assert ng.GroqLLMParams is not None
    assert len(ng.NATIVE_TOOL_TYPES) == 0
    assert callable(ng.build_response_format)
    assert callable(ng.parse_response)
    assert callable(ng.to_openai_function_tool)
