"""Mock factory functions for Gemini provider unit tests.

Provides mock objects that simulate the Gemini SDK response shapes
without requiring actual API access. These are importable functions
(not pytest fixtures) so test files can use them directly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def make_text_part(text: str, *, thought: bool = False) -> SimpleNamespace:
    """Create a mock Gemini Part with text."""
    return SimpleNamespace(
        text=text,
        inline_data=None,
        function_call=None,
        function_response=None,
        thought=text if thought else None,
        executable_code=None,
        code_execution_result=None,
    )


def make_function_call_part(
    name: str,
    args: dict[str, Any] | None = None,
    call_id: str | None = None,
) -> SimpleNamespace:
    """Create a mock Gemini Part with a function call."""
    return SimpleNamespace(
        text=None,
        inline_data=None,
        function_call=SimpleNamespace(
            name=name,
            args=args or {},
            id=call_id,
        ),
        function_response=None,
        thought=None,
        executable_code=None,
        code_execution_result=None,
    )


def make_code_execution_part(
    code: str = "print('hello')",
    output: str = "hello",
) -> SimpleNamespace:
    """Create a mock Gemini Part with code execution."""
    return SimpleNamespace(
        text=None,
        inline_data=None,
        function_call=None,
        function_response=None,
        thought=None,
        executable_code=SimpleNamespace(code=code, language="PYTHON"),
        code_execution_result=SimpleNamespace(output=output, outcome="OUTCOME_OK"),
    )


def make_candidate(parts: list[SimpleNamespace]) -> SimpleNamespace:
    """Create a mock Gemini candidate."""
    return SimpleNamespace(
        content=SimpleNamespace(parts=parts, role="model"),
    )


def make_usage_metadata(
    prompt: int = 10,
    candidates: int = 20,
    total: int = 30,
    thoughts: int = 0,
    cached: int = 0,
) -> SimpleNamespace:
    """Create mock usage metadata."""
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=total,
        thoughts_token_count=thoughts,
        cached_content_token_count=cached,
    )


def make_response(
    candidates: list[SimpleNamespace] | None = None,
    usage: SimpleNamespace | None = None,
    model_version: str | None = "gemini-2.5-flash",
) -> SimpleNamespace:
    """Create a mock Gemini GenerateContentResponse."""
    if candidates is None:
        candidates = [make_candidate([make_text_part("Hello!")])]
    if usage is None:
        usage = make_usage_metadata()
    return SimpleNamespace(
        candidates=candidates,
        usage_metadata=usage,
        model_version=model_version,
    )


def make_stream_chunks(
    texts: list[str],
    *,
    function_calls: list[tuple[str, dict[str, Any]]] | None = None,
    thoughts: list[str] | None = None,
    usage: SimpleNamespace | None = None,
) -> list[SimpleNamespace]:
    """Create a list of mock streaming chunks."""
    chunks = []
    for i, text in enumerate(texts):
        parts = [make_text_part(text)]
        chunk_usage = usage if i == len(texts) - 1 else None
        chunks.append(
            SimpleNamespace(
                candidates=[make_candidate(parts)],
                usage_metadata=chunk_usage,
            )
        )

    if thoughts:
        for thought_text in thoughts:
            parts = [make_text_part(thought_text, thought=True)]
            chunks.insert(
                0,
                SimpleNamespace(
                    candidates=[make_candidate(parts)],
                    usage_metadata=None,
                ),
            )

    if function_calls:
        for name, args in function_calls:
            parts = [make_function_call_part(name, args, call_id=f"call_{name}")]
            chunks.append(
                SimpleNamespace(
                    candidates=[make_candidate(parts)],
                    usage_metadata=None,
                )
            )

    return chunks
