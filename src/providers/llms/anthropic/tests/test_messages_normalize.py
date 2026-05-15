"""Response normalisation."""

from __future__ import annotations

from types import SimpleNamespace

from anthropic import NOT_GIVEN
from nucleusiq_anthropic.nb_anthropic.messages import (
    build_create_kwargs,
    normalize_message_response,
)


def test_normalize_text_and_tools() -> None:
    tb = lambda **kw: SimpleNamespace(**kw)

    resp = tb(
        content=[
            tb(type="text", text="Hey"),
            tb(type="tool_use", id="tid", name="sum", input={"a": 1}),
        ],
        usage=tb(
            input_tokens=3,
            output_tokens=5,
            cache_read_input_tokens=1,
            cache_creation_input_tokens=0,
        ),
        model="claude-haiku",
        id="mid",
    )

    normed = normalize_message_response(resp)

    md = normed.choices[0].message

    assert md.content.strip() == "Hey"

    assert md.tool_calls and md.tool_calls[0].function.name == "sum"

    assert normed.usage

    assert normed.usage.prompt_tokens == 4

    assert normed.response_id == "mid"


def test_normalize_handles_non_mapping_tool_payload() -> None:
    tb = lambda **kw: SimpleNamespace(**kw)

    blob = tb(type="tool_use", id="", name="n", input=[1, 2])

    resp = tb(
        content=[blob],
        usage=tb(
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        model="m",
        id="i",
    )

    normed = normalize_message_response(resp)
    args = normed.choices[0].message.tool_calls[0].function.arguments
    assert "1" in args


def test_normalize_json_dump_failure(monkeypatch) -> None:
    import nucleusiq_anthropic.nb_anthropic.messages as m

    tb = lambda **kw: SimpleNamespace(**kw)

    resp = tb(
        content=[tb(type="tool_use", id="z", name="z", input={"p": 1})],
        usage=tb(
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        model="m",
        id="i",
    )

    def _boom(*_a: object, **_kw: object) -> None:
        raise TypeError("forced")

    monkeypatch.setattr(m.json, "dumps", _boom)

    normed = normalize_message_response(resp)

    assert normed.choices[0].message.tool_calls[0].function.arguments == "{}"


def test_build_create_kwargs_skips_not_given_and_merges_metadata() -> None:
    kw = build_create_kwargs(
        model="m",
        framework_messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=10,
        temperature=0.0,
        top_p=0.9,
        stop=None,
        tools=None,
        tool_choice=None,
        merged_extras={"metadata": {"trace": "1"}, "ghost": NOT_GIVEN, "empty": None},
        extra_headers=None,
        stream=False,
    )

    assert kw["metadata"] == {"trace": "1"}
    assert "ghost" not in kw
    assert "empty" not in kw
    assert kw["temperature"] == 0.0
    assert kw["top_p"] is NOT_GIVEN
