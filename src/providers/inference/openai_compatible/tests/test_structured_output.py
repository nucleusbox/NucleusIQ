"""Structured-output policies — the vLLM tools/schema conflict."""

from __future__ import annotations

import json

import pytest
from nucleusiq.llms.errors import InvalidRequestError
from nucleusiq_openai_compatible.structured_output import (
    DropPolicy,
    ErrorPolicy,
    PromptPolicy,
    StructuredOutputPolicy,
    build_policy,
    render_schema_instruction,
)

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def decide(policy, **overrides):
    kwargs = {
        "schema": SCHEMA,
        "schema_name": "Answer",
        "has_tools": False,
        "supports_json_schema": True,
        "suppresses_tools": True,
    }
    kwargs.update(overrides)
    return policy.decide(**kwargs)


class TestFactory:
    @pytest.mark.parametrize(
        ("mode", "cls"),
        [("prompt", PromptPolicy), ("drop", DropPolicy), ("error", ErrorPolicy)],
    )
    def test_builds_each_policy(self, mode: str, cls: type) -> None:
        policy = build_policy(mode)
        assert isinstance(policy, cls)
        assert policy.name == mode

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="structured_output_with_tools"):
            build_policy("inject")

    def test_all_satisfy_the_protocol(self) -> None:
        for mode in ("prompt", "drop", "error"):
            assert isinstance(build_policy(mode), StructuredOutputPolicy)


class TestWithoutTools:
    """Policy-independent: with no tools there is no conflict to resolve."""

    @pytest.mark.parametrize("mode", ["prompt", "drop", "error"])
    def test_native_json_schema_used(self, mode: str) -> None:
        decision = decide(build_policy(mode))
        assert decision.response_format["type"] == "json_schema"
        assert decision.response_format["json_schema"]["name"] == "Answer"
        assert decision.response_format["json_schema"]["schema"] == SCHEMA
        assert decision.response_format["json_schema"]["strict"] is True
        assert decision.prompt_instruction is None

    @pytest.mark.parametrize("mode", ["prompt", "drop", "error"])
    def test_falls_back_to_json_object(self, mode: str) -> None:
        decision = decide(build_policy(mode), supports_json_schema=False)
        assert decision.response_format == {"type": "json_object"}
        assert "JSON Schema" in decision.prompt_instruction
        assert "json_schema" in decision.reason


class TestPromptPolicy:
    """The default: keeps the tool loop working AND delivers structured output."""

    def test_omits_response_format_and_injects_schema(self) -> None:
        decision = decide(PromptPolicy(), has_tools=True)
        assert decision.response_format is None, (
            "sending response_format with tools would make vLLM suppress "
            "tool calls entirely"
        )
        assert "answer" in decision.prompt_instruction
        assert "tools present" in decision.reason

    def test_instruction_carries_the_full_schema(self) -> None:
        decision = decide(PromptPolicy(), has_tools=True)
        assert json.dumps(SCHEMA, indent=2, sort_keys=True) in (
            decision.prompt_instruction
        )

    def test_does_not_warn(self, recwarn) -> None:
        decide(PromptPolicy(), has_tools=True)
        assert len(recwarn) == 0, "the default path is not a degradation"


class TestDropPolicy:
    def test_drops_schema_entirely_with_warning(self) -> None:
        with pytest.warns(UserWarning, match="response_format has been dropped"):
            decision = decide(DropPolicy(), has_tools=True)
        assert decision.response_format is None
        assert decision.prompt_instruction is None

    def test_warning_points_at_prompt_mode(self) -> None:
        with pytest.warns(UserWarning, match="structured_output_with_tools='prompt'"):
            decide(DropPolicy(), has_tools=True)


class TestErrorPolicy:
    def test_raises_before_the_http_call(self) -> None:
        with pytest.raises(InvalidRequestError, match="cannot be combined with tools"):
            decide(ErrorPolicy(), has_tools=True)

    def test_message_explains_both_remedies(self) -> None:
        with pytest.raises(InvalidRequestError) as exc:
            decide(ErrorPolicy(), has_tools=True)
        message = str(exc.value)
        assert "drop the tools" in message
        assert "'prompt'" in message


class TestSchemaInstruction:
    def test_forbids_markdown_fences(self) -> None:
        text = render_schema_instruction(SCHEMA)
        assert "no markdown code fences" in text

    def test_is_deterministic(self) -> None:
        assert render_schema_instruction(SCHEMA) == render_schema_instruction(SCHEMA)

    def test_keys_sorted_for_stable_prompt_caching(self) -> None:
        text = render_schema_instruction({"b": 1, "a": 2})
        assert text.index('"a"') < text.index('"b"')
