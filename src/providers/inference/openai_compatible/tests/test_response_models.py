"""Response normalization across servers that populate different fields."""

from __future__ import annotations

import pytest
from conftest import (
    FakeChoice,
    FakeMessage,
    FakeResponse,
    FakeToolCall,
    FakeUsage,
)
from nucleusiq_openai_compatible._shared.response_models import (
    NormalizedResponse,
    NormalizedToolCall,
    extract_reasoning,
    normalize_response,
)


class TestContent:
    def test_plain_text(self) -> None:
        result = normalize_response(FakeResponse(FakeMessage("hello")))
        assert result.content == "hello"
        assert result.finish_reason == "stop"
        assert not result.has_tool_calls

    def test_null_content(self) -> None:
        assert normalize_response(FakeResponse(FakeMessage(None))).content is None

    def test_non_string_content_becomes_none(self) -> None:
        assert normalize_response(FakeResponse(FakeMessage(["a"]))).content is None  # type: ignore[arg-type]

    def test_empty_choices_is_not_fatal(self) -> None:
        result = normalize_response(FakeResponse(choices=[]))
        assert result.content is None
        assert result.finish_reason is None

    def test_model_and_request_id(self) -> None:
        result = normalize_response(FakeResponse(id="req-1", model="gemma"))
        assert result.request_id == "req-1"
        assert result.model == "gemma"

    def test_header_request_id_wins(self) -> None:
        result = normalize_response(FakeResponse(id="body-id"), request_id="header-id")
        assert result.request_id == "header-id"

    def test_raw_is_preserved_but_not_compared(self) -> None:
        response = FakeResponse(FakeMessage("x"))
        assert normalize_response(response).raw is response
        assert normalize_response(response) == normalize_response(
            FakeResponse(FakeMessage("x"))
        )


class TestToolCalls:
    def test_single_call(self) -> None:
        message = FakeMessage(None, tool_calls=[FakeToolCall(name="search")])
        result = normalize_response(FakeResponse(message))
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == '{"q":"x"}'

    def test_multiple_parallel_calls(self) -> None:
        message = FakeMessage(
            None,
            tool_calls=[
                FakeToolCall(id="c1", name="search"),
                FakeToolCall(id="c2", name="calc"),
            ],
        )
        result = normalize_response(FakeResponse(message))
        assert [c.name for c in result.tool_calls] == ["search", "calc"]

    def test_unnamed_call_skipped(self) -> None:
        message = FakeMessage(None, tool_calls=[FakeToolCall(name=None)])
        assert normalize_response(FakeResponse(message)).tool_calls == ()

    def test_missing_id_is_synthesized(self) -> None:
        message = FakeMessage(None, tool_calls=[FakeToolCall(id=None, name="search")])
        assert normalize_response(FakeResponse(message)).tool_calls[0].id == "call_0"

    def test_null_arguments_become_empty_string(self) -> None:
        message = FakeMessage(
            None, tool_calls=[FakeToolCall(name="ping", arguments=None)]
        )
        assert normalize_response(FakeResponse(message)).tool_calls[0].arguments == ""

    def test_to_wire_round_trip(self) -> None:
        call = NormalizedToolCall(id="c1", name="search", arguments='{"q":"x"}')
        assert call.to_wire() == {
            "id": "c1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q":"x"}'},
        }


class TestUsage:
    def test_reported_usage(self) -> None:
        result = normalize_response(FakeResponse(usage=FakeUsage(10, 5, 15)))
        assert (result.prompt_tokens, result.completion_tokens) == (10, 5)
        assert result.total_tokens == 15
        assert result.usage_reported

    def test_total_is_derived_when_absent(self) -> None:
        result = normalize_response(FakeResponse(usage=FakeUsage(10, 5, None)))
        assert result.total_tokens == 15

    def test_missing_usage_is_flagged(self) -> None:
        # llama.cpp and some gateways omit usage entirely.
        result = normalize_response(FakeResponse(usage=None))
        assert result.usage_reported is False
        assert result.prompt_tokens is None

    def test_booleans_rejected_as_counts(self) -> None:
        result = normalize_response(FakeResponse(usage=FakeUsage(True, 5, 15)))  # type: ignore[arg-type]
        assert result.prompt_tokens is None


class TestUsageViewForTheFramework:
    """``response.usage`` is where the framework's accounting layer looks.

    ``UsageTracker.record_from_response`` and ``build_llm_call_record`` both
    return early when it is absent, so without this view every run reports
    zero tokens and every cost estimate is $0.00.
    """

    def test_exposes_a_nested_usage_view(self) -> None:
        result = normalize_response(FakeResponse(usage=FakeUsage(10, 5, 15)))
        assert result.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": 0,
        }

    def test_none_when_server_reported_nothing(self) -> None:
        """Zeros would be indistinguishable from a genuinely free call."""
        assert normalize_response(FakeResponse(usage=None)).usage is None

    def test_reasoning_tokens_are_carried(self) -> None:
        result = normalize_response(
            FakeResponse(usage=FakeUsage(10, 5, 15, reasoning_tokens=4))
        )
        assert result.usage["reasoning_tokens"] == 4

    def test_total_is_derived_when_the_server_omits_it(self) -> None:
        result = normalize_response(FakeResponse(usage=FakeUsage(10, 5, None)))
        assert result.usage["total_tokens"] == 15

    def test_framework_parser_reads_it(self) -> None:
        from nucleusiq.agents.observability._response_parser import (
            usage_dict_from_response,
        )

        result = normalize_response(FakeResponse(usage=FakeUsage(10, 5, 15)))
        assert usage_dict_from_response(result) == result.usage

    def test_usage_tracker_records_it(self) -> None:
        from nucleusiq.agents.usage.usage_tracker import CallPurpose, UsageTracker

        tracker = UsageTracker()
        tracker.record_from_response(
            CallPurpose.MAIN,
            normalize_response(FakeResponse(usage=FakeUsage(10, 5, 15))),
        )
        assert tracker.total_tokens == 15
        assert tracker.call_count == 1

    def test_reasoning_tokens_read_from_details(self) -> None:
        result = normalize_response(
            FakeResponse(usage=FakeUsage(10, 20, 30, reasoning_tokens=17))
        )
        assert result.reasoning_tokens == 17

    def test_reasoning_tokens_absent(self) -> None:
        assert normalize_response(FakeResponse()).reasoning_tokens is None


class TestReasoning:
    @pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
    def test_both_field_names_supported(self, field: str) -> None:
        message = FakeMessage("answer", reasoning="step by step", reasoning_field=field)
        result = normalize_response(FakeResponse(message))
        assert result.reasoning == "step by step"
        assert result.has_reasoning
        assert result.content == "answer"

    def test_new_name_wins_when_both_present(self) -> None:
        message = FakeMessage("answer", reasoning="new", reasoning_field="reasoning")
        message.reasoning_content = "legacy"  # type: ignore[attr-defined]
        assert normalize_response(FakeResponse(message)).reasoning == "new"

    def test_absent_reasoning(self) -> None:
        result = normalize_response(FakeResponse(FakeMessage("answer")))
        assert result.reasoning is None
        assert not result.has_reasoning

    def test_empty_reasoning_is_not_reasoning(self) -> None:
        message = FakeMessage("answer", reasoning="")
        assert normalize_response(FakeResponse(message)).reasoning is None

    def test_reasoning_never_merged_into_content(self) -> None:
        message = FakeMessage("answer", reasoning="secret chain of thought")
        result = normalize_response(FakeResponse(message))
        assert "secret" not in (result.content or ""), (
            "thinking must stay separate or it leaks into user-visible output "
            "and into the next turn's history"
        )

    def test_extract_from_dict_delta(self) -> None:
        assert extract_reasoning({"reasoning": "abc"}) == "abc"
        assert extract_reasoning({"reasoning_content": "abc"}) == "abc"

    def test_extract_returns_none(self) -> None:
        assert extract_reasoning(None) is None
        assert extract_reasoning({}) is None
        assert extract_reasoning({"reasoning": 42}) is None


class TestReasoningOnlyTrap:
    """vLLM #53284: template and parser disagree, the answer lands in reasoning."""

    def test_detected(self) -> None:
        message = FakeMessage(None, reasoning="the whole answer went here")
        assert normalize_response(FakeResponse(message)).reasoning_only

    def test_not_flagged_when_content_present(self) -> None:
        message = FakeMessage("answer", reasoning="thinking")
        assert not normalize_response(FakeResponse(message)).reasoning_only

    def test_not_flagged_when_tool_calls_present(self) -> None:
        message = FakeMessage(
            None, tool_calls=[FakeToolCall(name="search")], reasoning="thinking"
        )
        assert not normalize_response(FakeResponse(message)).reasoning_only, (
            "a tool call is a valid outcome; only a total absence of output "
            "signals the trap"
        )

    def test_plain_empty_response_is_not_the_trap(self) -> None:
        assert not normalize_response(FakeResponse(FakeMessage(None))).reasoning_only


class TestValueSemantics:
    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            normalize_response(FakeResponse()).content = "x"  # type: ignore[misc]

    def test_defaults(self) -> None:
        empty = NormalizedResponse(content=None)
        assert empty.tool_calls == ()
        assert not empty.has_reasoning
        assert not empty.usage_reported

    def test_finish_reason_non_string_dropped(self) -> None:
        response = FakeResponse(choices=[FakeChoice(FakeMessage("x"), finish_reason=7)])  # type: ignore[arg-type]
        assert normalize_response(response).finish_reason is None
