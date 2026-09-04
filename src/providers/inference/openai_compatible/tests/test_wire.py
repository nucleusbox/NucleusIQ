"""Payload construction and parameter stripping."""

from __future__ import annotations

import pytest
from nucleusiq.llms.errors import InvalidRequestError
from nucleusiq_openai_compatible._shared.wire import (
    OPENAI_ONLY_PARAMS,
    REASONING_PARAMS,
    PayloadBuilder,
    sanitize_messages,
)
from nucleusiq_openai_compatible.config import ConfigResolver

MESSAGES = [{"role": "user", "content": "hi"}]
TOOL = {"type": "function", "function": {"name": "search", "parameters": {}}}


def build(**overrides) -> PayloadBuilder:
    kwargs = {
        "base_url": "http://gpu:8000/v1",
        "model": "gemma",
        "engine": "vllm",
        "context_window": 32_768,
    }
    kwargs.update(overrides)
    return PayloadBuilder(ConfigResolver.resolve(**kwargs))


class TestCoreShape:
    def test_minimal_payload(self) -> None:
        payload = build().build(messages=MESSAGES)
        assert payload["model"] == "gemma"
        assert payload["messages"] == MESSAGES

    def test_output_limit_uses_configured_field(self) -> None:
        assert "max_tokens" in build().build(messages=[], max_output_tokens=256)
        assert "max_completion_tokens" in build(
            max_tokens_field="max_completion_tokens"
        ).build(messages=[], max_output_tokens=256)

    def test_configured_max_output_used_as_fallback(self) -> None:
        payload = build(max_output_tokens=99).build(messages=[])
        assert payload["max_tokens"] == 99

    def test_call_level_limit_overrides_configured(self) -> None:
        payload = build(max_output_tokens=99).build(messages=[], max_output_tokens=5)
        assert payload["max_tokens"] == 5

    def test_optional_fields_omitted_when_absent(self) -> None:
        payload = build().build(messages=[])
        for key in ("temperature", "stop", "response_format", "tools", "stream"):
            assert key not in payload

    def test_stop_and_temperature(self) -> None:
        payload = build().build(messages=[], temperature=0.2, stop=["END"])
        assert payload["temperature"] == 0.2
        assert payload["stop"] == ["END"]

    def test_empty_stop_omitted(self) -> None:
        assert "stop" not in build().build(messages=[], stop=[])

    def test_response_format_passed_through(self) -> None:
        fmt = {"type": "json_object"}
        assert build().build(messages=[], response_format=fmt)["response_format"] is fmt


class TestToolCallHistory:
    """The framework's canonical tool call is not the OpenAI wire shape.

    ``ToolCallRequest.to_dict()`` emits ``{"id", "name", "arguments"}`` and
    leaves the dialect to the provider.  Forwarding that verbatim breaks the
    second request of every tool loop, which is the only one that carries tool
    calls in history.
    """

    FLAT = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": "add", "arguments": '{"a":21,"b":21}', "id": "call_1"}],
    }

    def test_flat_call_is_nested_under_function(self) -> None:
        payload = build().build(messages=[self.FLAT])
        assert payload["messages"][0]["tool_calls"] == [
            {
                "type": "function",
                "function": {"name": "add", "arguments": '{"a":21,"b":21}'},
                "id": "call_1",
            }
        ]

    def test_already_nested_is_untouched(self) -> None:
        nested = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{}"},
                }
            ],
        }
        assert sanitize_messages([nested])[0] is nested

    def test_caller_messages_are_not_mutated(self) -> None:
        original = dict(self.FLAT)
        build().build(messages=[original])
        assert original["tool_calls"][0] == {
            "name": "add",
            "arguments": '{"a":21,"b":21}',
            "id": "call_1",
        }

    def test_id_omitted_when_absent(self) -> None:
        msg = {"role": "assistant", "tool_calls": [{"name": "f", "arguments": "{}"}]}
        assert "id" not in sanitize_messages([msg])[0]["tool_calls"][0]

    def test_partial_nesting_recovers_arguments(self) -> None:
        """A ``function`` block without arguments, seen from some gateways."""
        msg = {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "f"}, "arguments": '{"x":1}'}],
        }
        call = sanitize_messages([msg])[0]["tool_calls"][0]
        assert call["function"] == {"name": "f", "arguments": '{"x":1}'}

    def test_non_assistant_roles_untouched(self) -> None:
        msg = {"role": "tool", "content": "42", "tool_call_id": "call_1"}
        assert sanitize_messages([msg])[0] is msg

    def test_assistant_without_tool_calls_untouched(self) -> None:
        msg = {"role": "assistant", "content": "hello"}
        assert sanitize_messages([msg])[0] is msg

    def test_non_list_tool_calls_untouched(self) -> None:
        msg = {"role": "assistant", "tool_calls": None}
        assert sanitize_messages([msg])[0] is msg

    def test_non_dict_entries_survive(self) -> None:
        msg = {"role": "assistant", "tool_calls": ["nonsense"]}
        assert sanitize_messages([msg])[0]["tool_calls"] == ["nonsense"]


class TestTools:
    def test_tools_and_choice_included(self) -> None:
        payload = build().build(messages=[], tools=[TOOL], tool_choice="auto")
        assert payload["tools"] == [TOOL]
        assert payload["tool_choice"] == "auto"

    def test_tool_choice_omitted_without_tools(self) -> None:
        assert "tool_choice" not in build().build(messages=[], tool_choice="auto")

    def test_unsupported_tools_warn_and_drop(self) -> None:
        builder = build(supports_tools=False)
        with pytest.warns(UserWarning, match="enable-auto-tool-choice"):
            payload = builder.build(messages=[], tools=[TOOL])
        assert "tools" not in payload

    def test_unsupported_tools_raise_in_strict_mode(self) -> None:
        builder = build(supports_tools=False, strict_capabilities=True)
        with pytest.raises(InvalidRequestError, match="tool calling"):
            builder.build(messages=[], tools=[TOOL])


class TestStreaming:
    def test_usage_requested_when_supported(self) -> None:
        payload = build().build(messages=[], stream=True)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}

    def test_usage_not_requested_when_unsupported(self) -> None:
        payload = build(engine="tgi", context_window=8_192).build(
            messages=[], stream=True
        )
        assert payload["stream"] is True
        assert "stream_options" not in payload


class TestParameterStripping:
    @pytest.mark.parametrize("param", sorted(OPENAI_ONLY_PARAMS))
    def test_openai_only_params_dropped(self, param: str) -> None:
        payload = build().build(messages=[], extra={param: "value"})
        assert param not in payload, (
            f"{param} is OpenAI-cloud-only; a strict server answers 400 "
            "unknown parameter"
        )

    @pytest.mark.parametrize(
        "param", ["api_key", "auth", "response_schema", "llm_params"]
    )
    def test_internal_params_never_reach_the_wire(self, param: str) -> None:
        assert param not in build().build(messages=[], extra={param: "x"})

    def test_none_values_dropped(self) -> None:
        assert "seed" not in build().build(messages=[], extra={"seed": None})

    def test_supported_params_forwarded(self) -> None:
        payload = build().build(messages=[], extra={"seed": 7, "user": "u1"})
        assert payload["seed"] == 7
        assert payload["user"] == "u1"

    @pytest.mark.parametrize(
        ("key", "neutral"),
        [("top_p", 1.0), ("frequency_penalty", 0.0), ("presence_penalty", 0.0)],
    )
    def test_neutral_sampling_defaults_omitted(self, key: str, neutral: float) -> None:
        assert key not in build().build(messages=[], extra={key: neutral})

    @pytest.mark.parametrize(
        ("key", "value"), [("top_p", 0.9), ("frequency_penalty", 0.5)]
    )
    def test_non_default_sampling_forwarded(self, key: str, value: float) -> None:
        assert build().build(messages=[], extra={key: value})[key] == value

    def test_drops_are_logged_at_debug(self, caplog) -> None:
        with caplog.at_level("DEBUG"):
            build().build(messages=[], extra={"store": True, "modalities": ["text"]})
        assert "unsupported by engine" in caplog.text


class TestExtraBody:
    def test_forwarded_when_engine_allows(self) -> None:
        payload = build().build(messages=[], extra={"extra_body": {"top_k": 40}})
        assert payload["extra_body"] == {"top_k": 40}

    def test_dropped_when_engine_disallows(self) -> None:
        payload = build(engine="generic", context_window=4_096).build(
            messages=[], extra={"extra_body": {"top_k": 40}}
        )
        assert "extra_body" not in payload

    @pytest.mark.parametrize("value", [{}, None, "not-a-dict"])
    def test_empty_or_invalid_ignored(self, value: object) -> None:
        assert "extra_body" not in build().build(
            messages=[], extra={"extra_body": value}
        )

    def test_copied_not_aliased(self) -> None:
        source = {"top_k": 40}
        payload = build().build(messages=[], extra={"extra_body": source})
        source["top_k"] = 1
        assert payload["extra_body"]["top_k"] == 40


class TestParallelToolCalls:
    def test_forwarded_when_supported_and_tools_present(self) -> None:
        payload = build().build(
            messages=[], tools=[TOOL], extra={"parallel_tool_calls": True}
        )
        assert payload["parallel_tool_calls"] is True

    def test_omitted_without_tools(self) -> None:
        payload = build().build(messages=[], extra={"parallel_tool_calls": True})
        assert "parallel_tool_calls" not in payload

    def test_warns_and_drops_when_unsupported(self) -> None:
        builder = build(supports_parallel_tool_calls=False)
        with pytest.warns(UserWarning, match="parallel_tool_calls"):
            payload = builder.build(
                messages=[], tools=[TOOL], extra={"parallel_tool_calls": True}
            )
        assert "parallel_tool_calls" not in payload

    def test_raises_in_strict_mode(self) -> None:
        builder = build(supports_parallel_tool_calls=False, strict_capabilities=True)
        with pytest.raises(ValueError, match="parallel_tool_calls"):
            builder.build(
                messages=[], tools=[TOOL], extra={"parallel_tool_calls": True}
            )

    def test_false_is_not_forwarded(self) -> None:
        payload = build().build(
            messages=[], tools=[TOOL], extra={"parallel_tool_calls": False}
        )
        assert "parallel_tool_calls" not in payload


class TestReasoningParams:
    def test_reasoning_effort_forwarded_on_vllm(self) -> None:
        payload = build().build(messages=[], extra={"reasoning_effort": "high"})
        assert payload["reasoning_effort"] == "high", (
            "vLLM accepts reasoning_effort and maps it onto the chat "
            "template's thinking switch"
        )

    def test_reasoning_effort_stripped_on_incapable_engine(self) -> None:
        payload = build(engine="tgi", context_window=8_192).build(
            messages=[], extra={"reasoning_effort": "high"}
        )
        assert "reasoning_effort" not in payload

    def test_include_reasoning_needs_reasoning_support(self) -> None:
        assert "include_reasoning" in build().build(
            messages=[], extra={"include_reasoning": True}
        )
        assert "include_reasoning" not in build(
            engine="tgi", context_window=8_192
        ).build(messages=[], extra={"include_reasoning": True})

    def test_reasoning_params_are_not_in_the_openai_only_set(self) -> None:
        assert not (REASONING_PARAMS & OPENAI_ONLY_PARAMS)


class TestChatTemplateKwargs:
    def test_merged_into_extra_body(self) -> None:
        payload = build(chat_template_kwargs={"enable_thinking": True}).build(
            messages=[]
        )
        assert payload["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": True
        }

    def test_combined_with_call_level_extra_body(self) -> None:
        payload = build(chat_template_kwargs={"enable_thinking": True}).build(
            messages=[], extra={"extra_body": {"top_k": 40}}
        )
        assert payload["extra_body"]["top_k"] == 40
        assert payload["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": True
        }

    def test_per_call_keys_win(self) -> None:
        payload = build(chat_template_kwargs={"enable_thinking": True}).build(
            messages=[],
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
        assert payload["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": False
        }

    def test_not_sent_when_engine_disallows_extra_body(self) -> None:
        payload = build(
            engine="generic",
            context_window=4_096,
            chat_template_kwargs={"enable_thinking": True},
        ).build(messages=[])
        assert "extra_body" not in payload
