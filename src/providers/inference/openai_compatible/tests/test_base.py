"""``OpenAICompatibleLLM`` — construction, calls, streaming and credentials."""

from __future__ import annotations

import pytest
from conftest import (
    BASE_URL,
    MODEL,
    FakeChunk,
    FakeDelta,
    FakeMessage,
    FakeResponse,
    FakeToolCall,
    FakeUsage,
    drain,
)
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.llms.errors import InvalidRequestError, ModelNotFoundError
from nucleusiq_openai_compatible import OpenAICompatibleLLM
from nucleusiq_openai_compatible._shared.model_probe import ModelProbe, ProbeResult
from nucleusiq_openai_compatible.auth import BearerAuth, HeaderAuth
from nucleusiq_openai_compatible.llm_params import OpenAICompatibleLLMParams

MESSAGES = [{"role": "user", "content": "hello"}]
TOOL = {
    "name": "search",
    "description": "Search the web",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
}


async def astream(items):
    for item in items:
        yield item


class StubProbe(ModelProbe):
    def __init__(self, result: ProbeResult) -> None:  # noqa: D107
        self._result = result
        self._calls = 0

    async def probe(self, model: str | None = None) -> ProbeResult:
        self._calls += 1
        return self._result

    @property
    def cached(self) -> ProbeResult:
        return self._result


class TestContract:
    def test_is_a_base_llm(self, make_llm) -> None:
        assert isinstance(make_llm(), BaseLLM)

    def test_no_native_tool_types(self) -> None:
        assert frozenset() == OpenAICompatibleLLM.NATIVE_TOOL_TYPES, (
            "a self-hosted server runs no tools itself, so nothing can route "
            "away from Chat Completions"
        )

    def test_model_name_is_authoritative(self, make_llm) -> None:
        assert make_llm().model_name == MODEL

    def test_base_url_normalized_on_the_instance(self, make_llm) -> None:
        assert make_llm(base_url="http://gpu-node-1:8000").base_url == BASE_URL

    def test_capabilities_exposed(self, make_llm) -> None:
        assert make_llm().capabilities.engine == "vllm"

    def test_repr_excludes_credentials(self, make_llm) -> None:
        text = repr(make_llm(api_key="token-abc123"))
        assert "token-abc123" not in text
        assert "auth=<redacted>" in text
        assert MODEL in text


class TestProviderDeclaration:
    """Identity and structured-output support are declared, never inferred."""

    def test_declares_its_own_identity(self) -> None:
        assert OpenAICompatibleLLM.PROVIDER_NAME == "openai_compatible"

    def test_identity_is_not_openai_cloud(self) -> None:
        assert OpenAICompatibleLLM.PROVIDER_NAME != "openai", (
            "sharing OpenAI's wire dialect does not mean sharing its guarantees"
        )

    @pytest.mark.parametrize("engine", ["vllm", "sglang", "lmstudio"])
    def test_schema_capable_engines(self, make_llm, engine: str) -> None:
        assert make_llm(engine=engine).supports_native_structured_output is True

    @pytest.mark.parametrize("engine", ["generic", "tgi", "llamacpp"])
    def test_schema_incapable_engines(self, make_llm, engine: str) -> None:
        assert make_llm(engine=engine).supports_native_structured_output is False

    def test_explicit_override_beats_the_preset(self, make_llm) -> None:
        llm = make_llm(engine="generic", supports_json_schema=True)
        assert llm.supports_native_structured_output is True

    def test_capability_is_per_instance(self, make_llm) -> None:
        """Two deployments of the same class can disagree."""
        assert make_llm(engine="vllm").supports_native_structured_output
        assert not make_llm(engine="tgi").supports_native_structured_output


class TestConstruction:
    def test_model_alias_accepted(self) -> None:
        llm = OpenAICompatibleLLM(base_url=BASE_URL, model="m", context_window=4_096)
        assert llm.model_name == "m"

    def test_model_name_alias_accepted(self) -> None:
        llm = OpenAICompatibleLLM(
            base_url=BASE_URL, model_name="m", context_window=4_096
        )
        assert llm.model_name == "m"

    def test_agreeing_aliases_are_fine(self) -> None:
        llm = OpenAICompatibleLLM(
            base_url=BASE_URL, model_name="m", model="m", context_window=4_096
        )
        assert llm.model_name == "m"

    def test_disagreeing_aliases_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="disagree"):
            OpenAICompatibleLLM(
                base_url=BASE_URL, model_name="a", model="b", context_window=4_096
            )

    def test_env_fallbacks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://env-host:8000")
        monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "env-model")
        llm = OpenAICompatibleLLM(context_window=4_096)
        assert llm.base_url == "http://env-host:8000/v1"
        assert llm.model_name == "env-model"

    def test_missing_base_url_is_actionable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
        with pytest.raises(InvalidRequestError, match="base_url is required"):
            OpenAICompatibleLLM(model="m")

    def test_no_api_key_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
        llm = OpenAICompatibleLLM(base_url=BASE_URL, model="m", context_window=4_096)
        assert llm is not None, (
            "an unauthenticated local vLLM server is the common case; "
            "requiring a key would be the OpenAI provider's bug repeated"
        )

    def test_conflicting_credentials_rejected(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            OpenAICompatibleLLM(
                base_url=BASE_URL,
                model="m",
                context_window=4_096,
                api_key="k-value",
                auth=HeaderAuth("api-key", "v-value"),
            )

    def test_strict_from_llm_params(self) -> None:
        llm = OpenAICompatibleLLM(
            base_url=BASE_URL,
            model="m",
            context_window=4_096,
            llm_params=OpenAICompatibleLLMParams(strict_capabilities=True),
        )
        assert llm.capabilities.strict_capabilities


class TestContextWindow:
    def test_explicit_needs_no_probe(self, make_llm, fake_client) -> None:
        llm = make_llm(context_window=32_768)
        assert llm.get_context_window() == 32_768

    async def test_probe_fills_it_in(self, make_llm) -> None:
        llm = make_llm(context_window=None)
        llm._probe = StubProbe(
            ProbeResult(reachable=True, model_ids=(MODEL,), context_window=40_960)
        )
        assert llm.get_context_window() == 8_192
        await llm.call(messages=MESSAGES)
        assert llm.get_context_window() == 40_960

    async def test_probe_runs_once(self, make_llm) -> None:
        llm = make_llm(context_window=None)
        probe = StubProbe(ProbeResult(reachable=True, context_window=16_384))
        llm._probe = probe
        await llm.call(messages=MESSAGES)
        await llm.call(messages=MESSAGES)
        assert probe._calls == 1

    async def test_probe_skipped_when_window_explicit(self, make_llm) -> None:
        llm = make_llm(context_window=32_768)
        probe = StubProbe(ProbeResult(reachable=True, context_window=99))
        llm._probe = probe
        await llm.call(messages=MESSAGES)
        assert probe._calls == 0, "an explicit window means no round-trip"

    async def test_probe_disabled(self, make_llm) -> None:
        llm = make_llm(context_window=None, probe_context_window=False)
        llm._probe = StubProbe(ProbeResult(reachable=True, context_window=99_999))
        await llm.call(messages=MESSAGES)
        assert llm.get_context_window() == 8_192

    async def test_unreachable_probe_keeps_the_floor(self, make_llm) -> None:
        llm = make_llm(context_window=None)
        llm._probe = StubProbe(ProbeResult(reachable=False, error="refused"))
        await llm.call(messages=MESSAGES)
        assert llm.get_context_window() == 8_192


class TestModelValidation:
    async def test_served_model_passes(self, make_llm) -> None:
        llm = make_llm(validate_model=True)
        llm._probe = StubProbe(ProbeResult(reachable=True, model_ids=(MODEL,)))
        assert (await llm.call(messages=MESSAGES)).content == "hello"

    async def test_unserved_model_lists_alternatives(self, make_llm) -> None:
        llm = make_llm(validate_model=True)
        llm._probe = StubProbe(
            ProbeResult(reachable=True, model_ids=("llama-3", "qwen3"))
        )
        with pytest.raises(ModelNotFoundError) as exc:
            await llm.call(messages=MESSAGES)
        assert "llama-3" in str(exc.value) and "qwen3" in str(exc.value)

    async def test_unreachable_server_does_not_block(self, make_llm) -> None:
        llm = make_llm(validate_model=True)
        llm._probe = StubProbe(ProbeResult(reachable=False, error="refused"))
        assert await llm.call(messages=MESSAGES) is not None

    async def test_off_by_default(self, make_llm) -> None:
        llm = make_llm()
        llm._probe = StubProbe(ProbeResult(reachable=True, model_ids=("other",)))
        assert await llm.call(messages=MESSAGES) is not None


class TestPerCallModel:
    async def test_matching_model_allowed(self, make_llm) -> None:
        assert await make_llm().call(model=MODEL, messages=MESSAGES) is not None

    async def test_none_allowed(self, make_llm) -> None:
        assert await make_llm().call(model=None, messages=MESSAGES) is not None

    async def test_switch_rejected(self, make_llm) -> None:
        with pytest.raises(InvalidRequestError, match="(?i)one instance serves"):
            await make_llm().call(model="other-model", messages=MESSAGES)

    async def test_switch_rejected_when_streaming(self, make_llm) -> None:
        with pytest.raises(InvalidRequestError):
            await drain(make_llm().call_stream(model="other", messages=MESSAGES))


class TestCall:
    async def test_returns_normalized_content(self, make_llm, fake_client) -> None:
        fake_client.result = FakeResponse(FakeMessage("the answer"))
        result = await make_llm().call(messages=MESSAGES)
        assert result.content == "the answer"
        assert result.finish_reason == "stop"

    async def test_payload_shape(self, make_llm, fake_client) -> None:
        await make_llm().call(messages=MESSAGES, max_output_tokens=64)
        payload = fake_client.payloads[-1]
        assert payload["model"] == MODEL
        assert payload["messages"] == MESSAGES
        assert payload["max_tokens"] == 64
        assert "stream" not in payload

    async def test_tools_converted(self, make_llm, fake_client) -> None:
        await make_llm().call(messages=MESSAGES, tools=[TOOL])
        tool = fake_client.payloads[-1]["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "search"

    async def test_tool_calls_normalized(self, make_llm, fake_client) -> None:
        fake_client.result = FakeResponse(
            FakeMessage(None, tool_calls=[FakeToolCall(name="search")])
        )
        result = await make_llm().call(messages=MESSAGES, tools=[TOOL])
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "search"

    async def test_llm_params_merged_and_overridable(
        self, make_llm, fake_client
    ) -> None:
        llm = make_llm(llm_params=OpenAICompatibleLLMParams(seed=7, user="svc"))
        await llm.call(messages=MESSAGES)
        assert fake_client.payloads[-1]["seed"] == 7
        await llm.call(messages=MESSAGES, seed=99)
        assert fake_client.payloads[-1]["seed"] == 99

    async def test_usage_recorded(self, make_llm, fake_client) -> None:
        fake_client.result = FakeResponse(usage=FakeUsage(100, 20, 120))
        result = await make_llm().call(messages=MESSAGES)
        assert (result.prompt_tokens, result.completion_tokens) == (100, 20)

    async def test_missing_usage_is_noted_not_fatal(
        self, make_llm, fake_client, caplog
    ) -> None:
        fake_client.result = FakeResponse(usage=None)
        with caplog.at_level("DEBUG"):
            result = await make_llm().call(messages=MESSAGES)
        assert result.usage_reported is False
        assert "no usage" in caplog.text


class TestStructuredOutput:
    SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}

    async def test_native_schema_without_tools(self, make_llm, fake_client) -> None:
        await make_llm().call(messages=MESSAGES, response_schema=self.SCHEMA)
        assert fake_client.payloads[-1]["response_format"]["type"] == "json_schema"

    async def test_prompt_mode_keeps_tools_working(self, make_llm, fake_client) -> None:
        await make_llm().call(
            messages=MESSAGES, tools=[TOOL], response_schema=self.SCHEMA
        )
        payload = fake_client.payloads[-1]
        assert "response_format" not in payload, (
            "vLLM suppresses tool calls when response_format is present"
        )
        assert payload["tools"], "the tools must survive"
        assert any("answer" in str(m.get("content")) for m in payload["messages"])

    async def test_instruction_merges_into_existing_system_message(
        self, make_llm, fake_client
    ) -> None:
        messages = [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
        ]
        await make_llm().call(
            messages=messages, tools=[TOOL], response_schema=self.SCHEMA
        )
        sent = fake_client.payloads[-1]["messages"]
        assert len([m for m in sent if m["role"] == "system"]) == 1
        assert "You are terse." in sent[0]["content"]
        assert "answer" in sent[0]["content"]

    async def test_caller_messages_not_mutated(self, make_llm) -> None:
        messages = [{"role": "user", "content": "hi"}]
        await make_llm().call(
            messages=messages, tools=[TOOL], response_schema=self.SCHEMA
        )
        assert messages == [{"role": "user", "content": "hi"}]

    async def test_error_mode_raises_before_the_call(
        self, make_llm, fake_client
    ) -> None:
        llm = make_llm(structured_output_with_tools="error")
        with pytest.raises(InvalidRequestError, match="cannot be combined with tools"):
            await llm.call(messages=MESSAGES, tools=[TOOL], response_schema=self.SCHEMA)
        assert fake_client.payloads == [], "must fail before any HTTP traffic"

    async def test_schema_keys_never_reach_the_wire(
        self, make_llm, fake_client
    ) -> None:
        await make_llm().call(messages=MESSAGES, response_schema=self.SCHEMA)
        payload = fake_client.payloads[-1]
        assert "response_schema" not in payload
        assert "response_schema_name" not in payload


class TestCredentials:
    async def test_bearer_header_sent(self, make_llm, fake_client) -> None:
        await make_llm(api_key="token-abc123").call(messages=MESSAGES)
        assert fake_client.with_options_calls, "credentials must be applied per call"
        assert fake_client.with_options_calls[-1]["api_key"] == "token-abc123"

    async def test_custom_header_sent(self, make_llm, fake_client) -> None:
        llm = make_llm(auth=HeaderAuth("api-key", "azure-secret"))
        await llm.call(messages=MESSAGES)
        headers = fake_client.with_options_calls[-1]["default_headers"]
        assert headers["api-key"] == "azure-secret"

    async def test_no_auth_skips_with_options(self, make_llm, fake_client) -> None:
        await make_llm().call(messages=MESSAGES)
        assert fake_client.with_options_calls == []

    async def test_callable_key_resolved_each_call(self, make_llm, fake_client) -> None:
        keys = iter(["key-1", "key-2"])
        llm = make_llm(api_key=lambda: next(keys))
        await llm.call(messages=MESSAGES)
        await llm.call(messages=MESSAGES)
        used = [c["api_key"] for c in fake_client.with_options_calls]
        assert used == ["key-1", "key-2"], (
            "rotation must take effect without rebuilding the agent"
        )

    async def test_credential_read_exactly_once_per_call(
        self, make_llm, fake_client
    ) -> None:
        reads = {"n": 0}

        def mint() -> str:
            reads["n"] += 1
            return f"token-{reads['n']}"

        await make_llm(api_key=mint).call(messages=MESSAGES)
        assert reads["n"] == 1, (
            "a callable that mints a short-lived token must not be charged "
            "twice per request"
        )

    async def test_header_and_sdk_key_agree_under_rotation(
        self, make_llm, fake_client
    ) -> None:
        keys = iter(["key-1", "key-2"])
        await make_llm(api_key=lambda: next(keys)).call(messages=MESSAGES)
        options = fake_client.with_options_calls[-1]
        assert options["default_headers"]["Authorization"] == "Bearer key-1"
        assert options["api_key"] == "key-1", (
            "resolving the credential separately for the header and the SDK "
            "slot would send two different tokens"
        )

    async def test_per_call_key_overrides(self, make_llm, fake_client) -> None:
        llm = make_llm(api_key="default-key")
        await llm.call(messages=MESSAGES, api_key="tenant-key")
        assert fake_client.with_options_calls[-1]["api_key"] == "tenant-key"

    async def test_api_key_never_reaches_the_body(self, make_llm, fake_client) -> None:
        await make_llm().call(messages=MESSAGES, api_key="tenant-key")
        assert "api_key" not in fake_client.payloads[-1]

    async def test_auth_strategy_object(self, make_llm, fake_client) -> None:
        llm = make_llm(auth=BearerAuth("token-abc123"))
        await llm.call(messages=MESSAGES)
        assert fake_client.with_options_calls[-1]["api_key"] == "token-abc123"

    async def test_default_headers_preserved_alongside_auth(
        self, make_llm, fake_client
    ) -> None:
        llm = make_llm(
            auth=HeaderAuth("api-key", "s-value"),
            default_headers={"X-Tenant": "acme"},
        )
        await llm.call(messages=MESSAGES)
        headers = fake_client.with_options_calls[-1]["default_headers"]
        assert headers["X-Tenant"] == "acme"
        assert headers["api-key"] == "s-value"


class TestStreaming:
    async def test_tokens_then_complete(self, make_llm, fake_client) -> None:
        fake_client.result = lambda: astream(
            [
                FakeChunk(FakeDelta("Hel")),
                FakeChunk(FakeDelta("lo")),
                FakeChunk(FakeDelta(None), finish_reason="stop"),
            ]
        )
        events = await drain(make_llm().call_stream(messages=MESSAGES))
        tokens = [e.token for e in events if e.type == "token"]
        assert "".join(tokens) == "Hello"
        assert events[-1].type == "complete"
        assert events[-1].content == "Hello"

    async def test_stream_flag_and_usage_option(self, make_llm, fake_client) -> None:
        fake_client.result = lambda: astream([FakeChunk(FakeDelta("x"))])
        await drain(make_llm().call_stream(messages=MESSAGES))
        payload = fake_client.payloads[-1]
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}

    async def test_last_stream_holds_tool_calls(self, make_llm, fake_client) -> None:
        fake_client.result = lambda: astream(
            [
                FakeChunk(
                    FakeDelta(
                        None,
                        tool_calls=[
                            FakeToolCall(
                                index=0, id="c1", name="search", arguments='{"q":'
                            )
                        ],
                    )
                ),
                FakeChunk(
                    FakeDelta(
                        None,
                        tool_calls=[
                            FakeToolCall(index=0, id=None, name=None, arguments='"x"}')
                        ],
                    )
                ),
                FakeChunk(FakeDelta(None), finish_reason="tool_calls"),
            ]
        )
        llm = make_llm()
        await drain(llm.call_stream(messages=MESSAGES, tools=[TOOL]))
        calls = llm.last_stream.response.tool_calls
        assert len(calls) == 1
        assert calls[0].arguments == '{"q":"x"}', (
            "fragmented argument deltas must be reassembled before execution"
        )

    async def test_last_stream_is_none_before_streaming(self, make_llm) -> None:
        assert make_llm().last_stream is None

    async def test_usage_trailer_chunk(self, make_llm, fake_client) -> None:
        fake_client.result = lambda: astream(
            [
                FakeChunk(FakeDelta("hi")),
                FakeChunk(None, usage=FakeUsage(10, 2, 12)),
            ]
        )
        llm = make_llm()
        await drain(llm.call_stream(messages=MESSAGES))
        assert llm.last_stream.response.prompt_tokens == 10


class TestReasoning:
    async def test_flag_defaults_off(self, make_llm) -> None:
        assert make_llm().is_reasoning_model is False

    async def test_flag_declared(self, make_llm) -> None:
        assert make_llm(is_reasoning_model=True).is_reasoning_model is True

    async def test_reasoning_kept_out_of_content(self, make_llm, fake_client) -> None:
        fake_client.result = FakeResponse(
            FakeMessage("42", reasoning="first I considered...")
        )
        result = await make_llm(is_reasoning_model=True).call(messages=MESSAGES)
        assert result.content == "42"
        assert result.reasoning == "first I considered..."

    async def test_undeclared_reasoning_warns_once(
        self, make_llm, fake_client, caplog
    ) -> None:
        fake_client.result = FakeResponse(FakeMessage("42", reasoning="thinking"))
        llm = make_llm()
        with caplog.at_level("WARNING"):
            await llm.call(messages=MESSAGES)
            await llm.call(messages=MESSAGES)
        assert caplog.text.count("is_reasoning_model=True") == 1, (
            "a per-call warning would flood the logs of a long agent run"
        )

    async def test_declared_reasoning_does_not_warn(
        self, make_llm, fake_client, caplog
    ) -> None:
        fake_client.result = FakeResponse(FakeMessage("42", reasoning="thinking"))
        with caplog.at_level("WARNING"):
            await make_llm(is_reasoning_model=True).call(messages=MESSAGES)
        assert "is_reasoning_model=True" not in caplog.text

    async def test_reasoning_only_trap_is_explained(
        self, make_llm, fake_client, caplog
    ) -> None:
        fake_client.result = FakeResponse(
            FakeMessage(None, reasoning="the entire answer landed here")
        )
        with caplog.at_level("WARNING"):
            await make_llm(is_reasoning_model=True).call(messages=MESSAGES)
        assert "--reasoning-parser" in caplog.text
        assert "enable_thinking" in caplog.text

    async def test_chat_template_kwargs_sent(self, make_llm, fake_client) -> None:
        llm = make_llm(
            is_reasoning_model=True, chat_template_kwargs={"enable_thinking": True}
        )
        await llm.call(messages=MESSAGES)
        extra = fake_client.payloads[-1]["extra_body"]
        assert extra["chat_template_kwargs"] == {"enable_thinking": True}

    async def test_reasoning_effort_forwarded(self, make_llm, fake_client) -> None:
        await make_llm(is_reasoning_model=True).call(
            messages=MESSAGES, reasoning_effort="high"
        )
        assert fake_client.payloads[-1]["reasoning_effort"] == "high"


class TestTokenEstimation:
    def test_heuristic_by_default(self, make_llm) -> None:
        assert make_llm().estimate_tokens("a" * 400) == 100

    def test_injected_counter_used(self, make_llm) -> None:
        class Counter:
            method = "tokenizer"

            def count(self, text: str) -> int:
                return 7

        assert make_llm(token_counter=Counter()).estimate_tokens("anything") == 7

    def test_method_recorded_in_config(self, make_llm) -> None:
        class Counter:
            method = "tokenizer"

            def count(self, text: str) -> int:
                return 1

        llm = make_llm(token_counter=Counter(), tokenizer="google/gemma-4-27b-it")
        assert llm.capabilities.token_count_method == "tokenizer"


class TestValidate:
    async def test_healthy_endpoint(self, make_llm) -> None:
        llm = make_llm()
        llm._probe = StubProbe(
            ProbeResult(reachable=True, model_ids=(MODEL,), context_window=32_768)
        )
        report = await llm.validate()
        assert report.reachable
        assert report.ok

    async def test_unreachable_reports_rather_than_raises(self, make_llm) -> None:
        llm = make_llm()
        llm._probe = StubProbe(ProbeResult(reachable=False, error="connection refused"))
        report = await llm.validate()
        assert not report.reachable
        assert not report.ok
        assert any("refused" in e for e in report.errors)

    async def test_missing_model_is_an_error_not_an_exception(self, make_llm) -> None:
        llm = make_llm()
        llm._probe = StubProbe(ProbeResult(reachable=True, model_ids=("other",)))
        report = await llm.validate()
        assert not report.ok
        assert any("other" in e for e in report.errors)
