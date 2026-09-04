"""End-to-end over real HTTP, with no client-level fakes.

These are the only tests that exercise the ``openai`` SDK, URL resolution,
header transmission, SSE framing and real HTTP status codes.  Everything
below the model is real; only the model's answer is scripted.
"""

from __future__ import annotations

import json

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.errors import (
    AuthenticationError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderConnectionError,
)
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools import tool
from nucleusiq_openai_compatible import HeaderAuth, OpenAICompatibleLLM

from .loopback import LoopbackServer, chat_response, sse_chunk

MODEL = "test-model"


@pytest.fixture
def server():
    with LoopbackServer() as running:
        yield running


def build(server: LoopbackServer, **overrides) -> OpenAICompatibleLLM:
    kwargs = {
        "base_url": server.base_url,
        "model": MODEL,
        "engine": "vllm",
        "context_window": 32_768,
        "max_retries": 0,
        "timeout": 10.0,
    }
    kwargs.update(overrides)
    return OpenAICompatibleLLM(**kwargs)


@tool("add")
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class TestRouting:
    async def test_request_reaches_the_right_path(self, server) -> None:
        server.queue(body=chat_response("hi"))
        await build(server).call(messages=[{"role": "user", "content": "hi"}])
        assert server.completions[0].path == "/v1/chat/completions"

    async def test_base_url_without_v1_is_normalized_on_the_wire(self, server) -> None:
        server.queue(body=chat_response("hi"))
        llm = build(server, base_url=server.root_url)
        await llm.call(messages=[{"role": "user", "content": "hi"}])
        assert server.completions[0].path == "/v1/chat/completions", (
            "appending /v1 must survive all the way to the request line, not "
            "just the configured string"
        )

    async def test_unreachable_server_is_typed(self) -> None:
        llm = OpenAICompatibleLLM(
            base_url="http://127.0.0.1:1/v1",
            model=MODEL,
            context_window=4_096,
            max_retries=0,
            timeout=2.0,
        )
        with pytest.raises(ProviderConnectionError):
            await llm.call(messages=[{"role": "user", "content": "hi"}])


class TestPayloadOnTheWire:
    async def test_body_is_what_we_built(self, server) -> None:
        server.queue(body=chat_response("hi"))
        await build(server).call(
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=64,
            temperature=0.2,
        )
        body = server.completions[0].body
        assert body["model"] == MODEL
        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.2
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert "stream" not in body

    async def test_openai_only_params_never_leave_the_process(self, server) -> None:
        server.queue(body=chat_response("hi"))
        await build(server).call(
            messages=[{"role": "user", "content": "hi"}],
            store=True,
            service_tier="scale",
        )
        body = server.completions[0].body
        assert "store" not in body and "service_tier" not in body

    async def test_extra_body_is_flattened_into_the_json(self, server) -> None:
        server.queue(body=chat_response("hi"))
        await build(server).call(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"top_k": 40, "min_p": 0.05},
        )
        body = server.completions[0].body
        assert body["top_k"] == 40, (
            "the SDK merges extra_body into the top-level JSON; if this ever "
            "nests it under 'extra_body', vLLM would ignore the sampler"
        )
        assert body["min_p"] == 0.05

    async def test_chat_template_kwargs_arrive(self, server) -> None:
        server.queue(body=chat_response("hi"))
        llm = build(
            server,
            is_reasoning_model=True,
            chat_template_kwargs={"enable_thinking": True},
        )
        await llm.call(messages=[{"role": "user", "content": "hi"}])
        assert server.completions[0].body["chat_template_kwargs"] == {
            "enable_thinking": True
        }


class TestCredentialsOnTheWire:
    async def test_no_auth_sends_no_bearer(self, server) -> None:
        server.queue(body=chat_response("hi"))
        await build(server).call(messages=[{"role": "user", "content": "hi"}])
        sent = server.completions[0].header("authorization")
        # The SDK always sends its placeholder; what matters is that no real
        # credential was invented.
        assert sent in (None, "Bearer EMPTY")

    async def test_bearer_token_arrives(self, server) -> None:
        server.queue(body=chat_response("hi"))
        llm = build(server, api_key="token-abc123")
        await llm.call(messages=[{"role": "user", "content": "hi"}])
        assert server.completions[0].header("authorization") == "Bearer token-abc123"

    async def test_custom_header_arrives(self, server) -> None:
        server.queue(body=chat_response("hi"))
        llm = build(server, auth=HeaderAuth("api-key", "azure-secret"))
        await llm.call(messages=[{"role": "user", "content": "hi"}])
        assert server.completions[0].header("api-key") == "azure-secret"

    async def test_rotating_key_changes_between_requests(self, server) -> None:
        server.queue(body=chat_response("hi"))
        keys = iter(["key-1", "key-2"])
        llm = build(server, api_key=lambda: next(keys))
        await llm.call(messages=[{"role": "user", "content": "hi"}])
        await llm.call(messages=[{"role": "user", "content": "hi"}])
        seen = [r.header("authorization") for r in server.completions]
        assert seen == ["Bearer key-1", "Bearer key-2"]

    async def test_default_headers_survive_alongside_auth(self, server) -> None:
        server.queue(body=chat_response("hi"))
        llm = build(
            server,
            auth=HeaderAuth("X-API-Key", "gw-key"),
            default_headers={"X-Team": "platform"},
        )
        await llm.call(messages=[{"role": "user", "content": "hi"}])
        request = server.completions[0]
        assert request.header("x-api-key") == "gw-key"
        assert request.header("x-team") == "platform"


class TestResponseParsing:
    async def test_content_and_usage(self, server) -> None:
        server.queue(body=chat_response("the answer"))
        result = await build(server).call(messages=[{"role": "user", "content": "hi"}])
        assert result.content == "the answer"
        assert result.prompt_tokens == 11
        assert result.usage_reported

    async def test_tool_call_round_trip(self, server) -> None:
        server.queue(
            body=chat_response(
                None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": json.dumps({"a": 2, "b": 3}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )
        )
        result = await build(server).call(
            messages=[{"role": "user", "content": "add 2 and 3"}],
            tools=[{"name": "add", "parameters": {"type": "object"}}],
        )
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "add"
        assert json.loads(result.tool_calls[0].arguments) == {"a": 2, "b": 3}

    async def test_reasoning_field_survives_the_sdk(self, server) -> None:
        # The SDK's typed models drop unknown fields unless they are kept as
        # extras; reasoning must still be readable.
        server.queue(body=chat_response("42", reasoning="first I thought..."))
        result = await build(server, is_reasoning_model=True).call(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert result.reasoning == "first I thought..."
        assert result.content == "42"

    async def test_missing_usage_block(self, server) -> None:
        body = chat_response("hi")
        del body["usage"]
        server.queue(body=body)
        result = await build(server).call(messages=[{"role": "user", "content": "hi"}])
        assert result.usage_reported is False


class TestRealServerSentEvents:
    async def test_tokens_stream_through(self, server) -> None:
        server.queue(
            sse=[
                sse_chunk(content="Hel"),
                sse_chunk(content="lo"),
                sse_chunk(content="", finish_reason="stop"),
            ]
        )
        llm = build(server)
        tokens = [
            event.token
            async for event in llm.call_stream(
                messages=[{"role": "user", "content": "hi"}]
            )
            if event.type == "token"
        ]
        assert "".join(tokens) == "Hello"

    async def test_stream_options_requested(self, server) -> None:
        server.queue(sse=[sse_chunk(content="hi", finish_reason="stop")])
        llm = build(server)
        async for _ in llm.call_stream(messages=[{"role": "user", "content": "hi"}]):
            pass
        assert server.completions[0].body["stream_options"] == {"include_usage": True}

    async def test_usage_trailer_parsed(self, server) -> None:
        server.queue(
            sse=[
                sse_chunk(content="hi", finish_reason="stop"),
                sse_chunk(
                    usage={
                        "prompt_tokens": 9,
                        "completion_tokens": 3,
                        "total_tokens": 12,
                    }
                ),
            ]
        )
        llm = build(server)
        async for _ in llm.call_stream(messages=[{"role": "user", "content": "hi"}]):
            pass
        assert llm.last_stream.response.total_tokens == 12

    async def test_fragmented_tool_call_over_sse(self, server) -> None:
        server.queue(
            sse=[
                sse_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a":'},
                        }
                    ]
                ),
                sse_chunk(
                    tool_calls=[{"index": 0, "function": {"arguments": '2,"b":3}'}}]
                ),
                sse_chunk(content="", finish_reason="tool_calls"),
            ]
        )
        llm = build(server)
        async for _ in llm.call_stream(
            messages=[{"role": "user", "content": "add"}],
            tools=[{"name": "add", "parameters": {"type": "object"}}],
        ):
            pass
        (call,) = llm.last_stream.response.tool_calls
        assert json.loads(call.arguments) == {"a": 2, "b": 3}

    async def test_reasoning_deltas_tagged(self, server) -> None:
        server.queue(
            sse=[
                sse_chunk(reasoning="thinking..."),
                sse_chunk(content="42", finish_reason="stop"),
            ]
        )
        llm = build(server, is_reasoning_model=True)
        events = [
            event
            async for event in llm.call_stream(
                messages=[{"role": "user", "content": "hi"}]
            )
            if event.type == "token"
        ]
        tagged = [e.token for e in events if (e.metadata or {}).get("reasoning")]
        plain = [e.token for e in events if not (e.metadata or {}).get("reasoning")]
        assert tagged == ["thinking..."]
        assert plain == ["42"]


class TestRealHttpErrors:
    async def test_401(self, server) -> None:
        server.queue(status=401, body={"error": {"message": "invalid api key"}})
        with pytest.raises(AuthenticationError):
            await build(server).call(messages=[{"role": "user", "content": "hi"}])

    async def test_404(self, server) -> None:
        server.queue(status=404, body={"error": {"message": "model not found"}})
        with pytest.raises(ModelNotFoundError):
            await build(server).call(messages=[{"role": "user", "content": "hi"}])

    async def test_400_context_overflow(self, server) -> None:
        server.queue(
            status=400,
            body={
                "error": {
                    "message": (
                        "This model's maximum context length is 32768 tokens. "
                        "However, you requested 40000 tokens."
                    )
                }
            },
        )
        with pytest.raises(ContextLengthError):
            await build(server).call(messages=[{"role": "user", "content": "hi"}])

    async def test_credentials_absent_from_a_real_error(self, server) -> None:
        server.queue(
            status=400,
            body={"error": {"message": "rejected Authorization: Bearer token-abc123"}},
        )
        llm = build(server, api_key="token-abc123")
        with pytest.raises(Exception) as caught:
            await llm.call(messages=[{"role": "user", "content": "hi"}])
        assert "token-abc123" not in str(caught.value)


class TestProbeOverHttp:
    async def test_context_window_discovered(self, server) -> None:
        server.queue(body=chat_response("hi"))
        llm = build(server, context_window=None)
        assert llm.get_context_window() == 8_192
        await llm.call(messages=[{"role": "user", "content": "hi"}])
        assert llm.get_context_window() == 32_768
        assert server.model_lists, "the probe must actually hit /v1/models"

    async def test_validate_reports_a_healthy_server(self, server) -> None:
        report = await build(server).validate()
        assert report.ok
        assert report.reachable
        assert MODEL in report.served_models

    async def test_validate_reports_a_wrong_model(self, server) -> None:
        server.model_cards = [{"id": "some-other-model", "max_model_len": 4_096}]
        report = await build(server, model="test-model").validate()
        assert not report.ok
        assert "some-other-model" in report.errors[0]

    async def test_validate_reports_an_unreachable_server(self) -> None:
        llm = OpenAICompatibleLLM(
            base_url="http://127.0.0.1:1/v1",
            model=MODEL,
            context_window=4_096,
            max_retries=0,
            timeout=2.0,
        )
        report = await llm.validate()
        assert not report.ok
        assert not report.reachable


class TestAgentOverHttp:
    async def test_full_agent_run(self, server) -> None:
        server.queue(body=chat_response("Paris."))
        agent = Agent(
            name="HttpAgent",
            prompt=PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
                system="You are terse."
            ),
            llm=build(server),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Capital of France?"))
        assert "Paris" in str(result.output)

    async def test_agent_tool_loop_over_http(self, server) -> None:
        server.queue(
            body=chat_response(
                None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": json.dumps({"a": 2, "b": 3}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )
        )
        server.queue(body=chat_response("The sum is 5."))

        agent = Agent(
            name="HttpToolAgent",
            prompt=PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
                system="Use the tools."
            ),
            llm=build(server),
            tools=[add],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t2", objective="Add 2 and 3"))

        assert len(server.completions) >= 2
        followup = server.completions[-1].body["messages"]
        assert any(m.get("role") == "tool" for m in followup), (
            "the tool result must be echoed back over the wire"
        )
        assert "5" in str(result.output)
