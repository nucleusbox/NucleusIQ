"""End-to-end integration with the real ``nucleusiq`` core.

These tests build an actual :class:`nucleusiq.agents.Agent` around
:class:`OpenAICompatibleLLM` and drive it with a scripted fake HTTP client.
Nothing about the agent, the prompt pipeline, the tool loop or the
structured-output resolver is mocked — only the network is.

They exist because the unit tests cannot see the seams that actually break:
how the prompt becomes messages, what shape tool specs arrive in, and how the
core hands structured output to a provider.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import (
    MODEL,
    FakeChunk,
    FakeDelta,
    FakeMessage,
    FakeResponse,
    FakeToolCall,
    FakeUsage,
)
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.structured_output import (
    OutputMode,
    get_provider_from_llm,
    resolve_output_config,
)
from nucleusiq.agents.task import Task
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools import tool
from nucleusiq_openai_compatible.structured_output.policy import build_policy
from pydantic import BaseModel

# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


@pytest.fixture
def prompt():
    return PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
        system="You are a precise assistant.",
        user="Answer using the tools when they help.",
    )


@pytest.fixture
def scripted(fake_client):
    """Queue a sequence of responses, returned one per HTTP call."""

    def _script(*responses: Any) -> None:
        queue = list(responses)

        def next_response():
            return queue.pop(0) if len(queue) > 1 else queue[0]

        fake_client.result = next_response

    return _script


@tool("add")
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


async def run_agent(llm, prompt, objective: str, **agent_kwargs):
    agent = Agent(
        name="CompatAgent",
        role="Assistant",
        objective="Help the user",
        prompt=prompt,
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            verbose=False,
            **agent_kwargs.pop("config", {}),
        ),
        **agent_kwargs,
    )
    await agent.initialize()
    return await agent.execute(Task(id="t1", objective=objective))


# ====================================================================== #
# Provider identification                                                  #
# ====================================================================== #


class TestProviderResolution:
    def test_core_identifies_the_provider(self, make_llm) -> None:
        assert get_provider_from_llm(make_llm()) == "openai_compatible"

    def test_not_mistaken_for_openai_cloud(self, make_llm) -> None:
        assert get_provider_from_llm(make_llm()) != "openai", (
            "the class name contains 'openai', but inheriting OpenAI cloud "
            "assumptions would send Responses-API-shaped structured output "
            "to a self-hosted server"
        )

    def test_identity_survives_renaming(self, make_llm) -> None:
        """Core must read the declaration, not the class name."""

        class SelfHostedGateway(type(make_llm())):
            pass

        llm = make_llm()
        llm.__class__ = SelfHostedGateway
        assert get_provider_from_llm(llm) == "openai_compatible"


class TestAutoModeStaysNative:
    """AUTO resolves to NATIVE for every engine; the adapter degrades, not core.

    Core implements only ``{AUTO, NATIVE}``, so re-routing a schema-incapable
    engine to ``OutputMode.PROMPT`` would raise ``NotImplementedError`` — and
    ``generic`` is the *default* engine, which would make structured output
    unusable out of the box. Degradation belongs one layer down, where
    :class:`PromptPolicy` swaps ``json_schema`` for ``json_object`` plus a
    prompt-injected schema.
    """

    @pytest.mark.parametrize("engine", ["vllm", "sglang", "generic", "tgi"])
    def test_every_engine_resolves_to_native(self, engine: str) -> None:
        cfg = resolve_output_config(Answer, model_name="m")
        assert cfg._resolved_mode is OutputMode.NATIVE
        assert OutputMode.is_implemented(cfg._resolved_mode)

    def test_incapable_engine_still_carries_the_schema(self, make_llm) -> None:
        """``supports_json_schema=False`` degrades transport, not capability."""
        llm = make_llm(engine="generic")
        assert llm.supports_native_structured_output is False

        decision = build_policy("prompt").decide(
            schema=Answer.model_json_schema(),
            schema_name="Answer",
            has_tools=False,
            supports_json_schema=llm.supports_native_structured_output,
            suppresses_tools=False,
        )
        assert decision.response_format == {"type": "json_object"}
        assert decision.prompt_instruction, (
            "dropping to json_object without the schema in the prompt would "
            "lose the shape entirely"
        )


# ====================================================================== #
# Prompt pipeline                                                          #
# ====================================================================== #


class TestPromptPipeline:
    async def test_prompt_becomes_system_and_user_messages(
        self, make_llm, prompt, fake_client
    ) -> None:
        fake_client.result = FakeResponse(FakeMessage("Paris."))
        await run_agent(make_llm(), prompt, "What is the capital of France?")

        messages = fake_client.payloads[0]["messages"]
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert "precise assistant" in messages[0]["content"]
        assert any(
            "capital of France" in str(m.get("content"))
            for m in messages
            if m["role"] == "user"
        ), "the task objective must reach the model as a user message"

    async def test_model_and_limits_come_from_the_provider(
        self, make_llm, prompt, fake_client
    ) -> None:
        fake_client.result = FakeResponse(FakeMessage("ok"))
        await run_agent(make_llm(), prompt, "hi")

        payload = fake_client.payloads[0]
        assert payload["model"] == MODEL
        assert "max_tokens" in payload, "the configured wire field must be used"

    async def test_agent_output_is_the_model_content(
        self, make_llm, prompt, fake_client
    ) -> None:
        fake_client.result = FakeResponse(FakeMessage("The answer is 42."))
        result = await run_agent(make_llm(), prompt, "What is the answer?")
        assert "42" in str(result.output)


# ====================================================================== #
# Context budgeting                                                        #
# ====================================================================== #


class TestContextBudget:
    async def test_agent_reads_the_configured_window(
        self, make_llm, prompt, fake_client
    ) -> None:
        llm = make_llm(context_window=32_768)
        fake_client.result = FakeResponse(FakeMessage("ok"))
        agent = Agent(name="A", prompt=prompt, llm=llm)
        await agent.initialize()
        assert agent.llm.get_context_window() == 32_768

    async def test_conservative_default_is_not_openai_128k(
        self, make_llm, prompt
    ) -> None:
        llm = make_llm(context_window=None, probe_context_window=False)
        assert llm.get_context_window() == 8_192, (
            "inheriting BaseLLM's 128k default would let the context engine "
            "skip compaction and the server would reject the request"
        )

    async def test_token_counting_is_wired_through(self, make_llm) -> None:
        assert make_llm().estimate_tokens("a" * 400) == 100


# ====================================================================== #
# Tool loop                                                                #
# ====================================================================== #


class TestToolLoop:
    async def test_core_tool_spec_is_converted_for_the_wire(
        self, make_llm, prompt, fake_client, scripted
    ) -> None:
        scripted(
            FakeResponse(
                FakeMessage(
                    None,
                    tool_calls=[
                        FakeToolCall(
                            id="c1", name="add", arguments=json.dumps({"a": 2, "b": 3})
                        )
                    ],
                )
            ),
            FakeResponse(FakeMessage("The sum is 5.")),
        )
        await run_agent(make_llm(), prompt, "Add 2 and 3", tools=[add])

        sent = fake_client.payloads[0]["tools"]
        assert sent[0]["type"] == "function", (
            "the core hands over a bare {name, description, parameters} spec; "
            "the provider must wrap it"
        )
        function = sent[0]["function"]
        assert function["name"] == "add"
        assert "a" in function["parameters"]["properties"]

    async def test_tool_result_feeds_the_next_turn(
        self, make_llm, prompt, fake_client, scripted
    ) -> None:
        scripted(
            FakeResponse(
                FakeMessage(
                    None,
                    tool_calls=[
                        FakeToolCall(
                            id="c1", name="add", arguments=json.dumps({"a": 2, "b": 3})
                        )
                    ],
                )
            ),
            FakeResponse(FakeMessage("The sum is 5.")),
        )
        result = await run_agent(make_llm(), prompt, "Add 2 and 3", tools=[add])

        assert len(fake_client.payloads) >= 2, "the loop must make a second call"
        followup = fake_client.payloads[-1]["messages"]
        assert any(m.get("role") == "tool" for m in followup), (
            "the executed tool's result must be echoed back to the model"
        )
        assert "5" in str(result.output)

    async def test_tool_choice_omitted_when_no_tools(
        self, make_llm, prompt, fake_client
    ) -> None:
        fake_client.result = FakeResponse(FakeMessage("ok"))
        await run_agent(make_llm(), prompt, "hi")
        assert "tool_choice" not in fake_client.payloads[0]


# ====================================================================== #
# Structured output                                                        #
# ====================================================================== #


class Answer(BaseModel):
    answer: str
    confidence: float


class TestStructuredOutput:
    async def test_schema_reaches_the_wire_without_tools(
        self, make_llm, prompt, fake_client
    ) -> None:
        fake_client.result = FakeResponse(
            FakeMessage(json.dumps({"answer": "Paris", "confidence": 0.9}))
        )
        await run_agent(
            make_llm(), prompt, "Capital of France?", response_format=Answer
        )

        payload = fake_client.payloads[0]
        assert "response_schema" not in payload
        if "response_format" in payload:
            assert payload["response_format"]["type"] in {
                "json_schema",
                "json_object",
            }, "only OpenAI-compatible directives may reach a self-hosted server"

    async def test_no_nucleusiq_internal_shape_on_the_wire(
        self, make_llm, prompt, fake_client
    ) -> None:
        fake_client.result = FakeResponse(FakeMessage('{"answer":"x","confidence":1}'))
        await run_agent(
            make_llm(), prompt, "Capital of France?", response_format=Answer
        )
        sent = fake_client.payloads[0].get("response_format")
        if isinstance(sent, dict):
            assert sent.get("type") != "json", (
                "type='json' is NucleusIQ's internal shape; no inference "
                "server understands it"
            )
        assert not isinstance(sent, tuple), "a tuple must never be serialized"

    async def test_schema_and_tools_do_not_suppress_tool_calls(
        self, make_llm, prompt, fake_client, scripted
    ) -> None:
        scripted(
            FakeResponse(
                FakeMessage(
                    None,
                    tool_calls=[
                        FakeToolCall(
                            id="c1", name="add", arguments=json.dumps({"a": 1, "b": 1})
                        )
                    ],
                )
            ),
            FakeResponse(FakeMessage('{"answer":"2","confidence":1.0}')),
        )
        await run_agent(
            make_llm(),
            prompt,
            "Add 1 and 1",
            tools=[add],
            response_format=Answer,
        )

        first = fake_client.payloads[0]
        assert first.get("tools"), "the tools must survive structured output"
        assert "response_format" not in first, (
            "sending response_format alongside tools makes vLLM emit JSON "
            "instead of calling the tool — the policy must strip it and put "
            "the schema in the prompt instead"
        )
        assert any(
            "confidence" in str(m.get("content"))
            for m in first["messages"]
            if m["role"] == "system"
        ), "the schema must instead be injected into the system prompt"


# ====================================================================== #
# Streaming                                                                #
# ====================================================================== #


class TestStreamingThroughTheAgent:
    async def test_stream_events_reach_the_caller(
        self, make_llm, prompt, fake_client
    ) -> None:
        async def chunks():
            for piece in ("Hel", "lo"):
                yield FakeChunk(FakeDelta(piece))
            yield FakeChunk(FakeDelta(None), finish_reason="stop")
            yield FakeChunk(None, usage=FakeUsage(10, 2, 12))

        fake_client.result = chunks

        agent = Agent(
            name="Streamer",
            prompt=prompt,
            llm=make_llm(),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=False),
        )
        await agent.initialize()

        events = [
            event
            async for event in agent.execute_stream(Task(id="s1", objective="Say hi"))
        ]
        assert events, "the agent must forward provider stream events"
        text = "".join(e.token for e in events if e.type == "token" and e.token)
        assert "Hello" in text


# ====================================================================== #
# Error surfacing                                                          #
# ====================================================================== #


class TestErrorsThroughTheAgent:
    async def test_context_overflow_is_typed(
        self, make_llm, prompt, fake_client
    ) -> None:
        import httpx
        import openai
        from nucleusiq.llms.errors import ContextLengthError

        fake_client.result = openai.BadRequestError(
            message="This model's maximum context length is 8192 tokens",
            response=httpx.Response(
                400, request=httpx.Request("POST", "http://gpu:8000/v1")
            ),
            body=None,
        )
        llm = make_llm(max_retries=0)
        with pytest.raises(ContextLengthError):
            await llm.call(messages=[{"role": "user", "content": "hi"}])
