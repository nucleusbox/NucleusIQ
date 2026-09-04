"""Capability smoke tests against a **real** OpenAI-compatible server.

Everything else in this suite mocks the network: unit tests inject a fake
client, and ``test_over_http.py`` drives a loopback server that speaks the
protocol exactly as written. Neither can catch what actually bites in
production — a server that accepts ``tools`` but never emits ``tool_calls``, a
``max_model_len`` the probe cannot see, a reasoning model that buries thinking
in ``content``, a gateway that rewrites paths.

These tests are the answer to "is this provider working against *my*
endpoint?". They are skipped unless ``OPENAI_COMPATIBLE_BASE_URL`` is set and
deselected by default (``-m "not integration"``), so a normal run is untouched.

Run against any endpoint::

    # Ollama Cloud (no local GPU)
    export OPENAI_COMPATIBLE_BASE_URL="https://ollama.com/v1"
    export OPENAI_COMPATIBLE_API_KEY="$OLLAMA_API_KEY"
    export OPENAI_COMPATIBLE_MODEL="gemma4:31b"
    export OPENAI_COMPATIBLE_ENGINE="ollama"

    # local vLLM
    export OPENAI_COMPATIBLE_BASE_URL="http://gpu-node-1:8000/v1"
    export OPENAI_COMPATIBLE_MODEL="gemma-4-27b-it"
    export OPENAI_COMPATIBLE_ENGINE="vllm"

    pytest tests/integration/test_live_endpoint.py -m integration --no-cov -v

``--no-cov`` matters: the package-wide gate is 95%, and this file exercises
only the request path, so leaving coverage on would fail the run for the wrong
reason.

Token budgets are deliberately tiny — this is a wiring check, not an eval, and
it may be billing someone per token.

Set ``OPENAI_COMPATIBLE_REASONING_MODEL`` to additionally verify that a
thinking model's reasoning arrives separated from its answer.
"""

from __future__ import annotations

import os

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.errors import AuthenticationError, ModelNotFoundError
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.streaming.events import StreamEventType
from nucleusiq.tools import tool
from nucleusiq_openai_compatible import OpenAICompatibleLLM
from pydantic import BaseModel

BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "")
API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY")
ENGINE = os.getenv("OPENAI_COMPATIBLE_ENGINE", "generic")
REASONING_MODEL = os.getenv("OPENAI_COMPATIBLE_REASONING_MODEL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not BASE_URL,
        reason="set OPENAI_COMPATIBLE_BASE_URL to run live endpoint checks",
    ),
]

ASK = [{"role": "user", "content": "Reply with exactly: ok"}]


def build_llm(**overrides) -> OpenAICompatibleLLM:
    kwargs = {
        "base_url": BASE_URL,
        "model": MODEL,
        "engine": ENGINE,
        "api_key": API_KEY,
        "timeout": 120.0,
    }
    kwargs.update(overrides)
    return OpenAICompatibleLLM(**kwargs)


@pytest.fixture(scope="module")
def llm() -> OpenAICompatibleLLM:
    return build_llm()


# ====================================================================== #
# Preflight — run these first; everything else assumes they pass          #
# ====================================================================== #


class TestPreflight:
    async def test_endpoint_validates(self, llm: OpenAICompatibleLLM) -> None:
        report = await llm.validate()
        assert report.ok, f"\n{report.render()}"

    async def test_model_is_served(self, llm: OpenAICompatibleLLM) -> None:
        report = await llm.validate()
        assert report.model_found, (
            f"{MODEL!r} not served; endpoint offers: "
            f"{', '.join(report.served_models) or '(none reported)'}"
        )

    async def test_context_window_is_plausible(self, llm: OpenAICompatibleLLM) -> None:
        window = llm.get_context_window()
        assert window >= 2_048, f"implausible context window: {window}"

    async def test_unknown_model_names_the_alternatives(self) -> None:
        """The failure mode this provider exists to make legible."""
        wrong = build_llm(model="definitely-not-served-xyz", validate_model=True)
        with pytest.raises(ModelNotFoundError) as exc:
            await wrong.call(messages=ASK, max_output_tokens=8)
        assert "definitely-not-served-xyz" in str(exc.value)


# ====================================================================== #
# Direct provider use                                                     #
# ====================================================================== #


class TestCompletion:
    async def test_returns_content(self, llm: OpenAICompatibleLLM) -> None:
        response = await llm.call(messages=ASK, max_output_tokens=16)
        assert response.content and response.content.strip(), (
            "server returned empty content"
        )

    async def test_framework_shaped_view_agrees(self, llm: OpenAICompatibleLLM) -> None:
        """The agent modes read ``choices[0].message``, not the flat fields."""
        response = await llm.call(messages=ASK, max_output_tokens=16)
        assert response.choices[0].message.content == response.content

    async def test_reports_usage(self, llm: OpenAICompatibleLLM) -> None:
        response = await llm.call(messages=ASK, max_output_tokens=16)
        assert response.usage_reported, (
            "server reported no usage; context budgeting would be blind"
        )
        assert (response.total_tokens or 0) > 0

    async def test_streams_incrementally(self, llm: OpenAICompatibleLLM) -> None:
        tokens, completed = [], False
        async for event in llm.call_stream(
            messages=[{"role": "user", "content": "Count: 1 2 3"}],
            max_output_tokens=32,
        ):
            if event.type == StreamEventType.TOKEN:
                tokens.append(event.token)
            elif event.type == StreamEventType.COMPLETE:
                completed = True
        assert completed, "stream never emitted COMPLETE"
        assert tokens, "stream produced no token deltas (server may not chunk)"


# ====================================================================== #
# Tools                                                                   #
# ====================================================================== #


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current temperature for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


class TestTools:
    async def test_server_emits_tool_calls(self, llm: OpenAICompatibleLLM) -> None:
        if not llm.capabilities.supports_tools:
            pytest.skip(f"engine {ENGINE!r} declares no tool support")

        response = await llm.call(
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
            ],
            tools=[WEATHER_TOOL],
            max_output_tokens=128,
        )
        assert response.tool_calls, (
            "server accepted tools but returned no tool_calls. For vLLM, start "
            "it with --enable-auto-tool-choice --tool-call-parser <parser>."
        )
        assert response.tool_calls[0].name == "get_weather"

    async def test_tool_calls_reach_the_framework(
        self, llm: OpenAICompatibleLLM
    ) -> None:
        """``_get_tool_calls`` gates on a real list, so verify that view too."""
        if not llm.capabilities.supports_tools:
            pytest.skip(f"engine {ENGINE!r} declares no tool support")

        response = await llm.call(
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[WEATHER_TOOL],
            max_output_tokens=128,
        )
        calls = response.choices[0].message.tool_calls
        assert isinstance(calls, list) and calls
        assert calls[0].function.name == "get_weather"


# ====================================================================== #
# Structured output                                                       #
# ====================================================================== #


class City(BaseModel):
    name: str
    country: str


class TestStructuredOutput:
    async def test_schema_is_enforced(self, llm: OpenAICompatibleLLM) -> None:
        """Check the engine preset's json-schema claim against reality.

        A server that *silently ignores* ``response_format`` is the dangerous
        case: nothing errors, so the agent believes NATIVE mode succeeded and
        returns prose or fenced markdown where a validated object was
        promised. Failing here means the preset overstates the engine, and the
        remedy is ``supports_json_schema=False`` so AUTO falls back to PROMPT.
        """
        if not llm.supports_native_structured_output:
            pytest.skip(f"engine {ENGINE!r} declares no JSON-schema support")

        import json

        response = await llm.call(
            messages=[{"role": "user", "content": "Paris, France as JSON."}],
            response_schema=City.model_json_schema(),
            max_output_tokens=200,
        )
        content = response.content or ""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            pytest.fail(
                f"engine {ENGINE!r} declares supports_json_schema=True, but "
                f"{MODEL!r} returned content that is not bare JSON, so the "
                "schema was ignored rather than enforced. Pass "
                "supports_json_schema=False for this deployment.\n"
                f"content: {content[:200]!r}"
            )
        City.model_validate(payload)


# ====================================================================== #
# Through a real Agent                                                    #
# ====================================================================== #


@tool("add")
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


class TestAgent:
    async def test_prompt_pipeline(self, llm: OpenAICompatibleLLM) -> None:
        agent = Agent(
            name="LiveSmoke",
            llm=llm,
            prompt=PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
                system="Answer in one short sentence.",
                user="Be concise.",
            ),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )
        await agent.initialize()
        result = await agent.execute(
            Task(id="t1", objective="Name the capital of France.")
        )
        assert result.status == "success", getattr(result, "error", result)
        assert "paris" in str(result.output).lower()

    async def test_tool_loop(self, llm: OpenAICompatibleLLM) -> None:
        if not llm.capabilities.supports_tools:
            pytest.skip(f"engine {ENGINE!r} declares no tool support")

        agent = Agent(
            name="LiveToolSmoke",
            llm=llm,
            tools=[add],
            prompt=PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
                system="Use the add tool for arithmetic.",
                user="Always call the tool rather than computing yourself.",
            ),
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t2", objective="What is 21 plus 21?"))
        assert result.status == "success", getattr(result, "error", result)
        assert "42" in str(result.output)


# ====================================================================== #
# Reasoning / thinking models                                             #
# ====================================================================== #


class TestReasoning:
    async def test_thinking_is_separated_from_the_answer(self) -> None:
        if not REASONING_MODEL:
            pytest.skip("set OPENAI_COMPATIBLE_REASONING_MODEL to check thinking")

        llm = build_llm(model=REASONING_MODEL, is_reasoning_model=True)
        response = await llm.call(
            messages=[
                {"role": "user", "content": "If 2x = 10, what is x? Think first."}
            ],
            max_output_tokens=512,
        )
        assert response.content, "no visible answer"
        if not response.has_reasoning:
            pytest.skip(
                "server returned no separate reasoning field; thinking, if any, "
                "stays inline in content"
            )
        assert "<think>" not in response.content, (
            "thinking leaked into content despite a separate reasoning field"
        )
        assert response.reasoning == response.choices[0].message.reasoning


# ====================================================================== #
# Credentials                                                             #
# ====================================================================== #


class TestCredentials:
    async def test_bad_key_is_typed(self) -> None:
        if not API_KEY:
            pytest.skip("endpoint is unauthenticated; nothing to reject")

        bad = build_llm(api_key="sk-definitely-invalid-key")
        with pytest.raises(AuthenticationError):
            await bad.call(messages=ASK, max_output_tokens=8)

    async def test_key_absent_from_repr(self, llm: OpenAICompatibleLLM) -> None:
        if not API_KEY:
            pytest.skip("endpoint is unauthenticated")
        assert API_KEY not in repr(llm)
