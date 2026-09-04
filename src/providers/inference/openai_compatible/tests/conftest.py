"""Shared fakes for the OpenAI-compatible provider test suite.

Every collaborator the provider talks to is injectable, so these fakes are
passed in rather than monkeypatched over the ``openai`` SDK.  That keeps the
tests meaningful: they exercise our code, not our ability to patch someone
else's.
"""

from __future__ import annotations

from typing import Any

import pytest

BASE_URL = "http://gpu-node-1:8000/v1"
MODEL = "gemma-4-27b-it"


# ====================================================================== #
# Response builders                                                        #
# ====================================================================== #


class FakeFunction:
    def __init__(self, name: str | None, arguments: str | None) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(
        self,
        *,
        index: int | None = 0,
        id: str | None = "call_1",
        name: str | None = "search",
        arguments: str | None = '{"q":"x"}',
    ) -> None:
        self.index = index
        self.id = id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(
        self,
        content: str | None = "hello",
        *,
        tool_calls: list[FakeToolCall] | None = None,
        reasoning: str | None = None,
        reasoning_field: str = "reasoning",
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        if reasoning is not None:
            setattr(self, reasoning_field, reasoning)


class FakeUsage:
    def __init__(
        self,
        prompt_tokens: int | None = 12,
        completion_tokens: int | None = 8,
        total_tokens: int | None = 20,
        reasoning_tokens: int | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.completion_tokens_details = (
            type("Details", (), {"reasoning_tokens": reasoning_tokens})()
            if reasoning_tokens is not None
            else None
        )


class FakeChoice:
    def __init__(
        self, message: FakeMessage, finish_reason: str | None = "stop"
    ) -> None:
        self.message = message
        self.finish_reason = finish_reason


#: Distinguishes "argument omitted" from "explicitly None", which matters
#: because servers such as llama.cpp legitimately return no ``usage`` block.
UNSET: Any = object()


class FakeResponse:
    def __init__(
        self,
        message: FakeMessage | None = None,
        *,
        usage: FakeUsage | None = UNSET,
        choices: list[FakeChoice] | None = None,
        id: str = "req-abc",
        model: str = MODEL,
    ) -> None:
        self.id = id
        self.model = model
        self.usage = FakeUsage() if usage is UNSET else usage
        if choices is not None:
            self.choices = choices
        else:
            self.choices = [FakeChoice(message or FakeMessage())]


class FakeDelta:
    def __init__(
        self,
        content: str | None = None,
        *,
        reasoning: str | None = None,
        tool_calls: list[FakeToolCall] | None = None,
    ) -> None:
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls


class FakeChunk:
    def __init__(
        self,
        delta: FakeDelta | None = None,
        *,
        finish_reason: str | None = None,
        usage: FakeUsage | None = None,
        id: str = "stream-1",
    ) -> None:
        self.id = id
        self.usage = usage
        self.choices = [FakeChoice.__new__(FakeChoice)] if delta is not None else []
        if delta is not None:
            choice = self.choices[0]
            choice.delta = delta  # type: ignore[attr-defined]
            choice.finish_reason = finish_reason
            choice.message = None  # type: ignore[attr-defined]


# ====================================================================== #
# Fake SDK client                                                          #
# ====================================================================== #


class FakeCompletions:
    """Records the payload it was called with and returns a scripted result."""

    def __init__(self, owner: FakeClient) -> None:
        self._owner = owner

    async def create(self, **payload: Any) -> Any:
        self._owner.payloads.append(payload)
        result = self._owner.result
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result()
        return result


class FakeModels:
    def __init__(self, owner: FakeClient) -> None:
        self._owner = owner

    async def list(self) -> Any:
        if self._owner.models_error is not None:
            raise self._owner.models_error
        return type("Listing", (), {"data": self._owner.model_cards})()


class FakeClient:
    """Stand-in for ``openai.AsyncOpenAI``.

    Tracks constructor options, every payload sent, and every
    ``with_options`` call so credential handling can be asserted.
    """

    def __init__(self, **options: Any) -> None:
        self.options: dict[str, Any] = options
        self.payloads: list[dict[str, Any]] = []
        self.with_options_calls: list[dict[str, Any]] = []
        self.result: Any = FakeResponse()
        self.model_cards: list[Any] = [{"id": MODEL, "max_model_len": 32_768}]
        self.models_error: Exception | None = None
        self.chat = type("Chat", (), {"completions": FakeCompletions(self)})()
        self.models = FakeModels(self)

    def with_options(self, **options: Any) -> FakeClient:
        self.with_options_calls.append(options)
        clone = FakeClient(**{**self.options, **options})
        # Share mutable recording state so assertions can use either handle.
        clone.payloads = self.payloads
        clone.with_options_calls = self.with_options_calls
        clone.result = self.result
        clone.model_cards = self.model_cards
        clone.models_error = self.models_error
        return clone


async def drain(stream: Any) -> list[Any]:
    """Collect an async generator into a list."""
    return [event async for event in stream]


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture(scope="session")
def shared_http_client():
    """One httpx client for the whole session.

    Building an ``openai.AsyncOpenAI`` otherwise creates a fresh SSL context
    each time, which costs ~0.4s per provider instance and dominates the
    suite's runtime.  The transport is never used — every test replaces
    ``_client`` with a fake.
    """
    import httpx

    client = httpx.AsyncClient()
    yield client


@pytest.fixture
def make_llm(fake_client: FakeClient, shared_http_client):
    """Build a provider wired to *fake_client*, bypassing real HTTP."""
    from nucleusiq_openai_compatible import OpenAICompatibleLLM
    from nucleusiq_openai_compatible._shared.model_probe import ModelProbe

    def _make(**overrides: Any) -> OpenAICompatibleLLM:
        kwargs: dict[str, Any] = {
            "base_url": BASE_URL,
            "model": MODEL,
            "engine": "vllm",
            "context_window": 32_768,
            "http_client": shared_http_client,
        }
        kwargs.update(overrides)
        llm = OpenAICompatibleLLM(**kwargs)
        llm._client = fake_client
        llm._probe = ModelProbe(lambda: fake_client)
        return llm

    return _make
