"""``BaseAnthropic`` passes ``output_config`` for structured schemas."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel


class _Tiny(BaseModel):
    x: int


@pytest.mark.asyncio
async def test_call_injects_output_config_json_schema(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    from types import SimpleNamespace

    from nucleusiq_anthropic.nb_anthropic.base import BaseAnthropic

    llm = BaseAnthropic(async_mode=True)
    captured: dict = {}

    async def _spy(**kw):
        captured.update(kw)

        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"x":7}')],
            usage=SimpleNamespace(
                input_tokens=0,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="m",
            id="i",
        )

    llm._client.messages.create = _spy

    out = await llm.call(
        model="m",
        messages=[{"role": "user", "content": "give json"}],
        response_format=_Tiny,
    )

    assert isinstance(out, _Tiny)
    assert out.x == 7

    oc = captured.get("output_config")
    assert isinstance(oc, dict)
    assert oc["format"]["type"] == "json_schema"
    assert "schema" in oc["format"]


@pytest.mark.asyncio
async def test_call_drops_structured_when_tools_present(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    from types import SimpleNamespace

    from nucleusiq_anthropic.nb_anthropic.base import BaseAnthropic

    llm = BaseAnthropic(async_mode=True)
    llm._client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="{}")],
            usage=SimpleNamespace(
                input_tokens=0,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="m",
            id="i",
        ),
    )

    await llm.call(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "n", "input_schema": {"type": "object", "properties": {}}}],
        response_format=_Tiny,
    )

    kw = llm._client.messages.create.await_args.kwargs
    assert "output_config" not in kw
