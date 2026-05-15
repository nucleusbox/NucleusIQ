"""Streaming adapter unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from nucleusiq_anthropic.nb_anthropic.stream_adapter import _process_raw_events


@pytest.mark.asyncio
async def test_process_raw_events_token_and_tool() -> None:

    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(id="mid", model="claude"),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="Hi"),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(type="tool_use", id="t1", name="fn"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"a":'),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json="1}"),
        ),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(
                input_tokens=2,
                output_tokens=3,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        ),
    ]

    async def gen():

        for e in events:
            yield e

    out = []

    async for ev in _process_raw_events(gen()):
        out.append(ev)

    assert any(e.type == "token" for e in out)

    complete = out[-1]

    assert complete.type == "complete"

    assert complete.content == "Hi"

    assert complete.metadata["tool_calls"][0]["name"] == "fn"
