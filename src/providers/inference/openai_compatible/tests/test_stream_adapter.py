"""Streaming: fragment reassembly, thinking deltas and usage trailers."""

from __future__ import annotations

import pytest
from conftest import FakeChunk, FakeDelta, FakeToolCall, FakeUsage, drain
from nucleusiq_openai_compatible.nb_compat.stream_adapter import (
    StreamOutcome,
    ToolCallAccumulator,
    stream_events,
)


async def astream(items):
    for item in items:
        yield item


async def run(chunks) -> tuple[list, StreamOutcome]:
    outcome = StreamOutcome()
    events = await drain(stream_events(astream(chunks), outcome=outcome, model="m"))
    return events, outcome


def tokens(events) -> list[str]:
    return [e.token for e in events if e.type == "token"]


def content_tokens(events) -> list[str]:
    return [
        e.token
        for e in events
        if e.type == "token" and not (e.metadata or {}).get("reasoning")
    ]


def reasoning_tokens(events) -> list[str]:
    return [
        e.token
        for e in events
        if e.type == "token" and (e.metadata or {}).get("reasoning")
    ]


class TestAccumulator:
    def test_merges_argument_fragments(self) -> None:
        acc = ToolCallAccumulator()
        acc.add([FakeToolCall(index=0, id="c1", name="search", arguments='{"q":')])
        acc.add([FakeToolCall(index=0, id=None, name=None, arguments='"rain"}')])
        (call,) = acc.finish()
        assert call.arguments == '{"q":"rain"}'
        assert call.name == "search"
        assert call.id == "c1"

    def test_parallel_calls_kept_apart(self) -> None:
        acc = ToolCallAccumulator()
        acc.add(
            [
                FakeToolCall(index=0, id="c1", name="search", arguments="{"),
                FakeToolCall(index=1, id="c2", name="calc", arguments="["),
            ]
        )
        acc.add(
            [
                FakeToolCall(index=0, id=None, name=None, arguments="}"),
                FakeToolCall(index=1, id=None, name=None, arguments="]"),
            ]
        )
        calls = acc.finish()
        assert [(c.name, c.arguments) for c in calls] == [
            ("search", "{}"),
            ("calc", "[]"),
        ]

    def test_ordered_by_index_not_arrival(self) -> None:
        acc = ToolCallAccumulator()
        acc.add([FakeToolCall(index=1, id="c2", name="second")])
        acc.add([FakeToolCall(index=0, id="c1", name="first")])
        assert [c.name for c in acc.finish()] == ["first", "second"]

    def test_position_used_when_index_missing(self) -> None:
        acc = ToolCallAccumulator()
        acc.add(
            [
                FakeToolCall(index=None, id="c1", name="a"),
                FakeToolCall(index=None, id="c2", name="b"),
            ]
        )
        assert [c.name for c in acc.finish()] == ["a", "b"]

    def test_nameless_fragment_discarded(self) -> None:
        acc = ToolCallAccumulator()
        acc.add([FakeToolCall(index=0, id="c1", name=None, arguments="{}")])
        assert acc.finish() == ()

    def test_synthesizes_missing_id(self) -> None:
        acc = ToolCallAccumulator()
        acc.add([FakeToolCall(index=3, id=None, name="search")])
        assert acc.finish()[0].id == "call_3"

    def test_empty(self) -> None:
        acc = ToolCallAccumulator()
        assert not acc
        assert acc.finish() == ()

    def test_truthy_once_populated(self) -> None:
        acc = ToolCallAccumulator()
        acc.add([FakeToolCall(index=0, name="s")])
        assert acc

    @pytest.mark.parametrize("value", [None, ()])
    def test_no_op_on_empty_delta(self, value) -> None:
        acc = ToolCallAccumulator()
        acc.add(value)
        assert not acc


class TestContentStream:
    async def test_tokens_then_complete(self) -> None:
        events, outcome = await run(
            [
                FakeChunk(FakeDelta("Hel")),
                FakeChunk(FakeDelta("lo")),
                FakeChunk(FakeDelta(None), finish_reason="stop"),
            ]
        )
        assert tokens(events) == ["Hel", "lo"]
        assert events[-1].type == "complete"
        assert events[-1].content == "Hello"
        assert outcome.text == "Hello"

    async def test_empty_stream_still_completes(self) -> None:
        events, outcome = await run([])
        assert len(events) == 1
        assert events[0].type == "complete"
        assert outcome.response.content is None

    async def test_empty_string_deltas_emit_nothing(self) -> None:
        events, _ = await run([FakeChunk(FakeDelta(""))])
        assert tokens(events) == []

    async def test_finish_reason_recorded(self) -> None:
        _, outcome = await run([FakeChunk(FakeDelta("x"), finish_reason="length")])
        assert outcome.response.finish_reason == "length"

    async def test_request_id_from_first_chunk(self) -> None:
        _, outcome = await run(
            [FakeChunk(FakeDelta("a"), id="s1"), FakeChunk(FakeDelta("b"), id="s2")]
        )
        assert outcome.response.request_id == "s1"

    async def test_model_recorded(self) -> None:
        _, outcome = await run([FakeChunk(FakeDelta("a"))])
        assert outcome.response.model == "m"


class TestUsage:
    async def test_trailer_chunk_is_not_the_end_of_content(self) -> None:
        events, outcome = await run(
            [
                FakeChunk(FakeDelta("a")),
                FakeChunk(None, usage=FakeUsage(10, 2, 12)),
                FakeChunk(FakeDelta("b")),
            ]
        )
        assert tokens(events) == ["a", "b"], (
            "an empty choices list means a usage trailer, not end of stream"
        )
        assert outcome.response.prompt_tokens == 10

    async def test_usage_flag(self) -> None:
        _, outcome = await run([FakeChunk(None, usage=FakeUsage(5, 1, 6))])
        assert outcome.response.usage_reported

    async def test_absent_usage(self) -> None:
        _, outcome = await run([FakeChunk(FakeDelta("a"))])
        assert not outcome.response.usage_reported

    async def test_total_derived(self) -> None:
        _, outcome = await run([FakeChunk(None, usage=FakeUsage(10, 5, None))])
        assert outcome.response.total_tokens == 15


class TestReasoningStream:
    async def test_thinking_tagged_separately(self) -> None:
        events, outcome = await run(
            [
                FakeChunk(FakeDelta(None, reasoning="let me ")),
                FakeChunk(FakeDelta(None, reasoning="think")),
                FakeChunk(FakeDelta("42"), finish_reason="stop"),
            ]
        )
        assert reasoning_tokens(events) == ["let me ", "think"]
        assert content_tokens(events) == ["42"]
        assert outcome.reasoning == "let me think"

    async def test_complete_event_excludes_thinking(self) -> None:
        events, _ = await run(
            [
                FakeChunk(FakeDelta(None, reasoning="private thoughts")),
                FakeChunk(FakeDelta("the answer")),
            ]
        )
        assert events[-1].content == "the answer", (
            "thinking must not land in the answer text a caller renders or "
            "appends to history"
        )

    async def test_reasoning_on_the_outcome(self) -> None:
        _, outcome = await run([FakeChunk(FakeDelta("a", reasoning="why"))])
        assert outcome.response.reasoning == "why"
        assert outcome.response.has_reasoning

    async def test_no_reasoning_leaves_it_none(self) -> None:
        _, outcome = await run([FakeChunk(FakeDelta("a"))])
        assert outcome.response.reasoning is None
        assert outcome.reasoning == ""

    async def test_reasoning_and_content_in_one_delta(self) -> None:
        events, _ = await run([FakeChunk(FakeDelta("ans", reasoning="think"))])
        assert reasoning_tokens(events) == ["think"]
        assert content_tokens(events) == ["ans"]

    async def test_reasoning_only_stream_detected(self) -> None:
        _, outcome = await run([FakeChunk(FakeDelta(None, reasoning="all of it"))])
        assert outcome.response.reasoning_only


class TestToolCallStream:
    async def test_reassembled_onto_the_outcome(self) -> None:
        _, outcome = await run(
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
        (call,) = outcome.response.tool_calls
        assert call.arguments == '{"q":"x"}'
        assert outcome.response.finish_reason == "tool_calls"

    async def test_tool_calls_do_not_emit_tokens(self) -> None:
        events, _ = await run(
            [FakeChunk(FakeDelta(None, tool_calls=[FakeToolCall(index=0, name="s")]))]
        )
        assert tokens(events) == []

    async def test_content_and_tool_calls_together(self) -> None:
        events, outcome = await run(
            [
                FakeChunk(FakeDelta("Let me look. ")),
                FakeChunk(
                    FakeDelta(
                        None,
                        tool_calls=[
                            FakeToolCall(
                                index=0, id="c1", name="search", arguments="{}"
                            )
                        ],
                    )
                ),
            ]
        )
        assert outcome.text == "Let me look. "
        assert outcome.response.has_tool_calls


def complete(events):
    return next(e for e in events if e.type == "complete")


class TestCompleteEventMetadata:
    """The COMPLETE event is the streaming path's only channel to the agent.

    ``base_mode`` streaming reads ``metadata["tool_calls"]`` to decide whether
    to run the tool loop, and ``UsageTracker`` reads ``metadata["usage"]``.
    Neither errors when the key is missing — the loop just never fires and the
    token count stays zero — so these assertions guard a silent failure.
    """

    TOOL_CHUNKS = [
        FakeChunk(
            FakeDelta(
                None,
                tool_calls=[
                    FakeToolCall(index=0, id="c1", name="search", arguments='{"q":"x"}')
                ],
            ),
            finish_reason="tool_calls",
        )
    ]

    async def test_tool_calls_are_published_to_the_agent(self) -> None:
        events, _ = await run(self.TOOL_CHUNKS)
        assert complete(events).metadata["tool_calls"] == [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"x"}'},
            }
        ]

    async def test_published_calls_survive_the_framework_parser(self) -> None:
        from nucleusiq.agents.chat_models import ToolCallRequest

        events, _ = await run(self.TOOL_CHUNKS)
        raw = complete(events).metadata["tool_calls"]
        parsed = [ToolCallRequest.from_raw(tc) for tc in raw]
        assert [(c.id, c.name, c.arguments) for c in parsed] == [
            ("c1", "search", '{"q":"x"}')
        ]

    async def test_usage_is_published_when_reported(self) -> None:
        events, _ = await run(
            [
                FakeChunk(FakeDelta("hi")),
                FakeChunk(FakeDelta(None), usage=FakeUsage(7, 3, 10)),
            ]
        )
        assert complete(events).metadata["usage"] == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
            "reasoning_tokens": 0,
        }

    async def test_no_usage_key_when_server_reported_none(self) -> None:
        """Zeros would be indistinguishable from a genuinely free call."""
        events, _ = await run([FakeChunk(FakeDelta("hi"))])
        assert "usage" not in (complete(events).metadata or {})

    async def test_plain_text_stream_advertises_no_tool_loop(self) -> None:
        """An absent ``tool_calls`` key is what stops the agent looping."""
        events, _ = await run([FakeChunk(FakeDelta("hi"))])
        assert "tool_calls" not in (complete(events).metadata or {})

    async def test_finish_reason_and_model_are_published(self) -> None:
        events, _ = await run(self.TOOL_CHUNKS)
        metadata = complete(events).metadata
        assert metadata["finish_reason"] == "tool_calls"
        assert metadata["model"] == "m"

    async def test_reasoning_is_published_separately(self) -> None:
        events, _ = await run(
            [
                FakeChunk(FakeDelta(None, reasoning="thinking")),
                FakeChunk(FakeDelta("answer")),
            ]
        )
        assert complete(events).metadata["reasoning_content"] == "thinking"


class TestDictShapedChunks:
    """Some gateways hand back plain dicts rather than SDK models."""

    async def test_dict_chunks_supported(self) -> None:
        chunks = [
            {
                "id": "s1",
                "choices": [{"delta": {"content": "hi"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "s1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            },
        ]
        events, outcome = await run(chunks)
        assert tokens(events) == ["hi"]
        assert outcome.response.total_tokens == 4
        assert outcome.response.finish_reason == "stop"

    async def test_dict_reasoning(self) -> None:
        chunks = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]}]
        _, outcome = await run(chunks)
        assert outcome.response.reasoning == "hmm"
