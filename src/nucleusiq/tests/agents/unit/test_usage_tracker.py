"""Tests for UsageTracker, UsageRecord, CallPurpose, TokenOrigin, and mode-level wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq.tests.conftest import make_test_prompt
from nucleusiq.agents.usage.usage_tracker import (
    PURPOSE_ORIGIN_MAP,
    BucketStats,
    CallPurpose,
    TokenCount,
    TokenOrigin,
    UsageRecord,
    UsageSummary,
    UsageTracker,
)

# ================================================================== #
# CallPurpose enum                                                     #
# ================================================================== #


class TestCallPurpose:
    def test_values(self):
        assert CallPurpose.MAIN == "main"
        assert CallPurpose.PLANNING == "planning"
        assert CallPurpose.TOOL_LOOP == "tool_loop"
        assert CallPurpose.CRITIC == "critic"
        assert CallPurpose.REFINER == "refiner"

    def test_all_values_are_strings(self):
        for member in CallPurpose:
            assert isinstance(member.value, str)


# ================================================================== #
# TokenOrigin enum                                                     #
# ================================================================== #


class TestTokenOrigin:
    def test_values(self):
        assert TokenOrigin.USER == "user"
        assert TokenOrigin.FRAMEWORK == "framework"

    def test_purpose_origin_map(self):
        assert PURPOSE_ORIGIN_MAP[CallPurpose.MAIN] == TokenOrigin.USER
        assert PURPOSE_ORIGIN_MAP[CallPurpose.PLANNING] == TokenOrigin.FRAMEWORK
        assert PURPOSE_ORIGIN_MAP[CallPurpose.TOOL_LOOP] == TokenOrigin.FRAMEWORK
        assert PURPOSE_ORIGIN_MAP[CallPurpose.CRITIC] == TokenOrigin.FRAMEWORK
        assert PURPOSE_ORIGIN_MAP[CallPurpose.REFINER] == TokenOrigin.FRAMEWORK

    def test_all_purposes_mapped(self):
        for purpose in CallPurpose:
            assert purpose in PURPOSE_ORIGIN_MAP


# ================================================================== #
# Pydantic summary models                                              #
# ================================================================== #


class TestTokenCount:
    def test_defaults(self):
        tc = TokenCount()
        assert tc.prompt_tokens == 0
        assert tc.completion_tokens == 0
        assert tc.total_tokens == 0
        assert tc.reasoning_tokens == 0

    def test_model_dump(self):
        tc = TokenCount(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        d = tc.model_dump()
        assert d == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "reasoning_tokens": 0,
        }


class TestBucketStats:
    def test_defaults(self):
        bs = BucketStats()
        assert bs.calls == 0
        assert bs.total_tokens == 0

    def test_inherits_token_count(self):
        bs = BucketStats(prompt_tokens=10, calls=3)
        assert bs.prompt_tokens == 10
        assert bs.calls == 3

    def test_model_dump(self):
        bs = BucketStats(
            prompt_tokens=50, completion_tokens=20, total_tokens=70, calls=2
        )
        d = bs.model_dump()
        assert d["calls"] == 2
        assert d["total_tokens"] == 70


class TestUsageSummaryModel:
    def test_defaults(self):
        s = UsageSummary()
        assert s.call_count == 0
        assert s.total.total_tokens == 0
        assert s.by_purpose == {}
        assert s.by_origin == {}

    def test_model_dump_roundtrip(self):
        s = UsageSummary(
            total=TokenCount(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            call_count=2,
            by_purpose={"main": BucketStats(total_tokens=150, calls=2)},
            by_origin={"user": BucketStats(total_tokens=150, calls=2)},
        )
        d = s.model_dump()
        assert d["total"]["prompt_tokens"] == 100
        assert d["by_purpose"]["main"]["calls"] == 2
        assert d["by_origin"]["user"]["total_tokens"] == 150
        restored = UsageSummary.model_validate(d)
        assert restored.call_count == 2

    def test_summary_returns_dict(self):
        s = UsageSummary(
            total=TokenCount(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            call_count=2,
            by_purpose={"main": BucketStats(total_tokens=75, calls=1)},
            by_origin={"user": BucketStats(total_tokens=75, calls=1)},
        )
        d = s.summary()
        assert isinstance(d, dict)
        assert d["total"]["prompt_tokens"] == 100
        assert d["call_count"] == 2
        assert d["by_purpose"]["main"]["total_tokens"] == 75
        assert d["by_origin"]["user"]["calls"] == 1

    def test_summary_equals_model_dump(self):
        s = UsageSummary(
            total=TokenCount(prompt_tokens=50, total_tokens=50),
            call_count=1,
        )
        assert s.summary() == s.model_dump()

    def test_display_returns_string(self):
        s = UsageSummary(
            total=TokenCount(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            call_count=2,
            by_purpose={
                "main": BucketStats(total_tokens=75, calls=1),
                "tool_loop": BucketStats(total_tokens=75, calls=1),
            },
            by_origin={
                "user": BucketStats(total_tokens=75, calls=1),
                "framework": BucketStats(total_tokens=75, calls=1),
            },
        )
        text = s.display()
        assert isinstance(text, str)
        assert "Usage Summary (2 LLM calls)" in text
        assert "150 tokens" in text
        assert "main:" in text
        assert "tool_loop:" in text
        assert "user:" in text
        assert "framework:" in text
        assert "50%" in text

    def test_display_empty_tracker(self):
        s = UsageSummary()
        text = s.display()
        assert "0 LLM calls" in text
        assert "0 tokens" in text

    def test_display_single_call(self):
        s = UsageSummary(
            total=TokenCount(prompt_tokens=100, total_tokens=100),
            call_count=1,
            by_purpose={"main": BucketStats(total_tokens=100, calls=1)},
            by_origin={"user": BucketStats(total_tokens=100, calls=1)},
        )
        text = s.display()
        assert "1 LLM call)" in text
        assert "1 call)" in text
        assert "100%" in text


# ================================================================== #
# UsageRecord model                                                    #
# ================================================================== #


class TestUsageRecord:
    def test_defaults(self):
        rec = UsageRecord(purpose=CallPurpose.MAIN)
        assert rec.prompt_tokens == 0
        assert rec.completion_tokens == 0
        assert rec.total_tokens == 0
        assert rec.reasoning_tokens == 0
        assert rec.call_round == 1
        assert rec.origin == TokenOrigin.FRAMEWORK

    def test_with_values(self):
        rec = UsageRecord(
            purpose=CallPurpose.TOOL_LOOP,
            origin=TokenOrigin.FRAMEWORK,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=10,
            call_round=3,
        )
        assert rec.purpose == CallPurpose.TOOL_LOOP
        assert rec.origin == TokenOrigin.FRAMEWORK
        assert rec.prompt_tokens == 100
        assert rec.total_tokens == 150
        assert rec.call_round == 3

    def test_origin_user(self):
        rec = UsageRecord(purpose=CallPurpose.MAIN, origin=TokenOrigin.USER)
        assert rec.origin == TokenOrigin.USER


# ================================================================== #
# UsageTracker — record() and summary                                  #
# ================================================================== #


class TestUsageTracker:
    def test_empty_tracker(self):
        tracker = UsageTracker()
        assert tracker.call_count == 0
        assert tracker.total_tokens == 0
        summary = tracker.summary
        assert isinstance(summary, UsageSummary)
        assert summary.total.prompt_tokens == 0
        assert summary.total.completion_tokens == 0
        assert summary.total.total_tokens == 0
        assert summary.by_purpose == {}
        assert summary.by_origin == {}
        assert summary.call_count == 0

    def test_record_single_call(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.MAIN,
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        assert tracker.call_count == 1
        assert tracker.total_tokens == 150
        summary = tracker.summary
        assert summary.total.prompt_tokens == 100
        assert summary.total.completion_tokens == 50
        assert summary.by_purpose["main"].calls == 1

    def test_record_multiple_calls(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.MAIN,
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        tracker.record(
            CallPurpose.TOOL_LOOP,
            {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
        )
        tracker.record(
            CallPurpose.TOOL_LOOP,
            {"prompt_tokens": 250, "completion_tokens": 60, "total_tokens": 310},
        )

        assert tracker.call_count == 3
        assert tracker.total_tokens == 150 + 280 + 310

        summary = tracker.summary
        assert summary.by_purpose["main"].calls == 1
        assert summary.by_purpose["tool_loop"].calls == 2
        assert summary.by_purpose["tool_loop"].prompt_tokens == 450
        assert summary.by_purpose["tool_loop"].completion_tokens == 140

    def test_record_all_purposes(self):
        tracker = UsageTracker()
        for purpose in CallPurpose:
            tracker.record(
                purpose,
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        assert tracker.call_count == len(CallPurpose)
        summary = tracker.summary
        assert len(summary.by_purpose) == len(CallPurpose)
        for purpose in CallPurpose:
            assert summary.by_purpose[purpose.value].calls == 1

    def test_record_none_usage_is_noop(self):
        tracker = UsageTracker()
        tracker.record(CallPurpose.MAIN, None)
        assert tracker.call_count == 0

    def test_record_missing_keys(self):
        tracker = UsageTracker()
        tracker.record(CallPurpose.MAIN, {"prompt_tokens": 50})
        assert tracker.call_count == 1
        rec = tracker.records[0]
        assert rec.prompt_tokens == 50
        assert rec.completion_tokens == 0
        assert rec.total_tokens == 50  # auto-calculated

    def test_record_invalid_values(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.MAIN,
            {"prompt_tokens": "not_a_number", "completion_tokens": None},
        )
        rec = tracker.records[0]
        assert rec.prompt_tokens == 0
        assert rec.completion_tokens == 0

    def test_total_tokens_auto_calculated(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.MAIN,
            {"prompt_tokens": 100, "completion_tokens": 50},
        )
        rec = tracker.records[0]
        assert rec.total_tokens == 150

    def test_total_tokens_explicit(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.MAIN,
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 200},
        )
        rec = tracker.records[0]
        assert rec.total_tokens == 200

    def test_reasoning_tokens(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.MAIN,
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "reasoning_tokens": 30,
            },
        )
        summary = tracker.summary
        assert summary.total.reasoning_tokens == 30
        assert summary.by_purpose["main"].reasoning_tokens == 30

    def test_call_round(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.TOOL_LOOP,
            {"prompt_tokens": 10, "total_tokens": 10},
            call_round=3,
        )
        rec = tracker.records[0]
        assert rec.call_round == 3

    def test_reset(self):
        tracker = UsageTracker()
        tracker.record(CallPurpose.MAIN, {"prompt_tokens": 100, "total_tokens": 100})
        tracker.record(
            CallPurpose.TOOL_LOOP, {"prompt_tokens": 200, "total_tokens": 200}
        )
        assert tracker.call_count == 2

        tracker.reset()
        assert tracker.call_count == 0
        assert tracker.total_tokens == 0
        assert tracker.summary.by_purpose == {}

    def test_records_returns_copy(self):
        tracker = UsageTracker()
        tracker.record(CallPurpose.MAIN, {"prompt_tokens": 10, "total_tokens": 10})
        records = tracker.records
        records.clear()
        assert tracker.call_count == 1

    def test_summary_model_dump(self):
        """summary.model_dump() returns a plain dict for serialization."""
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.MAIN,
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        d = tracker.summary.model_dump()
        assert isinstance(d, dict)
        assert d["total"]["prompt_tokens"] == 100
        assert d["by_purpose"]["main"]["calls"] == 1
        assert d["call_count"] == 1


# ================================================================== #
# TokenOrigin — by_origin summary split                                #
# ================================================================== #


class TestByOriginSummary:
    def test_main_call_tagged_as_user(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.MAIN,
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        summary = tracker.summary
        assert "user" in summary.by_origin
        assert summary.by_origin["user"].total_tokens == 150

    def test_tool_loop_tagged_as_framework(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.TOOL_LOOP,
            {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
        )
        summary = tracker.summary
        assert "framework" in summary.by_origin
        assert summary.by_origin["framework"].total_tokens == 280

    def test_mixed_calls_split_correctly(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.MAIN,
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        tracker.record(
            CallPurpose.TOOL_LOOP,
            {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
        )
        tracker.record(
            CallPurpose.CRITIC,
            {"prompt_tokens": 60, "completion_tokens": 30, "total_tokens": 90},
        )

        summary = tracker.summary
        assert summary.by_origin["user"].total_tokens == 150
        assert summary.by_origin["user"].calls == 1
        assert summary.by_origin["framework"].total_tokens == 280 + 90
        assert summary.by_origin["framework"].calls == 2

    def test_explicit_origin_override(self):
        tracker = UsageTracker()
        tracker.record(
            CallPurpose.TOOL_LOOP,
            {"prompt_tokens": 50, "total_tokens": 50},
            origin=TokenOrigin.USER,
        )
        summary = tracker.summary
        assert summary.by_origin["user"].total_tokens == 50
        assert "framework" not in summary.by_origin

    def test_all_purposes_auto_origin(self):
        tracker = UsageTracker()
        for purpose in CallPurpose:
            tracker.record(
                purpose,
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        summary = tracker.summary
        assert summary.by_origin["user"].calls == 1
        assert summary.by_origin["framework"].calls == len(CallPurpose) - 1

    def test_reset_clears_by_origin(self):
        tracker = UsageTracker()
        tracker.record(CallPurpose.MAIN, {"prompt_tokens": 100, "total_tokens": 100})
        tracker.reset()
        assert tracker.summary.by_origin == {}


# ================================================================== #
# record_from_response — extract usage from provider response          #
# ================================================================== #


class TestRecordFromResponse:
    def test_response_with_usage_attribute(self):
        tracker = UsageTracker()
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                reasoning_tokens=0,
            ),
        )
        tracker.record_from_response(CallPurpose.MAIN, response)
        assert tracker.call_count == 1
        assert tracker.total_tokens == 150

    def test_response_with_pydantic_usage(self):
        tracker = UsageTracker()

        class MockUsage:
            prompt_tokens = 80
            completion_tokens = 40
            total_tokens = 120
            reasoning_tokens = 10

            def model_dump(self):
                return {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens,
                    "reasoning_tokens": self.reasoning_tokens,
                }

        response = SimpleNamespace(usage=MockUsage())
        tracker.record_from_response(CallPurpose.CRITIC, response)
        assert tracker.call_count == 1
        assert tracker.records[0].purpose == CallPurpose.CRITIC
        assert tracker.total_tokens == 120

    def test_response_with_dict_usage(self):
        tracker = UsageTracker()
        response = {
            "usage": {"prompt_tokens": 60, "completion_tokens": 30, "total_tokens": 90}
        }
        tracker.record_from_response(CallPurpose.PLANNING, response)
        assert tracker.call_count == 1
        assert tracker.total_tokens == 90

    def test_response_no_usage(self):
        tracker = UsageTracker()
        response = SimpleNamespace()
        tracker.record_from_response(CallPurpose.MAIN, response)
        assert tracker.call_count == 0

    def test_response_none(self):
        tracker = UsageTracker()
        tracker.record_from_response(CallPurpose.MAIN, None)
        assert tracker.call_count == 0


# ================================================================== #
# record_from_stream_metadata — extract usage from StreamEvent         #
# ================================================================== #


class TestRecordFromStreamMetadata:
    def test_metadata_with_usage_dict(self):
        tracker = UsageTracker()
        metadata = {
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        }
        tracker.record_from_stream_metadata(CallPurpose.MAIN, metadata)
        assert tracker.call_count == 1
        assert tracker.total_tokens == 70

    def test_metadata_with_pydantic_usage(self):
        tracker = UsageTracker()

        class Usage:
            def model_dump(self):
                return {
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                    "total_tokens": 40,
                }

        metadata = {"usage": Usage()}
        tracker.record_from_stream_metadata(
            CallPurpose.TOOL_LOOP, metadata, call_round=2
        )
        assert tracker.call_count == 1
        assert tracker.records[0].call_round == 2

    def test_metadata_no_usage(self):
        tracker = UsageTracker()
        tracker.record_from_stream_metadata(CallPurpose.MAIN, {"model": "gpt-4o"})
        assert tracker.call_count == 0

    def test_metadata_none(self):
        tracker = UsageTracker()
        tracker.record_from_stream_metadata(CallPurpose.MAIN, None)
        assert tracker.call_count == 0

    def test_empty_metadata(self):
        tracker = UsageTracker()
        tracker.record_from_stream_metadata(CallPurpose.MAIN, {})
        assert tracker.call_count == 0


# ================================================================== #
# Integration: call_llm records usage                                  #
# ================================================================== #


class TestCallLlmUsageRecording:
    @pytest.mark.asyncio
    async def test_call_llm_records_usage(self):
        from nucleusiq.agents.modes.base_mode import BaseExecutionMode

        class DummyMode(BaseExecutionMode):
            async def run(self, agent, task):
                return "done"

        usage_obj = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=0,
        )
        mock_response = SimpleNamespace(
            usage=usage_obj,
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=mock_response)

        tracker = UsageTracker()
        mock_agent = MagicMock()
        mock_agent.llm = mock_llm
        mock_agent._plugin_manager = None
        mock_agent._usage_tracker = tracker

        mode = DummyMode()
        call_kwargs: dict[str, Any] = {
            "model": "test",
            "messages": [],
            "max_output_tokens": 100,
        }

        await mode.call_llm(mock_agent, call_kwargs, purpose=CallPurpose.MAIN)
        assert tracker.call_count == 1
        assert tracker.total_tokens == 150
        assert tracker.records[0].purpose == CallPurpose.MAIN

    @pytest.mark.asyncio
    async def test_call_llm_with_critic_purpose(self):
        from nucleusiq.agents.modes.base_mode import BaseExecutionMode

        class DummyMode(BaseExecutionMode):
            async def run(self, agent, task):
                return "done"

        usage = {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}
        mock_response = {"usage": usage, "choices": [{"message": {"content": "ok"}}]}

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=mock_response)

        tracker = UsageTracker()
        mock_agent = MagicMock()
        mock_agent.llm = mock_llm
        mock_agent._plugin_manager = None
        mock_agent._usage_tracker = tracker

        mode = DummyMode()
        await mode.call_llm(
            mock_agent,
            {"model": "test", "messages": [], "max_output_tokens": 100},
            purpose=CallPurpose.CRITIC,
        )
        assert tracker.records[0].purpose == CallPurpose.CRITIC

    @pytest.mark.asyncio
    async def test_call_llm_no_tracker(self):
        """Agent without _usage_tracker should not crash."""
        from nucleusiq.agents.modes.base_mode import BaseExecutionMode

        class DummyMode(BaseExecutionMode):
            async def run(self, agent, task):
                return "done"

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=SimpleNamespace(choices=[]))

        mock_agent = MagicMock()
        mock_agent.llm = mock_llm
        mock_agent._plugin_manager = None
        del mock_agent._usage_tracker

        mode = DummyMode()
        await mode.call_llm(
            mock_agent, {"model": "test", "messages": [], "max_output_tokens": 100}
        )


# ================================================================== #
# Integration: Agent.last_usage                                        #
# ================================================================== #


class TestAgentLastUsage:
    def test_agent_has_usage_tracker(self):
        from nucleusiq.agents.agent import Agent

        agent = Agent(
            name="test", role="test", objective="test", prompt=make_test_prompt()
        )
        assert hasattr(agent, "_usage_tracker")
        assert isinstance(agent._usage_tracker, UsageTracker)

    def test_last_usage_returns_pydantic_model(self):
        from nucleusiq.agents.agent import Agent

        agent = Agent(
            name="test", role="test", objective="test", prompt=make_test_prompt()
        )
        usage = agent.last_usage
        assert isinstance(usage, UsageSummary)
        assert usage.call_count == 0
        assert usage.total.total_tokens == 0

    def test_last_usage_model_dump(self):
        """model_dump() gives a plain dict for serialization / logging."""
        from nucleusiq.agents.agent import Agent

        agent = Agent(
            name="test", role="test", objective="test", prompt=make_test_prompt()
        )
        agent.usage_tracker.record(
            CallPurpose.MAIN,
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        d = agent.last_usage.model_dump()
        assert isinstance(d, dict)
        assert d["total"]["prompt_tokens"] == 100
        assert d["by_purpose"]["main"]["calls"] == 1

    def test_usage_tracker_property(self):
        from nucleusiq.agents.agent import Agent

        agent = Agent(
            name="test", role="test", objective="test", prompt=make_test_prompt()
        )
        tracker = agent.usage_tracker
        assert isinstance(tracker, UsageTracker)
        tracker.record(CallPurpose.MAIN, {"prompt_tokens": 50, "total_tokens": 50})
        assert agent.last_usage.call_count == 1

    @pytest.mark.asyncio
    async def test_usage_reset_on_execute(self):
        """Verify _usage_tracker.reset() is called in _setup_execution."""
        from nucleusiq.agents.agent import Agent
        from nucleusiq.llms.mock_llm import MockLLM

        agent = Agent(
            name="test",
            role="test",
            objective="test",
            prompt=make_test_prompt(),
            llm=MockLLM(),
        )
        await agent.initialize()

        agent._usage_tracker.record(
            CallPurpose.MAIN, {"prompt_tokens": 999, "total_tokens": 999}
        )
        assert agent.last_usage.call_count == 1

        from nucleusiq.agents.task import Task

        task = Task(id="t1", objective="hello")
        await agent.execute(task)

        summary = agent.last_usage
        assert summary.total.total_tokens != 999 or summary.call_count == 0


# ================================================================== #
# Integration: streaming tool loop records usage per round             #
# ================================================================== #


class TestStreamingToolLoopUsage:
    @pytest.mark.asyncio
    async def test_streaming_loop_records_usage(self):
        """The _streaming_tool_call_loop should record usage from COMPLETE events."""
        from nucleusiq.agents.modes.base_mode import BaseExecutionMode
        from nucleusiq.streaming.events import StreamEvent

        class DummyMode(BaseExecutionMode):
            async def run(self, agent, task):
                return "done"

        usage_meta = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "tool_calls": [],
        }

        async def mock_stream(**kwargs):
            yield StreamEvent.token_event("Hello")
            yield StreamEvent.complete_event("Hello world", metadata=usage_meta)

        mock_llm = MagicMock()
        mock_llm.call_stream = mock_stream
        mock_llm.model_name = "test-model"

        tracker = UsageTracker()
        mock_agent = MagicMock()
        mock_agent.llm = mock_llm
        mock_agent._plugin_manager = None
        mock_agent._usage_tracker = tracker
        mock_agent._current_llm_overrides = {}
        mock_agent.config = MagicMock()
        mock_agent.config.llm_max_output_tokens = 1024
        mock_agent._resolve_response_format = MagicMock(return_value=None)
        mock_agent._get_structured_output_kwargs = MagicMock(return_value={})

        mode = DummyMode()
        from nucleusiq.agents.chat_models import ChatMessage

        messages = [ChatMessage(role="user", content="test")]

        events = []
        async for event in mode._streaming_tool_call_loop(
            mock_agent, messages, None, max_tool_calls=5
        ):
            events.append(event)

        assert tracker.call_count == 1
        assert tracker.total_tokens == 150
        assert tracker.records[0].purpose == CallPurpose.MAIN
