"""Integration tests for the attachment pipeline.

Verifies that all layers work together end-to-end:

    Task (with attachments)
      -> Agent.execute()
        -> Mode.build_messages()
          -> MessageBuilder.build(attachment_processor=llm.process_attachments)
            -> ChatMessage (multimodal content array)
              -> BaseLLM.call() receives multimodal messages

Tests are grouped by scenario:
    1. Text attachment — flows from Task through to LLM messages
    2. Image attachment — produces image_url content part
    3. FILE_URL attachment — framework fallback for non-provider LLM
    4. Mixed attachments — multiple types in one Task
    5. Provider processor override — custom processor wired via mode
    6. UsageTracker + attachments — token counting with file tasks
    7. Streaming + attachments — events contain correct content
"""

from __future__ import annotations

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.attachments import Attachment, AttachmentType
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM

from nucleusiq.tests.conftest import make_test_prompt

# ================================================================== #
# 1. Text attachment flow-through                                      #
# ================================================================== #


class TestTextAttachmentIntegration:
    @pytest.mark.asyncio
    async def test_text_attachment_reaches_llm_messages(self):
        """Task.attachments -> build_messages -> multimodal content -> MockLLM."""
        captured: list[list[dict]] = []
        original_call = MockLLM.call

        async def spy_call(self_llm, *, messages, **kwargs):
            captured.append(messages)
            return await original_call(self_llm, messages=messages, **kwargs)

        MockLLM.call = spy_call
        try:
            agent = Agent(
                name="TextBot",
                role="Analyst",
                objective="Read documents",
                prompt=make_test_prompt(),
                llm=MockLLM(),
                config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
            )

            task = Task(
                id="int-text-1",
                objective="Summarize the document below",
                attachments=[
                    Attachment(
                        type=AttachmentType.TEXT,
                        data="NucleusIQ is an agent framework.",
                        name="readme.txt",
                    ),
                ],
            )
            result = await agent.execute(task)
            assert result is not None

            assert len(captured) >= 1
            last_msgs = captured[-1]
            user_msg = next(m for m in reversed(last_msgs) if m.get("role") == "user")
            content = user_msg["content"]
            assert isinstance(content, list)
            text_parts = [p for p in content if p.get("type") == "text"]
            assert any("NucleusIQ" in p["text"] for p in text_parts)
        finally:
            MockLLM.call = original_call

    @pytest.mark.asyncio
    async def test_task_without_attachments_sends_plain_string(self):
        captured: list[list[dict]] = []
        original_call = MockLLM.call

        async def spy_call(self_llm, *, messages, **kwargs):
            captured.append(messages)
            return await original_call(self_llm, messages=messages, **kwargs)

        MockLLM.call = spy_call
        try:
            agent = Agent(
                name="PlainBot",
                role="Helper",
                objective="Answer questions",
                prompt=make_test_prompt(),
                llm=MockLLM(),
                config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
            )

            task = Task(id="int-plain", objective="What is Python?")
            await agent.execute(task)

            last_msgs = captured[-1]
            user_msg = next(m for m in reversed(last_msgs) if m.get("role") == "user")
            assert isinstance(user_msg["content"], str)
        finally:
            MockLLM.call = original_call


# ================================================================== #
# 2. Image attachment flow-through                                     #
# ================================================================== #


class TestImageAttachmentIntegration:
    @pytest.mark.asyncio
    async def test_image_url_produces_image_content_part(self):
        captured: list = []
        original_call = MockLLM.call

        async def spy_call(self_llm, *, messages, **kwargs):
            captured.append(messages)
            return await original_call(self_llm, messages=messages, **kwargs)

        MockLLM.call = spy_call
        try:
            agent = Agent(
                name="VisionBot",
                role="Image analyzer",
                objective="Describe images",
                prompt=make_test_prompt(),
                llm=MockLLM(),
                config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
            )

            task = Task(
                id="int-img",
                objective="What is in this image?",
                attachments=[
                    Attachment(
                        type=AttachmentType.IMAGE_URL,
                        data="https://example.com/photo.jpg",
                        detail="high",
                    ),
                ],
            )
            await agent.execute(task)

            last_msgs = captured[-1]
            user_msg = next(m for m in reversed(last_msgs) if m.get("role") == "user")
            content = user_msg["content"]
            assert isinstance(content, list)
            img_parts = [p for p in content if p.get("type") == "image_url"]
            assert len(img_parts) == 1
            assert img_parts[0]["image_url"]["url"] == "https://example.com/photo.jpg"
            assert img_parts[0]["image_url"]["detail"] == "high"
        finally:
            MockLLM.call = original_call


# ================================================================== #
# 3. FILE_URL attachment (framework fallback)                          #
# ================================================================== #


class TestFileUrlIntegration:
    @pytest.mark.asyncio
    async def test_file_url_framework_fallback(self):
        captured: list = []
        original_call = MockLLM.call

        async def spy_call(self_llm, *, messages, **kwargs):
            captured.append(messages)
            return await original_call(self_llm, messages=messages, **kwargs)

        MockLLM.call = spy_call
        try:
            agent = Agent(
                name="FileBot",
                role="Document reader",
                objective="Process files",
                prompt=make_test_prompt(),
                llm=MockLLM(),
                config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
            )

            task = Task(
                id="int-url",
                objective="Summarize this file",
                attachments=[
                    Attachment(
                        type=AttachmentType.FILE_URL,
                        data="https://example.com/report.pdf",
                        name="report.pdf",
                    ),
                ],
            )
            await agent.execute(task)

            last_msgs = captured[-1]
            user_msg = next(m for m in reversed(last_msgs) if m.get("role") == "user")
            content = user_msg["content"]
            assert isinstance(content, list)
            text_parts = [p for p in content if p.get("type") == "text"]
            assert any("File URL" in p["text"] for p in text_parts)
        finally:
            MockLLM.call = original_call


# ================================================================== #
# 4. Mixed attachments                                                 #
# ================================================================== #


class TestMixedAttachmentIntegration:
    @pytest.mark.asyncio
    async def test_multiple_attachment_types(self):
        captured: list = []
        original_call = MockLLM.call

        async def spy_call(self_llm, *, messages, **kwargs):
            captured.append(messages)
            return await original_call(self_llm, messages=messages, **kwargs)

        MockLLM.call = spy_call
        try:
            agent = Agent(
                name="MultiBot",
                role="Researcher",
                objective="Analyze multiple sources",
                prompt=make_test_prompt(),
                llm=MockLLM(),
                config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
            )

            task = Task(
                id="int-mixed",
                objective="Compare these inputs",
                attachments=[
                    Attachment(
                        type=AttachmentType.TEXT,
                        data="Revenue grew 15%.",
                        name="summary.txt",
                    ),
                    Attachment(
                        type=AttachmentType.IMAGE_URL,
                        data="https://example.com/chart.png",
                    ),
                    Attachment(
                        type=AttachmentType.FILE_BYTES,
                        data=b"col1,col2\n1,2\n3,4",
                        name="data.csv",
                    ),
                ],
            )
            await agent.execute(task)

            last_msgs = captured[-1]
            user_msg = next(m for m in reversed(last_msgs) if m.get("role") == "user")
            content = user_msg["content"]
            assert isinstance(content, list)
            types = {p["type"] for p in content}
            assert "text" in types
            assert "image_url" in types
        finally:
            MockLLM.call = original_call


# ================================================================== #
# 5. Provider processor override wired through mode                    #
# ================================================================== #


class TestProviderProcessorIntegration:
    @pytest.mark.asyncio
    async def test_custom_llm_process_attachments_used(self):
        """If llm has process_attachments, it should be used instead of framework default."""
        call_log: list = []

        class CustomLLM(MockLLM):
            def process_attachments(self, attachments):
                call_log.append(len(attachments))
                return [{"type": "custom_provider", "data": "transformed"}]

        agent = Agent(
            name="CustomBot",
            role="Transformer",
            objective="Transform inputs",
            prompt=make_test_prompt(),
            llm=CustomLLM(),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )

        task = Task(
            id="int-custom",
            objective="Process this",
            attachments=[
                Attachment(type=AttachmentType.TEXT, data="test", name="f.txt"),
            ],
        )
        await agent.execute(task)

        assert len(call_log) == 1
        assert call_log[0] == 1


# ================================================================== #
# 6. UsageTracker with attachment tasks                                #
# ================================================================== #


class TestUsageTrackerWithAttachments:
    @pytest.mark.asyncio
    async def test_usage_tracker_available_after_attachment_task(self):
        """Verify UsageTracker structure is populated after an attachment task.

        MockLLM responses don't include usage data, so call_count will be 0
        (no records created). The test verifies the tracker is accessible
        and structurally correct — real providers emit usage automatically.
        """
        agent = Agent(
            name="TrackerBot",
            role="Analyst",
            objective="Analyze files",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )

        task = Task(
            id="int-usage",
            objective="Summarize this document",
            attachments=[
                Attachment(
                    type=AttachmentType.TEXT,
                    data="Long document content here.",
                    name="doc.txt",
                ),
            ],
        )
        result = await agent.execute(task)
        assert result is not None

        usage = agent.last_usage
        assert hasattr(usage, "total")
        assert hasattr(usage, "call_count")
        assert hasattr(usage, "by_purpose")
        assert isinstance(usage.total.prompt_tokens, int)
        assert isinstance(usage.total.completion_tokens, int)

    @pytest.mark.asyncio
    async def test_usage_tracker_records_with_usage_in_response(self):
        """Simulate an LLM that returns usage data alongside attachments."""
        from types import SimpleNamespace

        class UsageMockLLM(MockLLM):
            async def call(self, *, messages, **kwargs):
                resp = await super().call(messages=messages, **kwargs)
                resp.usage = SimpleNamespace(
                    prompt_tokens=250,
                    completion_tokens=50,
                    total_tokens=300,
                    reasoning_tokens=0,
                )
                return resp

        agent = Agent(
            name="UsageBot",
            role="Analyst",
            objective="Analyze files",
            prompt=make_test_prompt(),
            llm=UsageMockLLM(),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )

        task = Task(
            id="int-usage-real",
            objective="Summarize this",
            attachments=[
                Attachment(type=AttachmentType.TEXT, data="Content.", name="f.txt"),
            ],
        )
        await agent.execute(task)

        usage = agent.last_usage
        assert usage.call_count >= 1
        assert usage.total.prompt_tokens >= 250
        assert usage.total.total_tokens >= 300


# ================================================================== #
# 7. Streaming with attachments                                        #
# ================================================================== #


class TestStreamingWithAttachments:
    @pytest.mark.asyncio
    async def test_stream_with_text_attachment(self):
        agent = Agent(
            name="StreamBot",
            role="Reader",
            objective="Read files",
            prompt=make_test_prompt(),
            llm=MockLLM(stream_chunk_size=5),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )

        task = Task(
            id="int-stream",
            objective="Read and summarize",
            attachments=[
                Attachment(
                    type=AttachmentType.TEXT,
                    data="Stream test content.",
                    name="notes.txt",
                ),
            ],
        )

        events = []
        async for event in agent.execute_stream(task):
            events.append(event)

        types = {e.type for e in events}
        assert "complete" in types

    @pytest.mark.asyncio
    async def test_stream_with_image_attachment(self):
        agent = Agent(
            name="VisionStream",
            role="Vision",
            objective="Describe images",
            prompt=make_test_prompt(),
            llm=MockLLM(stream_chunk_size=3),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )

        task = Task(
            id="int-stream-img",
            objective="Describe this",
            attachments=[
                Attachment(
                    type=AttachmentType.IMAGE_URL,
                    data="https://example.com/photo.jpg",
                ),
            ],
        )

        events = []
        async for event in agent.execute_stream(task):
            events.append(event)

        assert any(e.type == "complete" for e in events)
