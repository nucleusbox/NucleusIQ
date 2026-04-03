"""Integration tests for OpenAI native file input pipeline.

Verifies the full path from Task creation through to the wire format
that would be sent to OpenAI's API:

    Task (with file attachments)
      -> MessageBuilder.build(attachment_processor=llm.process_attachments)
        -> OpenAI-native content parts (Chat Completions format)
          -> messages_to_responses_input() (Responses API format)
            -> Wire payload ready for API call

Tests are grouped by scenario:
    1. PDF file — full pipeline from Task to Responses API wire format
    2. FILE_URL — Responses API input_file with file_url
    3. CSV/XLSX — file_bytes with supported extension -> native file
    4. Mixed — multiple types produce correct wire format
    5. Chat Completions path — file parts stay in CC format
    6. Mocked API call — verify actual payload sent to OpenAI client
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nucleusiq.agents.attachments import MAX_FILE_SIZE_BYTES, Attachment, AttachmentType
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.agents.task import Task


def _make_openai_llm():
    """Create a BaseOpenAI instance with a fake API key."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-fake"}):
        from nucleusiq_openai.nb_openai.base import BaseOpenAI

        return BaseOpenAI(model_name="gpt-4o", api_key="sk-test-fake")


# ================================================================== #
# 1. PDF — full pipeline to Responses API                             #
# ================================================================== #


class TestPDFPipeline:
    def test_pdf_task_to_responses_api_wire(self):
        """PDF attachment -> OpenAI native file -> Responses API input_file."""
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        llm = _make_openai_llm()
        pdf_bytes = b"%PDF-1.4 quarterly report data here"
        task = Task(
            id="pipe-pdf",
            objective="Summarize the quarterly report",
            attachments=[
                Attachment(
                    type=AttachmentType.PDF,
                    data=pdf_bytes,
                    name="Q4_report.pdf",
                ),
            ],
        )

        msgs = MessageBuilder.build(
            task,
            role="financial analyst",
            attachment_processor=llm.process_attachments,
        )
        msg_dicts = [m.to_dict() for m in msgs]

        instructions, items = messages_to_responses_input(msg_dicts, None)

        assert instructions is not None
        assert "financial analyst" in instructions

        user_items = [i for i in items if getattr(i, "role", None) == "user"]
        assert len(user_items) >= 1
        content = user_items[-1].content
        assert isinstance(content, list)

        input_text = [p for p in content if p["type"] == "input_text"]
        assert any("quarterly report" in p["text"] for p in input_text)

        input_file = [p for p in content if p["type"] == "input_file"]
        assert len(input_file) == 1
        assert input_file[0]["filename"] == "Q4_report.pdf"
        assert "application/pdf;base64," in input_file[0]["file_data"]

        decoded = base64.b64decode(input_file[0]["file_data"].split(",", 1)[1])
        assert decoded == pdf_bytes


# ================================================================== #
# 2. FILE_URL — Responses API input_file with file_url                #
# ================================================================== #


class TestFileURLPipeline:
    def test_file_url_to_responses_api(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        llm = _make_openai_llm()
        task = Task(
            id="pipe-url",
            objective="Extract data from this spreadsheet",
            attachments=[
                Attachment(
                    type=AttachmentType.FILE_URL,
                    data="https://storage.example.com/data.xlsx",
                    name="data.xlsx",
                ),
            ],
        )

        msgs = MessageBuilder.build(task, attachment_processor=llm.process_attachments)
        msg_dicts = [m.to_dict() for m in msgs]
        _, items = messages_to_responses_input(msg_dicts, None)

        user_items = [i for i in items if getattr(i, "role", None) == "user"]
        content = user_items[-1].content
        assert isinstance(content, list)

        input_file = [p for p in content if p["type"] == "input_file"]
        assert len(input_file) == 1
        assert input_file[0]["file_url"] == "https://storage.example.com/data.xlsx"


# ================================================================== #
# 3. CSV — file_bytes with supported extension                        #
# ================================================================== #


class TestCSVFilePipeline:
    def test_csv_file_bytes_to_native_format(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        llm = _make_openai_llm()
        csv_data = b"name,age,city\nAlice,30,NYC\nBob,25,SF"
        task = Task(
            id="pipe-csv",
            objective="What is the average age?",
            attachments=[
                Attachment(
                    type=AttachmentType.FILE_BYTES,
                    data=csv_data,
                    name="people.csv",
                ),
            ],
        )

        msgs = MessageBuilder.build(task, attachment_processor=llm.process_attachments)
        msg_dicts = [m.to_dict() for m in msgs]
        _, items = messages_to_responses_input(msg_dicts, None)

        content = items[-1].content
        input_file = [p for p in content if p["type"] == "input_file"]
        assert len(input_file) == 1
        assert input_file[0]["filename"] == "people.csv"
        assert "text/csv;base64," in input_file[0]["file_data"]

        decoded = base64.b64decode(input_file[0]["file_data"].split(",", 1)[1])
        assert decoded == csv_data


# ================================================================== #
# 4. Mixed attachments — correct type routing                         #
# ================================================================== #


class TestMixedPipeline:
    def test_mixed_types_correct_wire_format(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        llm = _make_openai_llm()
        task = Task(
            id="pipe-mix",
            objective="Analyze all of these together",
            attachments=[
                Attachment(
                    type=AttachmentType.PDF,
                    data=b"%PDF-1.4 content",
                    name="report.pdf",
                ),
                Attachment(
                    type=AttachmentType.IMAGE_URL,
                    data="https://img.example.com/chart.png",
                    detail="high",
                ),
                Attachment(
                    type=AttachmentType.TEXT,
                    data='{"revenue": 1000000}',
                    name="metrics.json",
                ),
                Attachment(
                    type=AttachmentType.FILE_URL,
                    data="https://cdn.example.com/extra.csv",
                    name="extra.csv",
                ),
            ],
        )

        msgs = MessageBuilder.build(task, attachment_processor=llm.process_attachments)
        msg_dicts = [m.to_dict() for m in msgs]
        _, items = messages_to_responses_input(msg_dicts, None)

        content = items[-1].content
        assert isinstance(content, list)

        types = [p["type"] for p in content]

        assert "input_text" in types
        assert types.count("input_file") >= 3
        assert "input_image" in types

        image_part = next(p for p in content if p["type"] == "input_image")
        assert image_part["image_url"] == "https://img.example.com/chart.png"
        assert image_part["detail"] == "high"


# ================================================================== #
# 5. Chat Completions path — file parts preserved                     #
# ================================================================== #


class TestChatCompletionsPath:
    def test_file_parts_preserved_for_chat_completions(self):
        """When NOT going through Responses API, file parts stay in CC format."""
        llm = _make_openai_llm()
        task = Task(
            id="pipe-cc",
            objective="Read this PDF",
            attachments=[
                Attachment(
                    type=AttachmentType.PDF,
                    data=b"%PDF test",
                    name="test.pdf",
                ),
            ],
        )

        msgs = MessageBuilder.build(task, attachment_processor=llm.process_attachments)
        user_msg = [m for m in msgs if m.role == "user"][-1]
        content = user_msg.content
        assert isinstance(content, list)

        file_parts = [p for p in content if p.get("type") == "file"]
        assert len(file_parts) == 1
        assert file_parts[0]["file"]["filename"] == "test.pdf"
        assert "application/pdf;base64," in file_parts[0]["file"]["file_data"]


# ================================================================== #
# 6. Mocked API call — verify payload structure                       #
# ================================================================== #


class TestMockedOpenAICallPayload:
    @pytest.mark.asyncio
    async def test_pdf_payload_sent_to_chat_completions(self):
        """Verify the actual payload sent to openai.chat.completions.create."""
        llm = _make_openai_llm()
        task = Task(
            id="pipe-mock",
            objective="Summarize",
            attachments=[
                Attachment(
                    type=AttachmentType.PDF,
                    data=b"%PDF mock content",
                    name="mock.pdf",
                ),
            ],
        )

        msgs = MessageBuilder.build(task, attachment_processor=llm.process_attachments)
        msg_dicts = [m.to_dict() for m in msgs]

        # The SDK's message object needs model_dump() — use MagicMock
        mock_msg = MagicMock()
        mock_msg.model_dump.return_value = {
            "role": "assistant",
            "content": "Summary of the PDF.",
            "tool_calls": None,
            "function_call": None,
        }
        mock_msg.content = "Summary of the PDF."
        mock_msg.tool_calls = None

        mock_choice = SimpleNamespace(
            message=mock_msg,
            finish_reason="stop",
        )
        mock_response = SimpleNamespace(
            choices=[mock_choice],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
        )

        llm._client = MagicMock()
        llm._client.chat = MagicMock()
        llm._client.chat.completions = MagicMock()
        llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await llm.call(
            model="gpt-4o",
            messages=msg_dicts,
            max_output_tokens=1024,
        )

        call_args = llm._client.chat.completions.create.call_args
        sent_messages = call_args.kwargs.get(
            "messages", call_args[1].get("messages") if len(call_args) > 1 else None
        )

        assert sent_messages is not None
        user_msg = next(m for m in reversed(sent_messages) if m.get("role") == "user")
        content = user_msg["content"]
        assert isinstance(content, list)

        file_parts = [p for p in content if p.get("type") == "file"]
        assert len(file_parts) == 1
        assert file_parts[0]["file"]["filename"] == "mock.pdf"

    @pytest.mark.asyncio
    async def test_file_size_rejection(self):
        """Oversized file should raise ValueError before reaching the API."""
        llm = _make_openai_llm()
        big_data = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        att = Attachment(
            type=AttachmentType.PDF,
            data=big_data,
            name="huge.pdf",
        )
        from nucleusiq.agents.errors import AttachmentValidationError

        with pytest.raises(AttachmentValidationError, match="exceeding the 50 MB limit"):
            llm.process_attachments([att])
