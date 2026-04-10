"""Tests for Attachment model, AttachmentProcessor, and multimodal message building."""

from __future__ import annotations

import base64

from nucleusiq.agents.attachments import (
    MAX_FILE_SIZE_BYTES,
    Attachment,
    AttachmentProcessor,
    AttachmentType,
    ContentPart,
)
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.errors import (
    AttachmentUnsupportedError,
    AttachmentValidationError,
)
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.agents.task import Task

# ================================================================== #
# Attachment model                                                     #
# ================================================================== #


class TestAttachmentModel:
    def test_text_attachment(self):
        att = Attachment(type=AttachmentType.TEXT, data="hello world", name="note.txt")
        assert att.type == AttachmentType.TEXT
        assert att.data == "hello world"
        assert att.name == "note.txt"
        assert att.mime_type is None
        assert att.detail is None

    def test_image_url_attachment(self):
        att = Attachment(
            type="image_url",
            data="https://example.com/img.png",
            detail="high",
        )
        assert att.type == AttachmentType.IMAGE_URL
        assert att.detail == "high"

    def test_image_base64_attachment(self):
        raw = b"\x89PNG\r\n\x1a\n"
        att = Attachment(
            type=AttachmentType.IMAGE_BASE64,
            data=raw,
            mime_type="image/png",
        )
        assert att.type == AttachmentType.IMAGE_BASE64
        assert att.data == raw
        assert att.mime_type == "image/png"

    def test_pdf_attachment(self):
        att = Attachment(type=AttachmentType.PDF, data=b"%PDF-1.4", name="report.pdf")
        assert att.type == AttachmentType.PDF

    def test_file_bytes_attachment(self):
        att = Attachment(
            type=AttachmentType.FILE_BYTES,
            data=b"csv,data,here",
            name="data.csv",
        )
        assert att.type == AttachmentType.FILE_BYTES

    def test_attachment_serialization(self):
        att = Attachment(type="text", data="content", name="file.txt")
        d = att.model_dump()
        assert d["type"] == "text"
        assert d["data"] == "content"
        assert d["name"] == "file.txt"

    def test_attachment_enum_values(self):
        assert AttachmentType.IMAGE_URL == "image_url"
        assert AttachmentType.IMAGE_BASE64 == "image_base64"
        assert AttachmentType.TEXT == "text"
        assert AttachmentType.PDF == "pdf"
        assert AttachmentType.FILE_BYTES == "file_bytes"


# ================================================================== #
# AttachmentProcessor                                                  #
# ================================================================== #


class TestAttachmentProcessor:
    def test_text_string(self):
        att = Attachment(type="text", data="Hello world", name="greeting.txt")
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert parts[0].type == "text"
        assert "[File: greeting.txt]" in parts[0].text
        assert "Hello world" in parts[0].text

    def test_text_bytes(self):
        att = Attachment(type="text", data=b"byte content")
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert "byte content" in parts[0].text

    def test_text_no_name(self):
        att = Attachment(type="text", data="plain content")
        parts = AttachmentProcessor.process([att])
        assert parts[0].text == "plain content"
        assert "[File:" not in parts[0].text

    def test_image_url(self):
        att = Attachment(
            type="image_url",
            data="https://example.com/photo.jpg",
            detail="high",
        )
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert parts[0].type == "image_url"
        assert parts[0].image_url["url"] == "https://example.com/photo.jpg"
        assert parts[0].image_url["detail"] == "high"

    def test_image_url_no_detail(self):
        att = Attachment(type="image_url", data="https://example.com/img.png")
        parts = AttachmentProcessor.process([att])
        assert "detail" not in parts[0].image_url

    def test_image_url_bytes_data(self):
        att = Attachment(type="image_url", data=b"https://example.com/img.png")
        parts = AttachmentProcessor.process([att])
        assert parts[0].image_url["url"] == "https://example.com/img.png"

    def test_image_base64_from_bytes(self):
        raw = b"\x89PNG fake image data"
        att = Attachment(
            type="image_base64",
            data=raw,
            mime_type="image/png",
            detail="low",
        )
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert parts[0].type == "image_url"
        expected_b64 = base64.b64encode(raw).decode()
        assert parts[0].image_url["url"] == f"data:image/png;base64,{expected_b64}"
        assert parts[0].image_url["detail"] == "low"

    def test_image_base64_from_string(self):
        b64_str = base64.b64encode(b"fake image").decode()
        att = Attachment(type="image_base64", data=b64_str, mime_type="image/jpeg")
        parts = AttachmentProcessor.process([att])
        assert f"data:image/jpeg;base64,{b64_str}" in parts[0].image_url["url"]

    def test_image_base64_default_mime(self):
        att = Attachment(type="image_base64", data=b"img")
        parts = AttachmentProcessor.process([att])
        assert "data:image/png;base64," in parts[0].image_url["url"]

    def test_pdf_without_pdfplumber(self, monkeypatch):
        """When pdfplumber is not installed, falls back to placeholder."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pdfplumber":
                raise ImportError("no pdfplumber")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        att = Attachment(type="pdf", data=b"%PDF-1.4", name="report.pdf")
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert parts[0].type == "text"
        assert "pdfplumber" in parts[0].text
        assert "[PDF: report.pdf]" in parts[0].text

    def test_file_bytes_utf8(self):
        att = Attachment(type="file_bytes", data=b"readable utf8 text", name="data.csv")
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert "readable utf8 text" in parts[0].text
        assert "[File: data.csv]" in parts[0].text

    def test_file_bytes_binary(self):
        binary = bytes(range(128, 256))
        att = Attachment(type="file_bytes", data=binary, name="blob.bin")
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert "base64-encoded" in parts[0].text

    def test_file_bytes_string(self):
        att = Attachment(type="file_bytes", data="string content", name="file.txt")
        parts = AttachmentProcessor.process([att])
        assert "string content" in parts[0].text

    def test_file_bytes_string_no_name(self):
        att = Attachment(type="file_bytes", data="content")
        parts = AttachmentProcessor.process([att])
        assert parts[0].text == "content"

    def test_multiple_attachments(self):
        atts = [
            Attachment(type="text", data="file 1 content", name="a.txt"),
            Attachment(type="image_url", data="https://img.com/b.png"),
            Attachment(type="text", data="file 2 content"),
        ]
        parts = AttachmentProcessor.process(atts)
        assert len(parts) == 3
        assert parts[0].type == "text"
        assert parts[1].type == "image_url"
        assert parts[2].type == "text"

    def test_empty_list(self):
        parts = AttachmentProcessor.process([])
        assert parts == []


# ================================================================== #
# ContentPart model                                                    #
# ================================================================== #


class TestContentPart:
    def test_text_part(self):
        p = ContentPart(type="text", text="hello")
        assert p.type == "text"
        assert p.text == "hello"
        assert p.image_url is None

    def test_image_part(self):
        p = ContentPart(type="image_url", image_url={"url": "https://x.com/i.png"})
        assert p.type == "image_url"
        assert p.text is None


# ================================================================== #
# Task.attachments                                                     #
# ================================================================== #


class TestTaskAttachments:
    def test_task_without_attachments(self):
        task = Task(id="t1", objective="hello")
        assert task.attachments is None
        d = task.to_dict()
        assert d["attachments"] is None

    def test_task_with_attachments(self):
        task = Task(
            id="t1",
            objective="Describe this image",
            attachments=[
                Attachment(type="image_url", data="https://img.com/pic.jpg"),
            ],
        )
        assert len(task.attachments) == 1
        assert task.attachments[0].type == AttachmentType.IMAGE_URL

    def test_task_from_dict_without_attachments(self):
        task = Task.from_dict({"id": "t1", "objective": "test"})
        assert task.attachments is None

    def test_task_serialization_with_attachments(self):
        task = Task(
            id="t1",
            objective="analyze",
            attachments=[
                Attachment(type="text", data="content", name="f.txt"),
            ],
        )
        d = task.to_dict()
        assert len(d["attachments"]) == 1
        assert d["attachments"][0]["type"] == "text"
        assert d["attachments"][0]["data"] == "content"


# ================================================================== #
# MessageBuilder — multimodal                                          #
# ================================================================== #


class TestMessageBuilderMultimodal:
    def test_plain_task_no_attachments(self):
        task = Task(id="t1", objective="What is 2+2?")
        msgs = MessageBuilder.build(task)
        user_msg = msgs[-1]
        assert user_msg.role == "user"
        assert user_msg.content == "What is 2+2?"
        assert isinstance(user_msg.content, str)

    def test_task_with_text_attachment(self):
        task = Task(
            id="t1",
            objective="Summarize this file",
            attachments=[
                Attachment(type="text", data="file content here", name="doc.txt"),
            ],
        )
        msgs = MessageBuilder.build(task)
        user_msg = msgs[-1]
        assert user_msg.role == "user"
        assert isinstance(user_msg.content, list)
        assert len(user_msg.content) == 2
        assert user_msg.content[0] == {"type": "text", "text": "Summarize this file"}
        assert user_msg.content[1]["type"] == "text"
        assert "file content here" in user_msg.content[1]["text"]

    def test_task_with_image_attachment(self):
        task = Task(
            id="t1",
            objective="What is in this image?",
            attachments=[
                Attachment(
                    type="image_url",
                    data="https://example.com/photo.jpg",
                    detail="high",
                ),
            ],
        )
        msgs = MessageBuilder.build(task)
        user_msg = msgs[-1]
        assert isinstance(user_msg.content, list)
        assert len(user_msg.content) == 2
        assert user_msg.content[0] == {
            "type": "text",
            "text": "What is in this image?",
        }
        assert user_msg.content[1]["type"] == "image_url"
        assert (
            user_msg.content[1]["image_url"]["url"] == "https://example.com/photo.jpg"
        )
        assert user_msg.content[1]["image_url"]["detail"] == "high"

    def test_task_with_mixed_attachments(self):
        task = Task(
            id="t1",
            objective="Analyze these",
            attachments=[
                Attachment(type="text", data="csv data", name="sales.csv"),
                Attachment(type="image_url", data="https://example.com/chart.png"),
            ],
        )
        msgs = MessageBuilder.build(task)
        user_msg = msgs[-1]
        assert isinstance(user_msg.content, list)
        assert len(user_msg.content) == 3
        assert user_msg.content[0]["type"] == "text"
        assert user_msg.content[0]["text"] == "Analyze these"
        assert user_msg.content[1]["type"] == "text"
        assert "csv data" in user_msg.content[1]["text"]
        assert user_msg.content[2]["type"] == "image_url"

    def test_multimodal_message_serialization(self):
        """Ensure ChatMessage.to_dict() preserves content arrays."""
        task = Task(
            id="t1",
            objective="Describe",
            attachments=[
                Attachment(type="image_url", data="https://img.com/a.jpg"),
            ],
        )
        msgs = MessageBuilder.build(task)
        user_msg = msgs[-1]
        d = user_msg.to_dict()
        assert isinstance(d["content"], list)
        assert d["role"] == "user"

    def test_dict_task_no_attachments(self):
        """Dict-based tasks produce plain string content (backward compat)."""
        msgs = MessageBuilder.build(
            {"id": "t1", "objective": "hello"},
        )
        assert isinstance(msgs[-1].content, str)

    def test_task_empty_attachments_list(self):
        """Empty attachments list behaves like no attachments."""
        task = Task(id="t1", objective="plain task", attachments=[])
        msgs = MessageBuilder.build(task)
        assert isinstance(msgs[-1].content, str)
        assert msgs[-1].content == "plain task"


# ================================================================== #
# ChatMessage content array support                                    #
# ================================================================== #


class TestChatMessageMultimodal:
    def test_string_content(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.content == "hello"
        d = msg.to_dict()
        assert d["content"] == "hello"

    def test_list_content(self):
        content = [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "https://x.com/i.png"}},
        ]
        msg = ChatMessage(role="user", content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        d = msg.to_dict()
        assert isinstance(d["content"], list)

    def test_none_content(self):
        msg = ChatMessage(role="assistant", content=None)
        assert msg.content is None

    def test_from_dict_with_list_content(self):
        d = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "http://img"}},
            ],
        }
        msg = ChatMessage.from_dict(d)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2


# ================================================================== #
# FILE_URL attachment type                                              #
# ================================================================== #


# ================================================================== #
# FILE_BASE64 attachment type                                           #
# ================================================================== #


class TestFileBase64Attachment:
    def test_file_base64_model(self):
        att = Attachment(
            type=AttachmentType.FILE_BASE64,
            data=base64.b64encode(b"hello world").decode(),
            name="data.csv",
            mime_type="text/csv",
        )
        assert att.type == AttachmentType.FILE_BASE64
        assert isinstance(att.data, str)

    def test_file_base64_text_content(self):
        """Base64 of UTF-8 text -> framework decodes and extracts text."""
        original = b"col1,col2\n1,2\n3,4"
        att = Attachment(
            type=AttachmentType.FILE_BASE64,
            data=base64.b64encode(original).decode(),
            name="data.csv",
        )
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert parts[0].type == "text"
        assert "col1,col2" in parts[0].text
        assert "[File: data.csv]" in parts[0].text

    def test_file_base64_binary_content(self):
        """Base64 of binary data -> framework returns size label."""
        binary = bytes(range(256))
        att = Attachment(
            type=AttachmentType.FILE_BASE64,
            data=base64.b64encode(binary).decode(),
            name="data.bin",
        )
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert parts[0].type == "text"
        assert "Binary file" in parts[0].text
        assert "256 bytes" in parts[0].text

    def test_file_base64_invalid_data(self):
        """Invalid base64 string -> graceful fallback."""
        att = Attachment(
            type=AttachmentType.FILE_BASE64,
            data="!!!not-valid-base64!!!",
            name="broken.dat",
        )
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert "invalid base64" in parts[0].text

    def test_file_base64_no_name(self):
        att = Attachment(
            type=AttachmentType.FILE_BASE64,
            data=base64.b64encode(b"text content").decode(),
        )
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert "text content" in parts[0].text

    def test_file_base64_in_task(self):
        b64 = base64.b64encode(b"task file content").decode()
        task = Task(
            id="t-b64",
            objective="Analyze",
            attachments=[
                Attachment(type=AttachmentType.FILE_BASE64, data=b64, name="f.txt"),
            ],
        )
        msgs = MessageBuilder.build(task)
        user_msg = [m for m in msgs if m.role == "user"][-1]
        assert isinstance(user_msg.content, list)
        text_parts = [p for p in user_msg.content if p.get("type") == "text"]
        assert any("task file content" in p["text"] for p in text_parts)

    def test_file_base64_default_process_attachments(self):
        """BaseLLM.process_attachments handles FILE_BASE64."""
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        b64 = base64.b64encode(b"csv,data").decode()
        att = Attachment(type=AttachmentType.FILE_BASE64, data=b64, name="f.csv")
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert "csv,data" in result[0]["text"]


class TestFileUrlAttachment:
    def test_file_url_model(self):
        att = Attachment(
            type=AttachmentType.FILE_URL,
            data="https://example.com/report.pdf",
            name="report.pdf",
        )
        assert att.type == AttachmentType.FILE_URL
        assert att.data == "https://example.com/report.pdf"

    def test_file_url_processor_fallback(self):
        att = Attachment(
            type=AttachmentType.FILE_URL,
            data="https://example.com/data.csv",
            name="data.csv",
        )
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert parts[0].type == "text"
        assert "File URL" in parts[0].text
        assert "https://example.com/data.csv" in parts[0].text
        assert "not fetched" in parts[0].text

    def test_file_url_with_bytes_data(self):
        att = Attachment(
            type=AttachmentType.FILE_URL,
            data=b"https://example.com/doc.pdf",
        )
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert "https://example.com/doc.pdf" in parts[0].text

    def test_file_url_in_multimodal_message(self):
        task = Task(
            id="t-url",
            objective="Summarize",
            attachments=[
                Attachment(
                    type=AttachmentType.FILE_URL,
                    data="https://example.com/letter.pdf",
                    name="letter.pdf",
                ),
            ],
        )
        msgs = MessageBuilder.build(task)
        user_msg = [m for m in msgs if m.role == "user"][-1]
        assert isinstance(user_msg.content, list)
        assert any("File URL" in str(p) for p in user_msg.content)


# ================================================================== #
# File size validation                                                  #
# ================================================================== #


class TestFileSizeValidation:
    def test_validate_size_under_limit(self):
        att = Attachment(type=AttachmentType.TEXT, data="short text")
        AttachmentProcessor.validate_size(att)

    def test_validate_size_over_limit(self):
        import pytest
        from nucleusiq.agents.attachments import MAX_FILE_SIZE_BYTES

        big_data = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        att = Attachment(type=AttachmentType.FILE_BYTES, data=big_data, name="big.bin")
        with pytest.raises(
            AttachmentValidationError, match="exceeding the 50 MB limit"
        ):
            AttachmentProcessor.validate_size(att)

    def test_validate_size_custom_limit(self):
        import pytest

        att = Attachment(type=AttachmentType.TEXT, data="hello world", name="f.txt")
        with pytest.raises(AttachmentValidationError, match="exceeding the 0 MB limit"):
            AttachmentProcessor.validate_size(att, limit=5)

    def test_validate_size_exactly_at_limit(self):
        att = Attachment(type=AttachmentType.TEXT, data="x" * 100)
        AttachmentProcessor.validate_size(att, limit=100)


# ================================================================== #
# BaseLLM.process_attachments (default implementation)                  #
# ================================================================== #


class TestBaseLLMProcessAttachments:
    def test_default_text(self):
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        att = Attachment(type=AttachmentType.TEXT, data="hello", name="note.txt")
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert "hello" in result[0]["text"]

    def test_default_image_url(self):
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        att = Attachment(type=AttachmentType.IMAGE_URL, data="https://img.com/pic.png")
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"] == "https://img.com/pic.png"

    def test_default_empty_list(self):
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        assert llm.process_attachments([]) == []

    def test_default_mixed_types(self):
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        atts = [
            Attachment(type=AttachmentType.TEXT, data="text content"),
            Attachment(type=AttachmentType.IMAGE_URL, data="https://img.com/i.png"),
        ]
        result = llm.process_attachments(atts)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"


# ================================================================== #
# MessageBuilder with attachment_processor parameter                    #
# ================================================================== #


class TestMessageBuilderProcessorParam:
    def test_custom_processor_called(self):
        called_with = []

        def custom_proc(attachments):
            called_with.extend(attachments)
            return [{"type": "custom", "data": "replaced"}]

        task = Task(
            id="t-proc",
            objective="Test",
            attachments=[
                Attachment(type=AttachmentType.TEXT, data="original", name="f.txt"),
            ],
        )
        msgs = MessageBuilder.build(task, attachment_processor=custom_proc)
        user_msg = [m for m in msgs if m.role == "user"][-1]
        assert isinstance(user_msg.content, list)
        assert len(called_with) == 1
        parts = user_msg.content
        custom_parts = [p for p in parts if p.get("type") == "custom"]
        assert len(custom_parts) == 1
        assert custom_parts[0]["data"] == "replaced"

    def test_none_processor_uses_framework_default(self):
        task = Task(
            id="t-def",
            objective="Describe",
            attachments=[
                Attachment(type=AttachmentType.TEXT, data="hello", name="f.txt"),
            ],
        )
        msgs = MessageBuilder.build(task, attachment_processor=None)
        user_msg = [m for m in msgs if m.role == "user"][-1]
        assert isinstance(user_msg.content, list)
        text_parts = [p for p in user_msg.content if p.get("type") == "text"]
        assert any("hello" in p["text"] for p in text_parts)


# ================================================================== #
# build_messages bug fix — Task object flows correctly                  #
# ================================================================== #


class TestBuildMessagesBugFix:
    def test_task_with_attachments_flows_through_mode(self):
        from unittest.mock import MagicMock

        from nucleusiq.agents.modes.base_mode import BaseExecutionMode

        class ConcreteMode(BaseExecutionMode):
            async def run(self, agent, task):
                pass

        mode = ConcreteMode()
        agent = MagicMock()
        agent.llm = None
        agent.prompt = None
        agent.role = "tester"
        agent.objective = "test"
        agent.memory = None
        agent._logger = MagicMock()

        task = Task(
            id="t-bug",
            objective="Summarize this file",
            attachments=[
                Attachment(
                    type=AttachmentType.TEXT, data="file content here", name="doc.txt"
                ),
            ],
        )

        messages = mode.build_messages(agent, task)
        user_msgs = [m for m in messages if m.role == "user"]
        last_user = user_msgs[-1]
        assert isinstance(last_user.content, list), (
            "Attachments should produce multimodal content array"
        )
        text_parts = [
            p
            for p in last_user.content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        assert any("file content here" in p["text"] for p in text_parts)

    def test_task_without_attachments_still_works(self):
        from unittest.mock import MagicMock

        from nucleusiq.agents.modes.base_mode import BaseExecutionMode

        class ConcreteMode(BaseExecutionMode):
            async def run(self, agent, task):
                pass

        mode = ConcreteMode()
        agent = MagicMock()
        agent.llm = None
        agent.prompt = None
        agent.role = "tester"
        agent.objective = "test"
        agent.memory = None
        agent._logger = MagicMock()

        task = Task(id="t-plain", objective="Just text")
        messages = mode.build_messages(agent, task)
        user_msgs = [m for m in messages if m.role == "user"]
        last_user = user_msgs[-1]
        assert isinstance(last_user.content, str)
        assert "Just text" in last_user.content

    def test_provider_processor_wired_through_mode(self):
        from unittest.mock import MagicMock

        from nucleusiq.agents.modes.base_mode import BaseExecutionMode

        class ConcreteMode(BaseExecutionMode):
            async def run(self, agent, task):
                pass

        mode = ConcreteMode()
        agent = MagicMock()
        agent.prompt = None
        agent.role = "agent"
        agent.objective = "obj"
        agent.memory = None
        agent._logger = MagicMock()

        call_record = []

        def mock_process(attachments):
            call_record.append(attachments)
            return [{"type": "file", "file": {"filename": "f.pdf", "file_data": "b64"}}]

        agent.llm = MagicMock()
        agent.llm.process_attachments = mock_process

        task = Task(
            id="t-wire",
            objective="Analyze",
            attachments=[
                Attachment(type=AttachmentType.PDF, data=b"%PDF", name="f.pdf"),
            ],
        )
        messages = mode.build_messages(agent, task)
        assert len(call_record) == 1
        user_msg = [m for m in messages if m.role == "user"][-1]
        assert isinstance(user_msg.content, list)
        file_parts = [p for p in user_msg.content if p.get("type") == "file"]
        assert len(file_parts) == 1


# ================================================================== #
# Exhaustiveness: every AttachmentType has a handler                   #
# ================================================================== #


class TestAttachmentExhaustiveness:
    """Ensure that every AttachmentType enum member is handled."""

    def test_all_types_have_framework_handler(self):
        """AttachmentProcessor._HANDLERS must cover every AttachmentType."""
        handler_types = set(AttachmentProcessor._HANDLERS.keys())
        enum_types = set(AttachmentType)
        missing = enum_types - handler_types
        assert not missing, (
            f"AttachmentProcessor is missing handlers for: "
            f"{sorted(m.value for m in missing)}"
        )

    def test_supported_types_returns_all(self):
        """supported_types() must return every enum member."""
        assert AttachmentProcessor.supported_types() == frozenset(AttachmentType)

    def test_every_type_produces_output(self):
        """Processing every type with valid data must produce at least one part."""
        samples = {
            AttachmentType.TEXT: ("hello", "note.txt"),
            AttachmentType.IMAGE_URL: ("https://example.com/img.png", "img.png"),
            AttachmentType.IMAGE_BASE64: (b"\x89PNG\r\n", "img.png"),
            AttachmentType.PDF: (b"%PDF-1.4 fake", "doc.pdf"),
            AttachmentType.FILE_BYTES: (b"raw data", "data.bin"),
            AttachmentType.FILE_BASE64: (
                base64.b64encode(b"hello").decode(),
                "data.txt",
            ),
            AttachmentType.FILE_URL: ("https://example.com/file.csv", "file.csv"),
        }
        assert set(samples.keys()) == set(AttachmentType), (
            "Test samples must cover every AttachmentType"
        )
        for atype, (data, name) in samples.items():
            att = Attachment(type=atype, data=data, name=name)
            parts = AttachmentProcessor.process([att])
            assert len(parts) >= 1, f"{atype.value} produced no content parts"

    def test_unknown_type_raises_value_error(self):
        """Passing an unrecognised type string must raise AttachmentUnsupportedError."""
        import pytest

        att = Attachment(type=AttachmentType.TEXT, data="test")
        att.__dict__["type"] = "nonexistent_type"
        with pytest.raises(
            AttachmentUnsupportedError, match="Unsupported attachment type"
        ):
            AttachmentProcessor._process_one(att)


# ================================================================== #
# Provider capability metadata (BaseLLM contract)                      #
# ================================================================== #


class TestBaseLLMCapabilityMetadata:
    """Verify the attachment capability contract on BaseLLM."""

    def test_base_llm_native_types_empty(self):
        """BaseLLM.NATIVE_ATTACHMENT_TYPES must be an empty frozenset."""
        from nucleusiq.llms.base_llm import BaseLLM

        assert frozenset() == BaseLLM.NATIVE_ATTACHMENT_TYPES

    def test_base_llm_supported_extensions_empty(self):
        """BaseLLM.SUPPORTED_FILE_EXTENSIONS must be an empty frozenset."""
        from nucleusiq.llms.base_llm import BaseLLM

        assert frozenset() == BaseLLM.SUPPORTED_FILE_EXTENSIONS

    def test_mock_llm_inherits_empty_capabilities(self):
        """MockLLM (no override) should inherit empty capabilities."""
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        assert frozenset() == llm.NATIVE_ATTACHMENT_TYPES
        assert frozenset() == llm.SUPPORTED_FILE_EXTENSIONS

    def test_describe_attachment_support_structure(self):
        """describe_attachment_support() returns required keys."""
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        info = llm.describe_attachment_support()

        assert "provider" in info
        assert "native_types" in info
        assert "supported_extensions" in info
        assert "type_details" in info
        assert "notes" in info

        assert info["provider"] == "MockLLM"
        assert info["native_types"] == []
        assert info["supported_extensions"] == []

    def test_describe_covers_all_attachment_types(self):
        """type_details must contain every AttachmentType."""
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        info = llm.describe_attachment_support()
        detail_keys = set(info["type_details"].keys())
        enum_vals = {t.value for t in AttachmentType}
        assert detail_keys == enum_vals

    def test_generic_provider_all_types_framework(self):
        """For a generic provider, all types should be 'framework'."""
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        info = llm.describe_attachment_support()
        for desc in info["type_details"].values():
            assert "framework" in desc


# ================================================================== #
# Validation hardening in process()                                     #
# ================================================================== #


class TestProcessValidation:
    """Verify that process() enforces size, MIME, and large-text checks."""

    def test_process_rejects_oversized_attachment(self):
        import pytest

        big = Attachment(type=AttachmentType.TEXT, data="x" * (MAX_FILE_SIZE_BYTES + 1))
        with pytest.raises(AttachmentValidationError, match="exceeding"):
            AttachmentProcessor.process([big])

    def test_process_passes_under_limit(self):
        att = Attachment(type=AttachmentType.TEXT, data="small", name="f.txt")
        parts = AttachmentProcessor.process([att])
        assert len(parts) == 1
        assert parts[0].type == "text"

    def test_mime_mismatch_warns_but_continues(self, caplog):
        import logging

        att = Attachment(
            type=AttachmentType.PDF, data=b"NOT_A_PDF_HEADER", name="f.pdf"
        )
        with caplog.at_level(logging.WARNING):
            parts = AttachmentProcessor.process([att])
        assert any("magic bytes" in r.message for r in caplog.records)
        assert len(parts) >= 1

    def test_mime_correct_pdf_no_warning(self, caplog):
        import logging

        att = Attachment(type=AttachmentType.PDF, data=b"%PDF-1.4 fake", name="ok.pdf")
        with caplog.at_level(logging.WARNING):
            AttachmentProcessor.process([att])
        mime_warnings = [r for r in caplog.records if "magic bytes" in r.message]
        assert len(mime_warnings) == 0

    def test_large_text_warns(self, caplog):
        import logging

        big_text = "x" * (101 * 1024)
        att = Attachment(type=AttachmentType.TEXT, data=big_text, name="huge.txt")
        with caplog.at_level(logging.WARNING):
            AttachmentProcessor.process([att])
        assert any("FileReadTool" in r.message for r in caplog.records)

    def test_small_text_no_warning(self, caplog):
        import logging

        att = Attachment(type=AttachmentType.TEXT, data="small", name="f.txt")
        with caplog.at_level(logging.WARNING):
            AttachmentProcessor.process([att])
        tool_warnings = [r for r in caplog.records if "FileReadTool" in r.message]
        assert len(tool_warnings) == 0
