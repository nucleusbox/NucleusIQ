"""Tests for OpenAI native file input support.

Covers:
- BaseOpenAI.process_attachments() override
- File size validation
- SUPPORTED_FILE_EXTENSIONS matching
- messages_to_responses_input() content part conversion
- Integration: Task -> mocked OpenAI call -> native file format
"""

from __future__ import annotations

import base64
from unittest.mock import patch

import pytest
from nucleusiq.agents.attachments import (
    MAX_FILE_SIZE_BYTES,
    Attachment,
    AttachmentType,
)
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.agents.task import Task

# ================================================================== #
# Helpers                                                              #
# ================================================================== #


def _make_openai_llm():
    """Create a BaseOpenAI instance with a fake API key."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-fake"}):
        from nucleusiq_openai.nb_openai.base import BaseOpenAI

        return BaseOpenAI(model_name="gpt-4o", api_key="sk-test-fake")


# ================================================================== #
# BaseOpenAI.process_attachments — PDF                                 #
# ================================================================== #


class TestOpenAIProcessAttachmentsPDF:
    def test_pdf_produces_native_file_part(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.PDF,
            data=b"%PDF-1.4 test content",
            name="report.pdf",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        part = result[0]
        assert part["type"] == "file"
        assert "file" in part
        assert part["file"]["filename"] == "report.pdf"
        assert part["file"]["file_data"].startswith("data:application/pdf;base64,")
        decoded = base64.b64decode(part["file"]["file_data"].split(",", 1)[1])
        assert decoded == b"%PDF-1.4 test content"

    def test_pdf_with_string_data(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.PDF,
            data="pdf-as-string",
            name="doc.pdf",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "file"
        assert result[0]["file"]["filename"] == "doc.pdf"

    def test_pdf_custom_mime(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.PDF,
            data=b"%PDF",
            name="scan.pdf",
            mime_type="application/x-pdf",
        )
        result = llm.process_attachments([att])
        assert "application/x-pdf" in result[0]["file"]["file_data"]


# ================================================================== #
# BaseOpenAI.process_attachments — FILE_BYTES                          #
# ================================================================== #


class TestOpenAIProcessAttachmentsFileBytes:
    def test_supported_extension_produces_native(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.FILE_BYTES,
            data=b"col1,col2\n1,2\n3,4",
            name="data.csv",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "file"
        assert result[0]["file"]["filename"] == "data.csv"
        assert "text/csv" in result[0]["file"]["file_data"]

    def test_unsupported_extension_uses_framework_fallback(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.FILE_BYTES,
            data=b"some binary data",
            name="data.xyz",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "text"

    def test_no_name_uses_framework_fallback(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.FILE_BYTES,
            data=b"unnamed file content",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "text"


# ================================================================== #
# BaseOpenAI.process_attachments — FILE_URL                            #
# ================================================================== #


class TestOpenAIProcessAttachmentsFileURL:
    def test_file_url_produces_input_file(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.FILE_URL,
            data="https://example.com/report.pdf",
            name="report.pdf",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        part = result[0]
        assert part["type"] == "input_file"
        assert part["file_url"] == "https://example.com/report.pdf"

    def test_file_url_bytes_data(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.FILE_URL,
            data=b"https://example.com/data.csv",
        )
        result = llm.process_attachments([att])
        assert result[0]["file_url"] == "https://example.com/data.csv"


# ================================================================== #
# BaseOpenAI.process_attachments — FILE_BASE64                         #
# ================================================================== #


class TestOpenAIProcessAttachmentsFileBase64:
    def test_base64_produces_native_file_no_double_encoding(self):
        """Base64 data should be passed directly, not re-encoded."""
        llm = _make_openai_llm()
        original = b"col1,col2\n1,2\n3,4"
        b64_str = base64.b64encode(original).decode()
        att = Attachment(
            type=AttachmentType.FILE_BASE64,
            data=b64_str,
            name="data.csv",
            mime_type="text/csv",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        part = result[0]
        assert part["type"] == "file"
        assert part["file"]["filename"] == "data.csv"
        expected_data_uri = f"data:text/csv;base64,{b64_str}"
        assert part["file"]["file_data"] == expected_data_uri

        decoded = base64.b64decode(part["file"]["file_data"].split(",", 1)[1])
        assert decoded == original

    def test_base64_mime_from_extension(self):
        llm = _make_openai_llm()
        b64_str = base64.b64encode(b"test").decode()
        att = Attachment(
            type=AttachmentType.FILE_BASE64,
            data=b64_str,
            name="report.pdf",
        )
        result = llm.process_attachments([att])
        assert "application/pdf" in result[0]["file"]["file_data"]

    def test_base64_default_mime(self):
        llm = _make_openai_llm()
        b64_str = base64.b64encode(b"test").decode()
        att = Attachment(
            type=AttachmentType.FILE_BASE64,
            data=b64_str,
            name="unknown_file",
        )
        result = llm.process_attachments([att])
        assert "application/octet-stream" in result[0]["file"]["file_data"]

    def test_base64_bytes_data(self):
        """Data can be bytes that decode to a base64 string."""
        llm = _make_openai_llm()
        original = b"hello"
        b64_bytes = base64.b64encode(original)  # bytes, not str
        att = Attachment(
            type=AttachmentType.FILE_BASE64,
            data=b64_bytes,
            name="hello.txt",
        )
        result = llm.process_attachments([att])
        assert result[0]["type"] == "file"


# ================================================================== #
# BaseOpenAI.process_attachments — TEXT                                #
# ================================================================== #


class TestOpenAIProcessAttachmentsText:
    def test_supported_text_extension_produces_native(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.TEXT,
            data='{"key": "value"}',
            name="config.json",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "file"
        assert result[0]["file"]["filename"] == "config.json"
        assert "application/json" in result[0]["file"]["file_data"]

    def test_unsupported_text_extension_uses_fallback(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.TEXT,
            data="some text",
            name="notes.random",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "text"

    def test_text_no_name_uses_fallback(self):
        llm = _make_openai_llm()
        att = Attachment(type=AttachmentType.TEXT, data="plain text")
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert "plain text" in result[0]["text"]


# ================================================================== #
# BaseOpenAI.process_attachments — images (framework fallback)         #
# ================================================================== #


class TestOpenAIProcessAttachmentsImages:
    def test_image_url_uses_framework_format(self):
        llm = _make_openai_llm()
        att = Attachment(
            type=AttachmentType.IMAGE_URL,
            data="https://img.com/photo.png",
            detail="high",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"] == "https://img.com/photo.png"
        assert result[0]["image_url"]["detail"] == "high"

    def test_image_base64_uses_framework_format(self):
        llm = _make_openai_llm()
        raw_bytes = b"\x89PNG\r\n\x1a\n"
        att = Attachment(
            type=AttachmentType.IMAGE_BASE64,
            data=raw_bytes,
            mime_type="image/png",
        )
        result = llm.process_attachments([att])
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")


# ================================================================== #
# File size validation in OpenAI provider                              #
# ================================================================== #


class TestOpenAIFileSizeValidation:
    def test_oversized_file_raises(self):
        llm = _make_openai_llm()
        big = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        att = Attachment(type=AttachmentType.PDF, data=big, name="huge.pdf")
        with pytest.raises(ValueError, match="exceeding the 50 MB limit"):
            llm.process_attachments([att])


# ================================================================== #
# Extension matching                                                   #
# ================================================================== #


class TestOpenAIExtensionMatching:
    def test_supported_extensions(self):
        llm = _make_openai_llm()
        for ext in [".pdf", ".csv", ".xlsx", ".docx", ".pptx", ".json", ".py"]:
            assert llm._is_supported_extension(f"file{ext}") is True

    def test_unsupported_extensions(self):
        llm = _make_openai_llm()
        for ext in [".xyz", ".mp4", ".wav", ".zip"]:
            assert llm._is_supported_extension(f"file{ext}") is False

    def test_no_name(self):
        llm = _make_openai_llm()
        assert llm._is_supported_extension(None) is False

    def test_no_extension(self):
        llm = _make_openai_llm()
        assert llm._is_supported_extension("README") is False

    def test_case_insensitive(self):
        llm = _make_openai_llm()
        assert llm._is_supported_extension("data.CSV") is True
        assert llm._is_supported_extension("REPORT.PDF") is True


# ================================================================== #
# MIME guessing                                                        #
# ================================================================== #


class TestOpenAIMimeGuessing:
    def test_known_extensions(self):
        llm = _make_openai_llm()
        assert llm._guess_mime("report.pdf") == "application/pdf"
        assert llm._guess_mime("data.csv") == "text/csv"
        assert llm._guess_mime("doc.docx") == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def test_unknown_extension(self):
        llm = _make_openai_llm()
        assert llm._guess_mime("file.xyz") is None

    def test_no_name(self):
        llm = _make_openai_llm()
        assert llm._guess_mime(None) is None


# ================================================================== #
# messages_to_responses_input — content part conversion                #
# ================================================================== #


class TestResponseNormalizerContentConversion:
    def test_text_to_input_text(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this"},
                ],
            }
        ]
        instructions, items = messages_to_responses_input(messages, None)
        assert instructions is None
        assert len(items) == 1
        content = items[0].content
        assert isinstance(content, list)
        assert content[0]["type"] == "input_text"
        assert content[0]["text"] == "Summarize this"

    def test_file_to_input_file(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze"},
                    {
                        "type": "file",
                        "file": {
                            "filename": "report.pdf",
                            "file_data": "data:application/pdf;base64,ABCD",
                        },
                    },
                ],
            }
        ]
        _, items = messages_to_responses_input(messages, None)
        content = items[0].content
        assert len(content) == 2
        file_part = content[1]
        assert file_part["type"] == "input_file"
        assert file_part["filename"] == "report.pdf"
        assert file_part["file_data"] == "data:application/pdf;base64,ABCD"

    def test_input_file_passthrough(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_url": "https://example.com/doc.pdf"},
                ],
            }
        ]
        _, items = messages_to_responses_input(messages, None)
        content = items[0].content
        assert content[0]["type"] == "input_file"
        assert content[0]["file_url"] == "https://example.com/doc.pdf"

    def test_image_url_to_input_image(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://img.com/pic.png",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        _, items = messages_to_responses_input(messages, None)
        content = items[0].content
        assert content[0]["type"] == "input_image"
        assert content[0]["image_url"] == "https://img.com/pic.png"
        assert content[0]["detail"] == "high"

    def test_plain_string_content_unchanged(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        messages = [{"role": "user", "content": "Hello world"}]
        _, items = messages_to_responses_input(messages, None)
        assert items[0].content == "Hello world"

    def test_mixed_content_full_conversion(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these files"},
                    {
                        "type": "file",
                        "file": {
                            "filename": "data.csv",
                            "file_data": "data:text/csv;base64,YQ==",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://img.com/chart.png"},
                    },
                    {"type": "input_file", "file_url": "https://example.com/extra.pdf"},
                ],
            }
        ]
        _, items = messages_to_responses_input(messages, None)
        content = items[0].content
        assert len(content) == 4
        assert content[0]["type"] == "input_text"
        assert content[1]["type"] == "input_file"
        assert content[1]["filename"] == "data.csv"
        assert content[2]["type"] == "input_image"
        assert content[3]["type"] == "input_file"
        assert content[3]["file_url"] == "https://example.com/extra.pdf"

    def test_file_with_file_id(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "file", "file": {"file_id": "file-abc123"}},
                ],
            }
        ]
        _, items = messages_to_responses_input(messages, None)
        content = items[0].content
        assert content[0]["type"] == "input_file"
        assert content[0]["file_id"] == "file-abc123"

    def test_system_message_still_extracted(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze"},
                    {"type": "file", "file": {"filename": "x.pdf", "file_data": "b64"}},
                ],
            },
        ]
        instructions, items = messages_to_responses_input(messages, None)
        assert instructions == "You are a helpful assistant"
        assert len(items) == 1
        assert items[0].content[0]["type"] == "input_text"


# ================================================================== #
# _convert_content_part_for_responses — edge cases                     #
# ================================================================== #


class TestConvertContentPart:
    def test_input_text_passthrough(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            _convert_content_part_for_responses,
        )

        part = {"type": "input_text", "text": "hello"}
        assert _convert_content_part_for_responses(part) is part

    def test_input_image_passthrough(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            _convert_content_part_for_responses,
        )

        part = {"type": "input_image", "image_url": "https://img.com"}
        assert _convert_content_part_for_responses(part) is part

    def test_unknown_type_passthrough(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            _convert_content_part_for_responses,
        )

        part = {"type": "custom_widget", "data": "xyz"}
        assert _convert_content_part_for_responses(part) is part

    def test_image_url_string_value(self):
        from nucleusiq_openai.nb_openai.response_normalizer import (
            _convert_content_part_for_responses,
        )

        part = {"type": "image_url", "image_url": "https://direct-url.com/img.png"}
        result = _convert_content_part_for_responses(part)
        assert result["type"] == "input_image"
        assert result["image_url"] == "https://direct-url.com/img.png"


# ================================================================== #
# Integration: Task -> MessageBuilder -> OpenAI native format          #
# ================================================================== #


class TestEndToEndOpenAIFileInput:
    def test_pdf_task_produces_native_file_in_messages(self):
        llm = _make_openai_llm()
        task = Task(
            id="e2e-pdf",
            objective="Summarize this report",
            attachments=[
                Attachment(
                    type=AttachmentType.PDF,
                    data=b"%PDF-1.4 sample content",
                    name="quarterly.pdf",
                ),
            ],
        )
        msgs = MessageBuilder.build(
            task,
            attachment_processor=llm.process_attachments,
        )
        user_msg = [m for m in msgs if m.role == "user"][-1]
        assert isinstance(user_msg.content, list)

        text_parts = [p for p in user_msg.content if p.get("type") == "text"]
        assert any("Summarize" in p["text"] for p in text_parts)

        file_parts = [p for p in user_msg.content if p.get("type") == "file"]
        assert len(file_parts) == 1
        assert file_parts[0]["file"]["filename"] == "quarterly.pdf"
        assert "application/pdf;base64," in file_parts[0]["file"]["file_data"]

    def test_file_url_task_produces_input_file(self):
        llm = _make_openai_llm()
        task = Task(
            id="e2e-url",
            objective="Analyze this letter",
            attachments=[
                Attachment(
                    type=AttachmentType.FILE_URL,
                    data="https://example.com/letter.pdf",
                    name="letter.pdf",
                ),
            ],
        )
        msgs = MessageBuilder.build(
            task,
            attachment_processor=llm.process_attachments,
        )
        user_msg = [m for m in msgs if m.role == "user"][-1]
        assert isinstance(user_msg.content, list)
        input_file_parts = [
            p for p in user_msg.content if p.get("type") == "input_file"
        ]
        assert len(input_file_parts) == 1
        assert input_file_parts[0]["file_url"] == "https://example.com/letter.pdf"

    def test_mixed_attachments(self):
        llm = _make_openai_llm()
        task = Task(
            id="e2e-mix",
            objective="Analyze all of these",
            attachments=[
                Attachment(
                    type=AttachmentType.PDF,
                    data=b"%PDF",
                    name="report.pdf",
                ),
                Attachment(
                    type=AttachmentType.IMAGE_URL,
                    data="https://img.com/chart.png",
                    detail="high",
                ),
                Attachment(
                    type=AttachmentType.TEXT,
                    data='{"sales": 1000}',
                    name="data.json",
                ),
                Attachment(
                    type=AttachmentType.FILE_URL,
                    data="https://example.com/extra.csv",
                    name="extra.csv",
                ),
            ],
        )
        msgs = MessageBuilder.build(
            task,
            attachment_processor=llm.process_attachments,
        )
        user_msg = [m for m in msgs if m.role == "user"][-1]
        parts = user_msg.content
        assert isinstance(parts, list)
        types = [p["type"] for p in parts]
        assert "text" in types
        assert "file" in types
        assert "image_url" in types
        assert "input_file" in types

    def test_full_pipeline_through_responses_normalizer(self):
        """End-to-end: attachment -> OpenAI processing -> Responses API normalization."""
        from nucleusiq_openai.nb_openai.response_normalizer import (
            messages_to_responses_input,
        )

        llm = _make_openai_llm()
        task = Task(
            id="e2e-full",
            objective="Summarize",
            attachments=[
                Attachment(
                    type=AttachmentType.PDF,
                    data=b"%PDF-1.4",
                    name="doc.pdf",
                ),
            ],
        )
        msgs = MessageBuilder.build(
            task,
            role="analyst",
            attachment_processor=llm.process_attachments,
        )
        msg_dicts = [m.to_dict() for m in msgs]
        instructions, items = messages_to_responses_input(msg_dicts, None)

        assert instructions is not None
        assert "analyst" in instructions

        user_items = [i for i in items if getattr(i, "role", None) == "user"]
        assert len(user_items) >= 1
        last_user = user_items[-1]
        content = last_user.content
        assert isinstance(content, list)

        input_text_parts = [p for p in content if p["type"] == "input_text"]
        assert any("Summarize" in p["text"] for p in input_text_parts)

        input_file_parts = [p for p in content if p["type"] == "input_file"]
        assert len(input_file_parts) == 1
        assert input_file_parts[0]["filename"] == "doc.pdf"
        assert "application/pdf;base64," in input_file_parts[0]["file_data"]


# ================================================================== #
# Exhaustiveness: OpenAI handles every AttachmentType                  #
# ================================================================== #


class TestOpenAIAttachmentExhaustiveness:
    """Ensure BaseOpenAI handles every AttachmentType without crashing."""

    def test_handled_types_cover_all_enum_members(self):
        """_HANDLED_ATTACHMENT_TYPES must equal set(AttachmentType)."""
        from nucleusiq_openai.nb_openai.base import BaseOpenAI

        assert frozenset(AttachmentType) == BaseOpenAI._HANDLED_ATTACHMENT_TYPES

    def test_every_type_is_routed_without_error(self):
        """process_attachments() must not raise for any valid AttachmentType."""
        import base64

        llm = _make_openai_llm()
        samples = {
            AttachmentType.TEXT: ("hello", "note.txt"),
            AttachmentType.IMAGE_URL: ("https://example.com/img.png", "img.png"),
            AttachmentType.IMAGE_BASE64: (b"\x89PNG\r\n", "img.png"),
            AttachmentType.PDF: (b"%PDF-1.4 fake", "doc.pdf"),
            AttachmentType.FILE_BYTES: (b"raw csv data", "data.csv"),
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
            parts = llm.process_attachments([att])
            assert len(parts) >= 1, (
                f"OpenAI process_attachments produced no output for {atype.value}"
            )

    def test_unknown_type_raises_value_error(self):
        """Passing an unrecognised type must raise ValueError, not silent fallback."""
        llm = _make_openai_llm()
        att = Attachment(type=AttachmentType.TEXT, data="test")
        att.__dict__["type"] = "nonexistent_type"
        with pytest.raises(ValueError, match="Unhandled attachment type"):
            llm.process_attachments([att])


# ================================================================== #
# Provider capability metadata (BaseOpenAI declarations)               #
# ================================================================== #


class TestOpenAICapabilityMetadata:
    """Verify BaseOpenAI declares attachment capabilities correctly."""

    def test_native_attachment_types_declared(self):
        """BaseOpenAI.NATIVE_ATTACHMENT_TYPES must be a non-empty frozenset."""
        from nucleusiq_openai.nb_openai.base import BaseOpenAI

        assert isinstance(BaseOpenAI.NATIVE_ATTACHMENT_TYPES, frozenset)
        assert len(BaseOpenAI.NATIVE_ATTACHMENT_TYPES) > 0

    def test_native_attachment_types_expected_members(self):
        """OpenAI should declare PDF, FILE_BASE64, FILE_URL, TEXT, FILE_BYTES."""
        from nucleusiq_openai.nb_openai.base import BaseOpenAI

        expected = {
            AttachmentType.PDF,
            AttachmentType.FILE_BASE64,
            AttachmentType.FILE_URL,
            AttachmentType.TEXT,
            AttachmentType.FILE_BYTES,
        }
        assert expected == BaseOpenAI.NATIVE_ATTACHMENT_TYPES

    def test_images_not_in_native_types(self):
        """IMAGE_URL and IMAGE_BASE64 are standard, not native file types."""
        from nucleusiq_openai.nb_openai.base import BaseOpenAI

        assert AttachmentType.IMAGE_URL not in BaseOpenAI.NATIVE_ATTACHMENT_TYPES
        assert AttachmentType.IMAGE_BASE64 not in BaseOpenAI.NATIVE_ATTACHMENT_TYPES

    def test_supported_file_extensions_non_empty(self):
        """BaseOpenAI.SUPPORTED_FILE_EXTENSIONS must contain known extensions."""
        from nucleusiq_openai.nb_openai.base import BaseOpenAI

        exts = BaseOpenAI.SUPPORTED_FILE_EXTENSIONS
        assert isinstance(exts, frozenset)
        assert ".pdf" in exts
        assert ".csv" in exts
        assert ".py" in exts
        assert ".xlsx" in exts

    def test_instance_inherits_capabilities(self):
        """An instance should expose the same class-level capabilities."""
        llm = _make_openai_llm()
        assert llm.NATIVE_ATTACHMENT_TYPES == type(llm).NATIVE_ATTACHMENT_TYPES
        assert llm.SUPPORTED_FILE_EXTENSIONS == type(llm).SUPPORTED_FILE_EXTENSIONS

    def test_describe_attachment_support_structure(self):
        """describe_attachment_support() must return required keys."""
        llm = _make_openai_llm()
        info = llm.describe_attachment_support()

        assert info["provider"] == "BaseOpenAI"
        assert "native_types" in info
        assert "supported_extensions" in info
        assert "type_details" in info
        assert "notes" in info

    def test_describe_covers_all_types(self):
        """type_details must contain every AttachmentType value."""
        llm = _make_openai_llm()
        info = llm.describe_attachment_support()
        detail_keys = set(info["type_details"].keys())
        enum_vals = {t.value for t in AttachmentType}
        assert detail_keys == enum_vals

    def test_describe_native_types_listed(self):
        """native_types list should contain the 5 expected types."""
        llm = _make_openai_llm()
        info = llm.describe_attachment_support()
        assert "pdf" in info["native_types"]
        assert "file_base64" in info["native_types"]
        assert "file_url" in info["native_types"]

    def test_describe_extensions_match_class(self):
        """supported_extensions should match SUPPORTED_FILE_EXTENSIONS."""
        llm = _make_openai_llm()
        info = llm.describe_attachment_support()
        assert set(info["supported_extensions"]) == llm.SUPPORTED_FILE_EXTENSIONS

    def test_describe_type_details_categories(self):
        """PDF should be 'native', IMAGE_URL should be 'standard'."""
        llm = _make_openai_llm()
        info = llm.describe_attachment_support()
        assert "native" in info["type_details"]["pdf"]
        assert "standard" in info["type_details"]["image_url"]
        assert "conditional" in info["type_details"]["text"]
