"""
Attachment model and processor for multimodal agent inputs.

Attachments let users pass files, images, and other media alongside
a Task.  The framework converts them into provider-compatible content
arrays so the same agent code works regardless of the underlying LLM.

Supported types (v0.4.0):
    image_url       — remote image for vision
    image_base64    — local image encoded as base64
    text            — inline text (.txt, .md, .csv, .json, .yaml, .xml, .html)
    pdf             — PDF document (text extraction, optional pdfplumber)
    file_bytes      — raw bytes, provider decides how to handle
    file_base64     — pre-encoded base64 file data (no double-encoding)
    file_url        — remote file URL (provider-native, e.g. OpenAI Responses API)
"""

from __future__ import annotations

import base64
import binascii
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB (OpenAI limit)


class AttachmentType(str, Enum):
    """Supported attachment media types.

    Every member **must** have a corresponding handler in
    ``AttachmentProcessor._HANDLERS`` (framework-level) and be routed in
    each provider's ``process_attachments()`` override.

    ┌───────────────┬──────────────────┬────────────────────────────────┐
    │ Type          │ data format      │ Description                    │
    ├───────────────┼──────────────────┼────────────────────────────────┤
    │ TEXT          │ str or bytes     │ Plain text (.txt .md .csv etc) │
    │ PDF           │ bytes (or str)   │ PDF document                   │
    │ FILE_BYTES    │ bytes            │ Raw file bytes (any format)    │
    │ FILE_BASE64   │ str (base64)     │ Pre-encoded base64 file data   │
    │ FILE_URL      │ str (URL)        │ Remote file URL (server-side)  │
    │ IMAGE_URL     │ str (URL)        │ Remote image URL (vision)      │
    │ IMAGE_BASE64  │ bytes or str     │ Local image as base64 / bytes  │
    └───────────────┴──────────────────┴────────────────────────────────┘

    Two categories:

    **Framework-level** (works with any LLM provider):
        ``TEXT``, ``PDF``, ``FILE_BYTES``, ``FILE_BASE64``,
        ``IMAGE_URL``, ``IMAGE_BASE64``

    **Provider-optimised** (best with providers that support native files):
        ``FILE_URL`` — remote file reference, provider fetches server-side.

    All types use the same ``Attachment`` API.  When a provider (e.g. OpenAI)
    supports native file input, it overrides ``process_attachments()`` and
    sends raw file data for server-side processing automatically.
    """

    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"
    TEXT = "text"
    PDF = "pdf"
    FILE_BYTES = "file_bytes"
    FILE_BASE64 = "file_base64"
    FILE_URL = "file_url"


class Attachment(BaseModel):
    """A file or media item attached to a Task.

    Parameters
    ----------
    type : AttachmentType
        How the data should be interpreted.  See ``AttachmentType`` for
        the full table of types, expected ``data`` formats, and behaviour.
    data : str | bytes
        The payload.  Format depends on ``type``:

        - ``TEXT``          — Python ``str`` (or UTF-8 ``bytes``).
        - ``PDF``           — raw PDF ``bytes`` (e.g. ``open("f.pdf","rb").read()``).
        - ``FILE_BYTES``    — raw ``bytes`` of any file.
        - ``FILE_BASE64``   — ``str`` containing base64-encoded file data.
        - ``FILE_URL``      — ``str`` containing a remote file URL.
        - ``IMAGE_URL``     — ``str`` containing a remote image URL.
        - ``IMAGE_BASE64``  — raw image ``bytes`` **or** a base64 ``str``.
    name : str | None
        Human-readable filename (used for labelling and for provider
        extension-based routing — e.g. ``"data.csv"``).
    mime_type : str | None
        MIME type hint (e.g. ``"image/png"``, ``"application/pdf"``).
        When omitted, the framework / provider infers from ``name``.
    detail : str | None
        Vision detail level (``"high"`` / ``"low"`` / ``"auto"``).
        Only meaningful for ``IMAGE_URL`` and ``IMAGE_BASE64``.
    """

    type: AttachmentType
    data: str | bytes
    name: str | None = Field(default=None, description="Filename for context")
    mime_type: str | None = Field(default=None, description="MIME type hint")
    detail: str | None = Field(
        default=None,
        description="Vision detail level (high / low / auto)",
    )


class ContentPart(BaseModel):
    """A single element inside a multimodal content array.

    Provider adapters convert these into the wire format their API expects.
    """

    type: str
    text: str | None = None
    image_url: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class AttachmentProcessor:
    """Converts ``Attachment`` objects into generic ``ContentPart`` lists.

    This is a **framework-level** utility.  It does not know about any
    provider — it produces ``ContentPart`` objects that each provider's
    ``process_attachments()`` can further transform or replace.

    **Exhaustiveness guarantee:** ``_HANDLERS`` must contain an entry for
    every ``AttachmentType`` member.  This is validated at class definition
    time — adding a new enum value without a handler will raise
    ``AssertionError`` on import.
    """

    @staticmethod
    def validate_size(att: Attachment, *, limit: int = MAX_FILE_SIZE_BYTES) -> None:
        """Raise ``AttachmentValidationError`` if *att* exceeds the file-size limit."""
        from nucleusiq.agents.errors import AttachmentValidationError

        size = len(att.data) if isinstance(att.data, (bytes, str)) else 0
        if size > limit:
            mb = limit / (1024 * 1024)
            raise AttachmentValidationError(
                f"Attachment '{att.name or '(unnamed)'}' is {size:,} bytes, "
                f"exceeding the {mb:.0f} MB limit",
                attachment_type=att.type.value,
                file_name=att.name,
            )

    @staticmethod
    def supported_types() -> frozenset[AttachmentType]:
        """Return the set of ``AttachmentType``s with registered handlers."""
        return frozenset(AttachmentProcessor._HANDLERS.keys())

    @staticmethod
    def process(attachments: list[Attachment]) -> list[ContentPart]:
        """Convert a list of attachments into content parts.

        Before conversion each attachment is validated:
        1. Size check — enforces ``MAX_FILE_SIZE_BYTES``.
        2. MIME magic-bytes check — warns (does not block) on mismatch.
        3. Large text warning — suggests ``FileReadTool`` for text > 100 KB.

        Returns an ordered list.  Text-based attachments become ``text``
        parts; image attachments become ``image_url`` parts.

        Raises ``ValueError`` if an attachment exceeds size limits or
        has an unrecognised type.
        """
        parts: list[ContentPart] = []
        for att in attachments:
            AttachmentProcessor.validate_size(att)
            AttachmentProcessor._check_mime_magic(att)
            AttachmentProcessor._warn_large_text(att)
            parts.extend(AttachmentProcessor._process_one(att))
        return parts

    @staticmethod
    def _process_one(att: Attachment) -> list[ContentPart]:
        from nucleusiq.agents.errors import AttachmentUnsupportedError

        handler = AttachmentProcessor._HANDLERS.get(att.type)
        if handler is None:
            raise AttachmentUnsupportedError(
                f"Unsupported attachment type '{att.type}'. "
                f"Supported types: {sorted(t.value for t in AttachmentProcessor._HANDLERS)}",
                attachment_type=str(att.type),
                file_name=att.name,
            )
        return handler(att)

    # ------------------------------------------------------------------ #
    # Per-type handlers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _text(att: Attachment) -> list[ContentPart]:
        text = (
            att.data
            if isinstance(att.data, str)
            else att.data.decode("utf-8", errors="replace")
        )
        label = f"[File: {att.name}]\n" if att.name else ""
        return [ContentPart(type="text", text=f"{label}{text}")]

    @staticmethod
    def _image_url(att: Attachment) -> list[ContentPart]:
        url = att.data if isinstance(att.data, str) else att.data.decode()
        img: dict[str, Any] = {"url": url}
        if att.detail:
            img["detail"] = att.detail
        return [ContentPart(type="image_url", image_url=img)]

    @staticmethod
    def _image_base64(att: Attachment) -> list[ContentPart]:
        if isinstance(att.data, bytes):
            b64 = base64.b64encode(att.data).decode()
        else:
            b64 = att.data
        mime = att.mime_type or "image/png"
        data_uri = f"data:{mime};base64,{b64}"
        img: dict[str, Any] = {"url": data_uri}
        if att.detail:
            img["detail"] = att.detail
        return [ContentPart(type="image_url", image_url=img)]

    @staticmethod
    def _pdf(att: Attachment) -> list[ContentPart]:
        """Extract text from a PDF.

        Uses ``pdfplumber`` if available; otherwise falls back to a
        placeholder warning so the framework doesn't hard-depend on it.
        """
        try:
            import pdfplumber  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "pdfplumber is not installed — PDF text extraction unavailable. "
                "Install with: pip install pdfplumber"
            )
            label = f"[PDF: {att.name}]" if att.name else "[PDF attachment]"
            return [
                ContentPart(
                    type="text",
                    text=f"{label}\n(PDF text extraction requires pdfplumber)",
                )
            ]

        import io

        raw = att.data if isinstance(att.data, bytes) else att.data.encode("latin-1")
        pages_text: list[str] = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"--- Page {i} ---\n{text}")

        label = f"[PDF: {att.name}]\n" if att.name else ""
        combined = "\n\n".join(pages_text) if pages_text else "(empty PDF)"
        return [ContentPart(type="text", text=f"{label}{combined}")]

    @staticmethod
    def _file_bytes(att: Attachment) -> list[ContentPart]:
        """Raw bytes — attempt UTF-8 decode; fall back to base64."""
        if isinstance(att.data, bytes):
            try:
                text = att.data.decode("utf-8")
                label = f"[File: {att.name}]\n" if att.name else ""
                return [ContentPart(type="text", text=f"{label}{text}")]
            except UnicodeDecodeError:
                b64 = base64.b64encode(att.data).decode()
                return [
                    ContentPart(
                        type="text",
                        text=f"[Binary file: {att.name or 'attachment'}] (base64-encoded)\n{b64}",
                    )
                ]
        label = f"[File: {att.name}]\n" if att.name else ""
        return [ContentPart(type="text", text=f"{label}{att.data}")]

    @staticmethod
    def _file_base64(att: Attachment) -> list[ContentPart]:
        """Pre-encoded base64 file data.

        The framework decodes the base64 payload and attempts text extraction.
        Providers that support native file input (e.g. OpenAI) will use the
        base64 data directly without decoding — no double-encoding.
        """
        b64_str = att.data if isinstance(att.data, str) else att.data.decode()
        try:
            raw = base64.b64decode(b64_str)
        except (binascii.Error, ValueError) as exc:
            from nucleusiq.agents.errors import AttachmentProcessingError

            label = att.name or "attachment"
            proc_err = AttachmentProcessingError(
                f"Invalid base64 data for attachment {label!r}: {exc}",
                file_name=att.name,
                attachment_type=AttachmentType.FILE_BASE64.value,
            )
            logger.warning("%s: %s", proc_err.__class__.__name__, proc_err)
            return [
                ContentPart(
                    type="text",
                    text=f"[File: {label}] (invalid base64 data)",
                )
            ]

        try:
            text = raw.decode("utf-8")
            label = f"[File: {att.name}]\n" if att.name else ""
            return [ContentPart(type="text", text=f"{label}{text}")]
        except UnicodeDecodeError:
            label = att.name or "attachment"
            return [
                ContentPart(
                    type="text",
                    text=f"[Binary file: {label}] ({len(raw):,} bytes, base64-encoded)",
                )
            ]

    @staticmethod
    def _file_url(att: Attachment) -> list[ContentPart]:
        """Remote file URL — framework fallback is a text note.

        Providers that support native file-URL input (e.g. OpenAI
        Responses API) override ``BaseLLM.process_attachments()`` and
        never reach this handler.
        """
        url = att.data if isinstance(att.data, str) else att.data.decode()
        label = att.name or url
        return [
            ContentPart(
                type="text",
                text=(
                    f"[File URL: {label}]\n"
                    f"URL: {url}\n"
                    "(File URL attachments require a provider that supports "
                    "native file-URL input. The file content was not fetched.)"
                ),
            )
        ]

    # ------------------------------------------------------------------ #
    # Validation helpers                                                   #
    # ------------------------------------------------------------------ #

    _MIME_SIGNATURES: dict[AttachmentType, tuple[bytes, ...]] = {
        AttachmentType.PDF: (b"%PDF",),
        AttachmentType.IMAGE_BASE64: (b"\x89PNG", b"\xff\xd8"),
    }

    _LARGE_TEXT_THRESHOLD = 100 * 1024  # 100 KB

    @staticmethod
    def _check_mime_magic(att: Attachment) -> None:
        """Warn (not block) if the leading bytes don't match the declared type."""
        sigs = AttachmentProcessor._MIME_SIGNATURES.get(att.type)
        if sigs is None:
            return
        raw: bytes | None = None
        if isinstance(att.data, bytes):
            raw = att.data[:8]
        elif isinstance(att.data, str) and att.type == AttachmentType.PDF:
            raw = att.data[:8].encode("latin-1", errors="ignore")
        if raw is None:
            return
        if not any(raw.startswith(sig) for sig in sigs):
            logger.warning(
                "Attachment '%s' declared as %s but magic bytes don't match "
                "(got %r). Processing will continue.",
                att.name or "(unnamed)",
                att.type.value,
                raw[:4],
            )

    @staticmethod
    def _warn_large_text(att: Attachment) -> None:
        """Log a guidance message when a text attachment exceeds 100 KB."""
        if att.type not in (AttachmentType.TEXT, AttachmentType.FILE_BYTES):
            return
        size = len(att.data) if isinstance(att.data, (bytes, str)) else 0
        if size > AttachmentProcessor._LARGE_TEXT_THRESHOLD:
            logger.warning(
                "Attachment '%s' is %d KB of text. Consider using "
                "FileReadTool with line ranges for large files.",
                att.name or "(unnamed)",
                size // 1024,
            )

    # ------------------------------------------------------------------ #
    # Handler registry (exhaustiveness-checked at import time)            #
    # ------------------------------------------------------------------ #

    _HANDLERS: dict[
        AttachmentType,
        staticmethod[..., list[ContentPart]],
    ] = {
        AttachmentType.TEXT: _text.__func__,  # type: ignore[attr-defined]
        AttachmentType.IMAGE_URL: _image_url.__func__,  # type: ignore[attr-defined]
        AttachmentType.IMAGE_BASE64: _image_base64.__func__,  # type: ignore[attr-defined]
        AttachmentType.PDF: _pdf.__func__,  # type: ignore[attr-defined]
        AttachmentType.FILE_BYTES: _file_bytes.__func__,  # type: ignore[attr-defined]
        AttachmentType.FILE_BASE64: _file_base64.__func__,  # type: ignore[attr-defined]
        AttachmentType.FILE_URL: _file_url.__func__,  # type: ignore[attr-defined]
    }


# ====================================================================== #
# Import-time exhaustiveness check                                         #
# ====================================================================== #

_missing = set(AttachmentType) - set(AttachmentProcessor._HANDLERS)
if _missing:
    raise AssertionError(
        f"AttachmentProcessor is missing handlers for: "
        f"{sorted(m.value for m in _missing)}. "
        f"Every AttachmentType MUST have a corresponding handler."
    )
