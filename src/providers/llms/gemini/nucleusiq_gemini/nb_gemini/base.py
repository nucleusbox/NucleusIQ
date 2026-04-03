"""Gemini provider for NucleusIQ.

This module provides the Gemini LLM client that implements the ``BaseLLM``
contract using Google's ``google-genai`` SDK.

**Delegation-based design (not a god class)**:

``BaseGemini`` is the orchestrator that wires together specialized
collaborators — each with a single responsibility:

- ``GeminiClient``       — SDK communication (send/receive)
- ``response_normalizer`` — raw SDK response → ``GeminiLLMResponse``
- ``tool_converter``      — ``BaseTool`` specs → Gemini function declarations
- ``stream_adapters``     — streaming chunks → ``StreamEvent`` objects
- ``model_config``        — model capabilities and metadata lookups

Structured Output Support::

    response = await llm.call(
        model="gemini-2.5-flash",
        messages=[...],
        response_format=MyPydanticModel,
    )
"""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.streaming.events import StreamEvent
from pydantic import BaseModel

from nucleusiq_gemini._shared.model_config import (
    get_max_output_tokens,
    supports_thinking,
)
from nucleusiq_gemini._shared.models import GenerationConfig, ThinkingConfig
from nucleusiq_gemini._shared.response_models import (
    AssistantMessage,
)
from nucleusiq_gemini.nb_gemini.client import GeminiClient
from nucleusiq_gemini.nb_gemini.response_normalizer import (
    messages_to_gemini_contents,
    normalize_response,
)
from nucleusiq_gemini.nb_gemini.stream_adapters import stream_gemini
from nucleusiq_gemini.structured_output import (
    build_gemini_response_format,
    parse_gemini_response,
)
from nucleusiq_gemini.tools.gemini_tool import NATIVE_TOOL_TYPES
from nucleusiq_gemini.tools.tool_converter import build_tools_payload

logger = logging.getLogger(__name__)

__all__ = ["BaseGemini"]


class BaseGemini(BaseLLM):
    """Google Gemini client implementing the ``BaseLLM`` contract.

    Delegates to specialized collaborators for each concern:
    - Client wrapper for SDK communication
    - Response normalizer for output normalization
    - Tool converter for function declaration formatting
    - Stream adapter for streaming event conversion

    **Open/Closed Principle** — the public ``call()`` interface is stable;
    new features (thinking, code execution) are absorbed by collaborators.
    """

    NATIVE_TOOL_TYPES: frozenset[str] = NATIVE_TOOL_TYPES

    NATIVE_ATTACHMENT_TYPES: frozenset = frozenset(
        {
            "image_url",
            "image_base64",
            "pdf",
            "file_bytes",
            "file_base64",
            "text",
        }
    )

    SUPPORTED_FILE_EXTENSIONS: frozenset[str] = frozenset(
        {
            ".pdf",
            ".txt",
            ".md",
            ".json",
            ".html",
            ".htm",
            ".xml",
            ".css",
            ".js",
            ".py",
            ".csv",
            ".tsv",
        }
    )

    _MIME_MAP: dict[str, str] = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".html": "text/html",
        ".htm": "text/html",
        ".xml": "text/xml",
        ".css": "text/css",
        ".js": "application/javascript",
        ".py": "text/x-python",
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
    }

    # ================================================================== #
    # __init__                                                            #
    # ================================================================== #

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.7,
        top_k: int | None = None,
    ) -> None:
        """Initialize Gemini client with sensible defaults.

        Args:
            model_name: Model identifier (e.g. ``"gemini-2.5-flash"``).
            api_key: Gemini API key (falls back to ``GEMINI_API_KEY`` env var).
            temperature: Default sampling temperature.
            top_k: Default Top-K sampling value.
        """
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self.top_k = top_k
        self._logger = logging.getLogger("BaseGemini")

        if not self.api_key:
            from nucleusiq.llms.errors import AuthenticationError

            raise AuthenticationError(
                "GEMINI_API_KEY is required. Set it as an environment variable "
                "or pass it to the constructor."
            )

        self._client = GeminiClient(api_key=self.api_key)

    # ================================================================== #
    # Tool spec conversion                                                #
    # ================================================================== #

    def _convert_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Convert generic BaseTool spec to Gemini function declaration format.

        Delegates to ``tool_converter.convert_tool_spec()`` (SRP).
        """
        from nucleusiq_gemini.tools.tool_converter import convert_tool_spec

        return convert_tool_spec(spec)

    # ================================================================== #
    # Attachment processing (Gemini-native multimodal)                     #
    # ================================================================== #

    def process_attachments(
        self,
        attachments: list,
    ) -> list[dict[str, Any]]:
        """Convert attachments to Gemini-native inline_data content parts.

        Gemini natively processes images, audio, video, and documents via
        inline_data (base64) or file_data (URI). This method produces
        Gemini-specific content parts for multimodal input.
        """
        from nucleusiq.agents.attachments import AttachmentType

        parts: list[dict[str, Any]] = []
        for att in attachments:
            att_type = att.type

            if att_type in (AttachmentType.IMAGE_URL, AttachmentType.IMAGE_BASE64):
                parts.extend(self._process_image_attachment(att))
            elif att_type == AttachmentType.PDF:
                parts.extend(self._process_file_attachment(att, "application/pdf"))
            elif att_type == AttachmentType.FILE_BASE64:
                parts.extend(self._process_base64_attachment(att))
            elif att_type in (AttachmentType.FILE_BYTES, AttachmentType.TEXT):
                parts.extend(self._process_file_attachment(att))
            elif att_type == AttachmentType.FILE_URL:
                parts.extend(self._framework_fallback([att]))
            else:
                parts.extend(self._framework_fallback([att]))

        return parts

    def _process_image_attachment(self, att: Any) -> list[dict[str, Any]]:
        """Process image attachments to Gemini inline_data format."""
        from nucleusiq.agents.attachments import AttachmentType

        if att.type == AttachmentType.IMAGE_BASE64:
            data = att.data if isinstance(att.data, str) else att.data.decode()
            mime = att.mime_type or self._guess_mime(att.name) or "image/png"
            return [{"inline_data": {"mime_type": mime, "data": data}}]

        url = att.data if isinstance(att.data, str) else att.data.decode()
        if url.startswith("data:"):
            mime, _, b64 = url.partition(";base64,")
            mime = mime.replace("data:", "")
            return [{"inline_data": {"mime_type": mime, "data": b64}}]

        return [{"text": f"[Image: {url}]"}]

    def _process_file_attachment(
        self, att: Any, fallback_mime: str | None = None
    ) -> list[dict[str, Any]]:
        """Process file attachments to Gemini inline_data format."""
        raw = att.data if isinstance(att.data, bytes) else att.data.encode("utf-8")
        b64 = base64.b64encode(raw).decode()
        mime = (
            att.mime_type
            or self._guess_mime(att.name)
            or fallback_mime
            or "application/octet-stream"
        )
        return [{"inline_data": {"mime_type": mime, "data": b64}}]

    def _process_base64_attachment(self, att: Any) -> list[dict[str, Any]]:
        """Process pre-encoded base64 attachments."""
        b64_str = att.data if isinstance(att.data, str) else att.data.decode()
        mime = att.mime_type or self._guess_mime(att.name) or "application/octet-stream"
        return [{"inline_data": {"mime_type": mime, "data": b64_str}}]

    def _framework_fallback(self, attachments: list) -> list[dict[str, Any]]:
        """Delegate to the framework's default attachment processing."""
        return super().process_attachments(attachments)

    def _guess_mime(self, name: str | None) -> str | None:
        if not name:
            return None
        dot = name.rfind(".")
        if dot < 0:
            return None
        return self._MIME_MAP.get(name[dot:].lower())

    # ================================================================== #
    # call() — non-streaming entry point                                   #
    # ================================================================== #

    async def call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        response_format: type[BaseModel] | type | dict[str, Any] | None = None,
        seed: int | None = None,
        top_k: int | None = None,
        thinking_config: dict[str, Any] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        candidate_count: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call Gemini API with optional structured output.

        Converts standard BaseLLM messages to Gemini format, builds
        generation config, and delegates to the client wrapper.

        ``max_output_tokens`` maps directly to Gemini's native
        ``max_output_tokens`` generation config parameter.
        """
        output_schema_type = None
        gen_config_extras: dict[str, Any] = {}

        if response_format is not None:
            if tools:
                logger.warning(
                    "Gemini API does not support structured JSON output "
                    "(response_mime_type='application/json') combined with "
                    "function calling in the same request. Dropping "
                    "response_format to avoid a 400 error. Use prompt-based "
                    "JSON extraction instead when tools are enabled."
                )
            elif isinstance(response_format, tuple) and len(response_format) == 2:
                provider_format, raw_schema = response_format
                if isinstance(provider_format, dict):
                    gen_config_extras.update(provider_format)
                if isinstance(raw_schema, type):
                    output_schema_type = raw_schema
            elif isinstance(response_format, dict):
                gen_config_extras.update(response_format)
            else:
                fmt = build_gemini_response_format(response_format)
                if fmt:
                    gen_config_extras.update(fmt)
                    if isinstance(response_format, type):
                        output_schema_type = response_format

        system_instruction, contents = messages_to_gemini_contents(messages)

        config = self._build_generation_config(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed,
            top_k=top_k,
            thinking_config=thinking_config,
            candidate_count=candidate_count,
            model=model,
            **gen_config_extras,
        )

        api_config = config.to_api_dict()
        if system_instruction:
            api_config["system_instruction"] = system_instruction
        if safety_settings:
            api_config["safety_settings"] = safety_settings

        if tools:
            tool_specs = self._resolve_tool_specs(tools)
            tools_payload = build_tools_payload(tool_specs)
            api_config["tools"] = tools_payload

        if tool_choice is not None:
            tool_config = self._build_tool_config(tool_choice)
            if tool_config:
                api_config["tool_config"] = tool_config

        raw_response = await self._client.generate_content(
            model=model,
            contents=contents,
            config=api_config,
        )

        result = normalize_response(raw_response)

        if output_schema_type is not None and result.choices:
            msg = result.choices[0].message
            if isinstance(msg, AssistantMessage) and msg.content:
                return parse_gemini_response(msg.to_dict(), output_schema_type)

        return result

    # ================================================================== #
    # call_stream() — streaming entry point                                #
    # ================================================================== #

    async def call_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream Gemini output as ``StreamEvent`` objects.

        Yields:
            ``StreamEvent`` — ``TOKEN`` events for each text delta,
            ``THINKING`` events for reasoning, then one ``COMPLETE``
            event with accumulated content and metadata.
        """
        system_instruction, contents = messages_to_gemini_contents(messages)

        top_k = kwargs.pop("top_k", None)
        thinking_config = kwargs.pop("thinking_config", None)
        safety_settings = kwargs.pop("safety_settings", None)
        candidate_count = kwargs.pop("candidate_count", None)
        seed = kwargs.pop("seed", None)

        config = self._build_generation_config(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed,
            top_k=top_k,
            thinking_config=thinking_config,
            candidate_count=candidate_count,
            model=model,
        )

        api_config = config.to_api_dict()
        if system_instruction:
            api_config["system_instruction"] = system_instruction
        if safety_settings:
            api_config["safety_settings"] = safety_settings

        if tools:
            tool_specs = self._resolve_tool_specs(tools)
            tools_payload = build_tools_payload(tool_specs)
            api_config["tools"] = tools_payload

        if tool_choice is not None:
            tool_config = self._build_tool_config(tool_choice)
            if tool_config:
                api_config["tool_config"] = tool_config

        sync_stream = await self._client.generate_content_stream(
            model=model,
            contents=contents,
            config=api_config,
        )

        async for event in stream_gemini(sync_stream):
            yield event

    # ================================================================== #
    # Private helpers                                                      #
    # ================================================================== #

    @staticmethod
    def _resolve_tool_specs(tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tool objects to spec dicts if needed.

        Handles both raw dicts and BaseTool/GeminiNativeTool objects
        by calling ``get_spec()`` when available.
        """
        resolved = []
        for tool in tools:
            if isinstance(tool, dict):
                resolved.append(tool)
            elif hasattr(tool, "get_spec"):
                resolved.append(tool.get_spec())
            else:
                resolved.append(tool)
        return resolved

    def _build_generation_config(
        self,
        *,
        max_output_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        top_k: int | None = None,
        thinking_config: dict[str, Any] | None = None,
        candidate_count: int | None = None,
        model: str = "",
        **extras: Any,
    ) -> GenerationConfig:
        """Build a ``GenerationConfig`` from call parameters."""
        effective_temp = temperature if temperature is not None else self.temperature
        effective_top_k = top_k if top_k is not None else self.top_k

        model_max = get_max_output_tokens(model)
        effective_max = (
            min(max_output_tokens, model_max) if max_output_tokens else model_max
        )

        tc = None
        if thinking_config and supports_thinking(model):
            budget = thinking_config.get("thinking_budget", 0)
            tc = ThinkingConfig(thinking_budget=budget)

        return GenerationConfig(
            temperature=effective_temp,
            top_p=top_p,
            top_k=effective_top_k,
            max_output_tokens=effective_max,
            stop_sequences=stop,
            seed=seed,
            candidate_count=candidate_count,
            frequency_penalty=frequency_penalty if frequency_penalty != 0.0 else None,
            presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
            thinking_config=tc,
            **{k: v for k, v in extras.items() if v is not None},
        )

    @staticmethod
    def _build_tool_config(tool_choice: Any) -> dict[str, Any] | None:
        """Convert BaseLLM ``tool_choice`` to Gemini ``tool_config``."""
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            mode_map = {
                "auto": "AUTO",
                "none": "NONE",
                "required": "ANY",
                "any": "ANY",
            }
            mode = mode_map.get(tool_choice.lower())
            if mode:
                return {"function_calling_config": {"mode": mode}}

        if isinstance(tool_choice, dict):
            fn = tool_choice.get("function", {})
            fn_name = fn.get("name") if isinstance(fn, dict) else None
            if fn_name:
                return {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [fn_name],
                    }
                }

        return None

    # ================================================================== #
    # Public helpers                                                      #
    # ================================================================== #

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.

        Uses a simple heuristic (4 chars per token) since Gemini does
        not expose a public tokenizer. For precise counts, use the
        ``count_tokens`` API via ``self._client.raw_client``.
        """
        return max(1, len(text) // 4)
