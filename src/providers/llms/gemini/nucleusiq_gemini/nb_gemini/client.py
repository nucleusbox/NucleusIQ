"""Thin client wrapper around ``google.genai.Client``.

**Single Responsibility**: Only handles SDK communication — no response
normalization, no tool conversion, no streaming adaptation.

The client exposes two methods:
- ``generate_content()`` — non-streaming request/response (with retry)
- ``generate_content_stream()`` — streaming (returns iterator, no retry)

Both accept pre-built kwargs dicts so the caller (``BaseGemini``) controls
payload construction.
"""

from __future__ import annotations

import logging
from typing import Any

from nucleusiq_gemini._shared.retry import call_with_retry

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper around ``google.genai.Client`` for Gemini API communication.

    Isolates SDK dependency behind a clean interface. Callers interact
    with plain dicts and this class handles SDK object construction.

    Non-streaming calls use ``call_with_retry`` for automatic retry on
    rate limits, server errors, and connection failures. Streaming calls
    are not retried (mid-stream retries are not safe).
    """

    def __init__(
        self,
        *,
        api_key: str,
        max_retries: int = 3,
    ) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._max_retries = max_retries

    async def generate_content(
        self,
        *,
        model: str,
        contents: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> Any:
        """Send a non-streaming generate_content request with retry.

        Args:
            model: Gemini model identifier.
            contents: Conversation contents (messages).
            config: Generation config dict.

        Returns:
            Raw Gemini SDK ``GenerateContentResponse``.

        Raises:
            ValueError: For auth/permission/bad-request errors.
            google.genai.errors.ServerError: After exhausting retries.
            google.genai.errors.ClientError: After exhausting retries on 429.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "contents": contents,
        }
        if config:
            kwargs["config"] = self._build_config(config)

        def _api_call() -> Any:
            return self._client.models.generate_content(**kwargs)

        return await call_with_retry(
            _api_call,
            max_retries=self._max_retries,
            logger=logger,
        )

    async def generate_content_stream(
        self,
        *,
        model: str,
        contents: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> Any:
        """Send a streaming generate_content request.

        Streaming calls are **not retried** — mid-stream retries are not
        safe. Errors during streaming are caught by ``stream_adapters``
        and converted to ``StreamEvent.error_event()``.

        Args:
            model: Gemini model identifier.
            contents: Conversation contents (messages).
            config: Generation config dict.

        Returns:
            Sync iterator of Gemini SDK response chunks.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "contents": contents,
        }
        if config:
            kwargs["config"] = self._build_config(config)

        return self._client.models.generate_content_stream(**kwargs)

    @property
    def raw_client(self) -> Any:
        """Access the underlying ``genai.Client`` for advanced use cases."""
        return self._client

    @staticmethod
    def _build_config(config: dict[str, Any]) -> Any:
        """Convert a config dict to ``GenerateContentConfig``.

        Handles native Gemini tools (google_search, code_execution,
        url_context, google_maps) by converting them into proper SDK
        ``types.Tool`` objects, since the SDK Pydantic model rejects
        raw dicts for these.
        """
        from google.genai import types

        config = dict(config)
        raw_tools = config.pop("tools", None)

        if raw_tools:
            sdk_tools = []
            for tool_dict in raw_tools:
                sdk_tools.append(GeminiClient._convert_tool_dict(tool_dict))
            config["tools"] = sdk_tools

        return types.GenerateContentConfig(**config)

    @staticmethod
    def _convert_tool_dict(tool_dict: dict[str, Any]) -> Any:
        """Convert a tool dict into the appropriate SDK type.

        Function declarations pass through as dicts (the SDK handles
        those). Native tools are converted to ``types.Tool(...)`` with
        the proper nested type object.
        """
        from google.genai import types

        if "function_declarations" in tool_dict:
            return tool_dict

        tool_type = tool_dict.get("type", "")

        if tool_type == "google_search":
            gs_config = tool_dict.get("google_search", {})
            return types.Tool(google_search=types.GoogleSearch(**gs_config))

        if tool_type == "code_execution":
            return types.Tool(code_execution=types.ToolCodeExecution())

        if tool_type == "url_context":
            return types.Tool(url_context=types.UrlContext())

        if tool_type == "google_maps":
            return types.Tool(google_maps=types.GoogleMaps())

        return tool_dict
