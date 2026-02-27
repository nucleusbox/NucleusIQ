# File: src/nucleusiq/core/llms/base_llm.py
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from nucleusiq.llms.base import BaseLanguageModel
from nucleusiq.streaming.events import StreamEvent


class BaseLLM(BaseLanguageModel, ABC):
    """Abstract base class for Language Model adapters.

    Subclasses **must** implement:

    * ``call()`` — non-streaming request/response.
    * ``call_stream()`` *(optional)* — real streaming; the default
      fallback calls ``call()`` and yields a single ``COMPLETE`` event.

    The streaming contract uses ``StreamEvent`` so that every provider
    emits the same event type and consumer code stays provider-agnostic.
    """

    def convert_tool_specs(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert ``BaseTool`` instances to LLM-specific tool format.

        Override in providers to customise conversion.
        """
        converted = []
        for tool in tools:
            if hasattr(tool, "get_spec"):
                spec = tool.get_spec()
                converted.append(self._convert_tool_spec(spec))
            else:
                converted.append(tool)
        return converted

    def _convert_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Convert a generic tool spec to LLM-specific format.

        Default implementation returns *spec* unchanged.
        """
        return spec

    # ------------------------------------------------------------------ #
    # Non-streaming call                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_tokens: int = 150,
        temperature: float = 0.5,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Send messages to the model and return a complete response.

        Provider-specific parameters (from ``LLMParams`` subclasses) are
        forwarded via ``**kwargs``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Streaming call                                                      #
    # ------------------------------------------------------------------ #

    async def call_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_tokens: int = 150,
        temperature: float = 0.5,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream LLM output as ``StreamEvent`` objects.

        **Default behaviour (non-streaming fallback):** delegates to
        ``call()`` and yields a single ``StreamEvent.complete_event``.

        Providers that support real token-by-token streaming should
        override this method and yield ``StreamEvent.token_event()``
        for each delta, followed by ``StreamEvent.complete_event()``
        with the full accumulated text.
        """
        response = await self.call(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs,
        )
        content = self._extract_content_from_response(response)
        if content:
            yield StreamEvent.complete_event(content)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_content_from_response(response: Any) -> str | None:
        """Best-effort extraction of text content from a provider response.

        Handles both dict-style and attribute-style response objects.
        """
        if response is None:
            return None

        choices = getattr(response, "choices", None)
        if not choices:
            return None

        msg = getattr(choices[0], "message", None)
        if msg is None:
            return None

        if isinstance(msg, dict):
            return msg.get("content")
        return getattr(msg, "content", None)
