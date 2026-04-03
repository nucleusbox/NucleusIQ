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

    **Attachment capability contract** (mirrors ``NATIVE_TOOL_TYPES``):

    * ``NATIVE_ATTACHMENT_TYPES`` — ``frozenset`` of ``AttachmentType``
      members that this provider processes **natively** (server-side).
      Empty by default; providers override to declare capabilities.
    * ``SUPPORTED_FILE_EXTENSIONS`` — ``frozenset`` of file extensions
      (e.g. ``".pdf"``, ``".csv"``) that this provider handles natively
      for ``FILE_BYTES`` / ``TEXT`` routing.  Empty by default.
    * ``describe_attachment_support()`` — user-facing introspection
      returning a structured summary of what this provider does with
      each attachment type.
    """

    NATIVE_ATTACHMENT_TYPES: frozenset = frozenset()
    """Attachment types this provider handles natively (server-side).
    Override in provider subclass to declare capabilities."""

    SUPPORTED_FILE_EXTENSIONS: frozenset[str] = frozenset()
    """File extensions this provider processes natively for
    ``FILE_BYTES`` and ``TEXT`` attachments.  Override in provider."""

    def describe_attachment_support(self) -> dict[str, Any]:
        """Return a structured summary of how this provider handles attachments.

        Users can call ``agent.llm.describe_attachment_support()`` to
        understand what happens with each ``AttachmentType`` for their
        chosen provider.

        Returns
        -------
        dict
            Keys: ``provider``, ``native_types``, ``supported_extensions``,
            ``type_details`` (per-type description), ``notes``.
        """
        from nucleusiq.agents.attachments import AttachmentType

        all_types = sorted(AttachmentType, key=lambda t: t.value)
        native_vals = {t.value for t in self.NATIVE_ATTACHMENT_TYPES}
        type_details: dict[str, str] = {}
        for t in all_types:
            if t.value in native_vals:
                type_details[t.value] = "native (provider processes server-side)"
            else:
                type_details[t.value] = "framework (client-side text extraction)"

        return {
            "provider": type(self).__name__,
            "native_types": sorted(native_vals),
            "supported_extensions": sorted(self.SUPPORTED_FILE_EXTENSIONS),
            "type_details": type_details,
            "notes": (
                "All AttachmentType values are supported. "
                "Types not in native_types use framework-level "
                "client-side processing (text extraction, base64 encoding)."
            ),
        }

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
    # Attachment processing                                                #
    # ------------------------------------------------------------------ #

    def process_attachments(
        self,
        attachments: list,
    ) -> list[dict[str, Any]]:
        """Convert ``Attachment`` objects into provider-ready content-part dicts.

        The default implementation delegates to the framework-level
        ``AttachmentProcessor`` which extracts text client-side.
        Providers that support native file input (e.g. OpenAI's
        ``input_file`` / ``file`` content parts) should override this
        method to produce their API-native format so the model receives
        raw files for server-side processing.

        Parameters
        ----------
        attachments : list[Attachment]
            Attachment objects from ``Task.attachments``.

        Returns
        -------
        list[dict]
            Content-part dicts ready for inclusion in a multimodal
            message (e.g. ``{"type": "text", "text": "..."}``,
            ``{"type": "image_url", "image_url": {...}}``).
        """
        from nucleusiq.agents.attachments import AttachmentProcessor, ContentPart

        parts: list[dict[str, Any]] = []
        processed: list[ContentPart] = AttachmentProcessor.process(attachments)
        for cp in processed:
            if cp.type == "text" and cp.text is not None:
                parts.append({"type": "text", "text": cp.text})
            elif cp.type == "image_url" and cp.image_url is not None:
                parts.append({"type": "image_url", "image_url": cp.image_url})
            elif cp.metadata is not None:
                parts.append({"type": cp.type, **cp.metadata})
        return parts

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
        max_output_tokens: int = 150,
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

        ``max_output_tokens`` is the provider-neutral name for the
        maximum number of tokens to generate.  Each provider translates
        this to the API-specific parameter name internally.

        Raises:
            nucleusiq.llms.errors.AuthenticationError: Invalid API key (401).
            nucleusiq.llms.errors.PermissionDeniedError: Access denied (403).
            nucleusiq.llms.errors.RateLimitError: Too many requests (429),
                after exhausting retry attempts.
            nucleusiq.llms.errors.InvalidRequestError: Bad parameters (400).
            nucleusiq.llms.errors.ModelNotFoundError: Model not found (404).
            nucleusiq.llms.errors.ContentFilterError: Content blocked by safety.
            nucleusiq.llms.errors.ProviderServerError: Provider 5xx, after retries.
            nucleusiq.llms.errors.ProviderConnectionError: Network failure, after retries.
            nucleusiq.llms.errors.ProviderError: Other provider-specific error.
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
        max_output_tokens: int = 150,
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
            max_output_tokens=max_output_tokens,
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
    # Token estimation                                                     #
    # ------------------------------------------------------------------ #

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in *text*.

        The default uses a ~4 chars/token heuristic that works across
        providers with zero dependencies.  Provider subclasses should
        override with a precise tokenizer (e.g. ``tiktoken`` for OpenAI,
        Gemini ``count_tokens`` API).

        This method is designed to be passed as a callback::

            ContextWindowPlugin(max_tokens=8000, token_counter=llm.estimate_tokens)
            TokenBudgetMemory(max_tokens=4096, token_counter=llm.estimate_tokens)
        """
        return max(1, len(text) // 4)

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
