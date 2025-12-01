"""
OpenAI provider for NucleusIQ.

This module provides OpenAI client that supports both async and sync modes.
The mode is determined by the `async_mode` parameter during initialization.
"""

from __future__ import annotations

import os
import asyncio
import time
import logging
import httpx
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import openai
from nucleusiq.llms.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class _Choice(BaseModel):
    """Minimal wrapper so we match BaseLLM expectation."""
    message: Dict[str, Any]


class _LLMResponse(BaseModel):
    choices: List[_Choice]


class BaseOpenAI(BaseLLM):
    """
    OpenAI client for ChatCompletion with tool-calling support.
    
    Supports both async and sync modes based on `async_mode` parameter.
    Default is async mode (True).
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.7,
        logit_bias: Optional[Dict[str, float]] = None,
        async_mode: bool = True,
    ) -> None:
        """
        Initialize OpenAI chat client with sensible defaults.
        
        Args:
            async_mode: If True, uses async client. If False, uses sync client.
                       Default is True (async).
        
        Values can be overridden via arguments or environment variables:
          - OPENAI_API_KEY
          - OPENAI_API_BASE
          - OPENAI_ORG_ID
        """
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.logit_bias = logit_bias
        self.async_mode = async_mode
        self._logger = logging.getLogger("BaseOpenAI")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")

        # Create appropriate client based on mode
        if self.async_mode:
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        else:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

    # ---------- public helpers ----------
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string."""
        import tiktoken
        enc = tiktoken.encoding_for_model(self.model_name)
        return len(enc.encode(text))

    # ---------- BaseLLM impl ----------
    async def call(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 256,
        temperature: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Any:
        """
        Call OpenAI API.
        
        If async_mode=True, this is an async method.
        If async_mode=False, this method runs sync code but is still async-compatible.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        
        # Only include optional parameters if they are provided
        if self.logit_bias is not None:
            payload["logit_bias"] = self.logit_bias
        if tools is not None:
            payload["tools"] = tools
        if stop is not None:
            payload["stop"] = stop

        attempt = 0
        while True:
            try:
                if self.async_mode:
                    # Async mode
                    if stream:
                        async for chunk in self._client.chat.completions.create(**payload):
                            first = chunk.choices[0].delta
                            return _LLMResponse(
                                choices=[_Choice(message=first.model_dump())]
                            )
                    else:
                        resp = await self._client.chat.completions.create(**payload)
                        return _LLMResponse(
                            choices=[_Choice(message=resp.choices[0].message.model_dump())]
                        )
                else:
                    # Sync mode - run in executor to make it async-compatible
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if stream:
                        # For sync streaming, we need to handle it differently
                        chunks = []
                        for chunk in self._client.chat.completions.create(**payload):
                            chunks.append(chunk)
                        if chunks:
                            first = chunks[0].choices[0].delta
                            return _LLMResponse(
                                choices=[_Choice(message=first.model_dump())]
                            )
                    else:
                        resp = await loop.run_in_executor(
                            None,
                            lambda: self._client.chat.completions.create(**payload)
                        )
                        return _LLMResponse(
                            choices=[_Choice(message=resp.choices[0].message.model_dump())]
                        )
            except openai.RateLimitError as e:
                # Rate limit errors - retry with exponential backoff
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error(
                        f"Rate limit exceeded after {self.max_retries} retries: {e}"
                    )
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"Rate limit hit ({e}); retry {attempt}/{self.max_retries} in {backoff}s"
                )
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except openai.APIError as e:
                # API errors (500, 502, 503, etc.) - retry with exponential backoff
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error(
                        f"API error after {self.max_retries} retries: {e}"
                    )
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"API error ({e}); retry {attempt}/{self.max_retries} in {backoff}s"
                )
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except openai.APIConnectionError as e:
                # Connection errors - retry with exponential backoff
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error(
                        f"Connection error after {self.max_retries} retries: {e}"
                    )
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"Connection error ({e}); retry {attempt}/{self.max_retries} in {backoff}s"
                )
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except httpx.HTTPError as e:
                # HTTP errors (timeouts, network issues) - retry with exponential backoff
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error(
                        f"HTTP error after {self.max_retries} retries: {e}"
                    )
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"HTTP error ({e}); retry {attempt}/{self.max_retries} in {backoff}s"
                )
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except openai.AuthenticationError as e:
                # Authentication errors - don't retry, fail immediately
                self._logger.error(f"Authentication failed: {e}")
                raise ValueError(f"Invalid API key or authentication failed: {e}") from e
            except openai.PermissionError as e:
                # Permission errors - don't retry, fail immediately
                self._logger.error(f"Permission denied: {e}")
                raise ValueError(f"Permission denied: {e}") from e
            except openai.InvalidRequestError as e:
                # Invalid request errors - don't retry, fail immediately
                self._logger.error(f"Invalid request: {e}")
                raise ValueError(f"Invalid request parameters: {e}") from e
            except Exception as e:
                # Unexpected errors - log and re-raise
                self._logger.error(f"Unexpected error during OpenAI call: {e}", exc_info=True)
                raise
