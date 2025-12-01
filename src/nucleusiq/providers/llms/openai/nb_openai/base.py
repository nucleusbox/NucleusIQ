#base.py
from __future__ import annotations

import os, asyncio, json, tiktoken, time, logging, httpx
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import openai                           # ⬅ requires openai>=1.14
from nucleusiq.llms.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class _Choice(BaseModel):
    """Minimal wrapper so we match BaseLLM expectation."""
    message: Dict[str, Any]


class _LLMResponse(BaseModel):
    choices: List[_Choice]


class BaseOpenAI(BaseLLM):
    """Slim async client for OpenAI ChatCompletion with tool-calling support."""

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
    ) -> None:
        """
        Initialize an async OpenAI chat client with sensible defaults.
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
        self._logger = logging.getLogger("BaseOpenAI")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")

        self._client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    # ---------- public helpers ----------
    def estimate_tokens(self, text: str) -> int:
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
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "logit_bias": self.logit_bias,
            "tools": tools,
            "stream": stream,
            "stop": stop,
        }

        attempt = 0
        while True:
            try:
                if stream:
                    # Return first chunk immediately as a faux-stream for simplicity
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
            except (openai.RateLimitError, httpx.HTTPError) as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"OpenAI call failed ({e}); retry {attempt}/{self.max_retries} in {backoff}s"
                )
                await asyncio.sleep(backoff)