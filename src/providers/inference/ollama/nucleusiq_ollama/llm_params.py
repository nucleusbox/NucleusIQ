"""Typed parameters merged into :class:`BaseOllama` calls."""

from __future__ import annotations

from typing import Any

from nucleusiq.llms.llm_params import LLMParams
from pydantic import ConfigDict, Field

from nucleusiq_ollama._shared.wire import ThinkLevel


class OllamaLLMParams(LLMParams):
    """Parameters forwarded to Ollama ``chat`` (beyond base LLM sampling)."""

    model_config = ConfigDict(extra="forbid")

    think: bool | ThinkLevel | None = Field(
        None,
        description="Enable model reasoning stream; maps to Ollama ``think``.",
    )
    keep_alive: float | str | None = Field(
        None,
        description="Ollama model keep-alive duration.",
    )

    def to_call_kwargs(self) -> dict[str, Any]:
        data = self.model_dump()
        return {k: v for k, v in data.items() if v is not None}
