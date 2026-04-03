"""Base language models class."""

from __future__ import annotations

from typing import Any

from pydantic import Field


class BaseLanguageModel:
    """
    Base class for all language models in NucleusIQ.

    Provides shared fields inherited by ``BaseLLM``.
    Not intended to be instantiated directly — use ``BaseLLM`` subclasses.
    """

    metadata: dict[str, Any] | None = Field(default=None, exclude=True)
