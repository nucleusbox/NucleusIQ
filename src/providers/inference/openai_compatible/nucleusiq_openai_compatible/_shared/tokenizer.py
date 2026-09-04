"""Token counting strategies.

Self-hosted open-weight models use tokenizer families unrelated to OpenAI's,
so ``tiktoken`` would produce systematically wrong counts — worst on code and
non-English text, which is exactly where a context budget matters.

Two strategies behind a one-method Protocol:

* :class:`HFTokenizerCounter` — exact counts from the model's own tokenizer,
  when the caller declares one and the ``[tokenizer]`` extra is installed.
* :class:`HeuristicCounter` — the framework-wide ``len(text) // 4``
  approximation, matching ``BaseLLM.estimate_tokens``.

Which one is in use is recorded on every call record, so telemetry can tell
measured counts from estimated ones.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

__all__ = [
    "HFTokenizerCounter",
    "HeuristicCounter",
    "TokenCounter",
    "build_token_counter",
    "tokenizer_backend_available",
]

_logger = logging.getLogger(__name__)


@runtime_checkable
class TokenCounter(Protocol):
    """Strategy interface for counting tokens in a string."""

    def count(self, text: str) -> int:
        """Return the number of tokens in *text* (always at least 1)."""
        ...

    @property
    def method(self) -> str:
        """Either ``"tokenizer"`` or ``"heuristic"``, for telemetry."""
        ...


def tokenizer_backend_available() -> bool:
    """Whether the optional ``tokenizers`` backend is importable."""
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


class HeuristicCounter:
    """Approximate counts at ~4 characters per token.

    Identical to ``BaseLLM.estimate_tokens`` so budgets computed here agree
    with the framework default when no tokenizer is declared.
    """

    __slots__ = ()

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)

    @property
    def method(self) -> str:
        return "heuristic"

    def __repr__(self) -> str:
        return "HeuristicCounter()"


class HFTokenizerCounter:
    """Exact counts from a Hugging Face tokenizer.

    Args:
        tokenizer: A repo id (``"google/gemma-4-27b-it"``) or a path to a
            local ``tokenizer.json``.

    Raises:
        ImportError: The ``[tokenizer]`` extra is not installed.
        ValueError: The tokenizer could not be loaded, with the underlying
            reason attached — a silent fallback to the heuristic here would
            corrupt budgets invisibly.
    """

    __slots__ = ("_encode", "_name")

    def __init__(self, tokenizer: str) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "tokenizer= requires the optional dependency. Install with "
                "`pip install 'nucleusiq-openai-compatible[tokenizer]'`, or "
                "omit tokenizer= to use the ~4 chars/token heuristic."
            ) from exc

        self._name = tokenizer
        loader = (
            Tokenizer.from_file
            if tokenizer.endswith(".json")
            else Tokenizer.from_pretrained
        )
        try:
            tok = loader(tokenizer)
        except Exception as exc:
            raise ValueError(
                f"Could not load tokenizer {tokenizer!r}: {exc}. Pass a valid "
                "Hugging Face repo id or a path to tokenizer.json, or omit "
                "tokenizer= to use the heuristic counter."
            ) from exc
        self._encode = tok.encode

    def count(self, text: str) -> int:
        return max(1, len(self._encode(text).ids))

    @property
    def method(self) -> str:
        return "tokenizer"

    def __repr__(self) -> str:
        return f"HFTokenizerCounter(tokenizer={self._name!r})"


def build_token_counter(tokenizer: str | None) -> TokenCounter:
    """Return the appropriate counter for the declared *tokenizer*."""
    if tokenizer:
        return HFTokenizerCounter(tokenizer)
    return HeuristicCounter()
