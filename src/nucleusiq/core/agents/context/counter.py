"""TokenCounter — provider-agnostic token estimation protocol.

The core framework depends on the ``TokenCounter`` protocol, not on
any specific tokenizer library.  Providers inject precise implementations:

- OpenAI → ``TiktokenCounter`` (uses ``tiktoken``)
- Gemini → heuristic (``len(text) // 4``, avoids API-call cost)
- Default → ``DefaultTokenCounter`` (same heuristic, zero dependencies)

Usage::

    counter: TokenCounter = DefaultTokenCounter()
    tokens = counter.count("Hello, world!")
    total = counter.count_messages(messages)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nucleusiq.agents.chat_models import ChatMessage


@runtime_checkable
class TokenCounter(Protocol):
    """Minimal interface for token estimation.

    Two methods — one for raw text, one for message lists.
    Providers implement with their native tokenizer.
    """

    def count(self, text: str) -> int:
        """Estimate tokens in a text string."""
        ...

    def count_messages(self, messages: list[ChatMessage]) -> int:
        """Estimate total tokens across a list of chat messages."""
        ...


class DefaultTokenCounter:
    """Heuristic token counter — ``len(text) // 4``.

    Matches the existing ``BaseLLM.estimate_tokens()`` behaviour.
    Zero external dependencies.  Suitable for providers without a
    public tokenizer and as a fallback.
    """

    def count(self, text: str) -> int:
        """~4 characters per token heuristic."""
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[ChatMessage]) -> int:
        """Sum token estimates for all messages.

        Accounts for role overhead (~4 tokens per message framing).
        """
        total = 0
        for msg in messages:
            total += 4  # role/name framing overhead
            content = msg.content
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text", "")
                        if text:
                            total += self.count(text)
            if msg.name:
                total += self.count(msg.name)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += self.count(str(tc))
        return total
