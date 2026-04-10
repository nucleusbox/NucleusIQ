"""ObservationMasker — Tier 0 post-response strategy.

After the model produces a response, tool results that have been
"consumed" (the model saw them and responded) are replaced with
slim markers.  Full content is preserved in ``ContentStore``.

This is NOT part of the ``CompactionPipeline`` — it runs
unconditionally after every LLM response via ``engine.post_response()``.

Responsibility (SRP): identify consumed tool-result messages and
replace their content with a compact marker.  Does NOT decide
*whether* to run — that is ``ContextEngine``'s decision.

Research backing:
    - Morph Research (Feb 2026): Claude Code uses 5.5x fewer tokens
      than Cursor.  Primary mechanism: stripping consumed tool outputs.
    - 80% of context rot comes from stale tool results that the model
      has already incorporated into its reasoning.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nucleusiq.agents.chat_models import ChatMessage
    from nucleusiq.agents.context.counter import TokenCounter
    from nucleusiq.agents.context.store import ContentStore

_MASKED_MARKER = "[observation consumed — {tokens} tokens offloaded | ref: {key}]"


class ObservationMasker:
    """Replaces consumed tool results with slim markers.

    A tool result is "consumed" when it appears before the most recent
    assistant message — the model has already seen and responded to it.

    Design:
        - Stateless — receives messages, returns modified copy.
        - Does NOT mutate input list (returns a new list).
        - Stores full content in ``ContentStore`` for potential rehydration.
    """

    __slots__ = ()

    def mask(
        self,
        messages: list[ChatMessage],
        token_counter: TokenCounter,
        store: ContentStore,
    ) -> tuple[list[ChatMessage], int, int]:
        """Mask consumed tool results in the message list.

        A tool result at index *i* is "consumed" if there exists an
        assistant message at index *j > i*.  Tool results after the
        last assistant message are NOT masked (the model hasn't
        responded to them yet).

        Args:
            messages: Current conversation messages.
            token_counter: For counting tokens freed.
            store: Where to offload full content.

        Returns:
            Tuple of (new_messages, observations_masked, tokens_freed).
        """
        from nucleusiq.agents.chat_models import ChatMessage as CM

        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx < 0:
            return list(messages), 0, 0

        result: list[ChatMessage] = []
        masked_count = 0
        total_freed = 0

        for i, msg in enumerate(messages):
            if (
                i < last_assistant_idx
                and msg.role == "tool"
                and isinstance(msg.content, str)
                and not msg.content.startswith("[observation consumed")
                and not msg.content.startswith("[context_ref:")
            ):
                original_tokens = token_counter.count(msg.content)
                if original_tokens < 20:
                    result.append(msg)
                    continue

                key = f"obs:{msg.name or 'tool'}:{uuid.uuid4().hex[:8]}"
                store.store(
                    key=key,
                    content=msg.content,
                    original_tokens=original_tokens,
                )

                marker = _MASKED_MARKER.format(tokens=original_tokens, key=key)
                marker_tokens = token_counter.count(marker)
                freed = max(0, original_tokens - marker_tokens)

                result.append(
                    CM(
                        role=msg.role,
                        content=marker,
                        name=msg.name,
                        tool_call_id=msg.tool_call_id,
                    )
                )
                masked_count += 1
                total_freed += freed
            else:
                result.append(msg)

        return result, masked_count, total_freed
