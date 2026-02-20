"""
SummaryWindowMemory — recent messages verbatim + older ones summarized.

Best for: balancing detail (recent) with long-term context (summary).
Trade-off: 1 LLM call when window overflows; moderate token usage.

When the window fills up, the oldest messages are summarized into a
running summary, and the window is trimmed.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import Field, ConfigDict

from nucleusiq.memory.base import BaseMemory

if TYPE_CHECKING:
    from nucleusiq.llms.base_llm import BaseLLM


_SUMMARIZE_PROMPT = (
    "Summarize the following conversation messages into a concise "
    "paragraph.  Return ONLY the summary.\n\n"
    "Previous summary:\n{prev_summary}\n\n"
    "Messages to incorporate:\n{messages}\n\n"
    "Updated summary:"
)


class SummaryWindowMemory(BaseMemory):
    """Hybrid: recent window of verbatim messages + running summary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window_size: int = Field(
        default=10,
        gt=0,
        description="Number of recent messages kept verbatim.",
    )
    llm: Optional[Any] = Field(
        default=None,
        description="BaseLLM instance used for summarization.",
    )
    llm_model: str = Field(
        default="default",
        description="Model name passed to llm.call().",
    )
    summary_max_tokens: int = Field(
        default=512,
        description="Max tokens for the summarization call.",
    )

    _messages: deque = deque()
    _summary: str = ""

    def model_post_init(self, __context: Any) -> None:
        self._messages: deque = deque()
        self._summary: str = ""

    @property
    def strategy_name(self) -> str:
        return "summary_window"

    # -- Sync core -------------------------------------------------------

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.window_size:
            self._compact_sync()

    def get_context(
        self, query: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, str]]:
        ctx: List[Dict[str, str]] = []
        if self._summary:
            ctx.append({"role": "system", "content": self._summary})
        ctx.extend(self._messages)
        return ctx

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""

    def export_state(self) -> Dict[str, Any]:
        return {
            "messages": list(self._messages),
            "summary": self._summary,
            "window_size": self.window_size,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        self._messages = deque(state.get("messages", []))
        self._summary = state.get("summary", "")

    # -- Async (preferred path) ------------------------------------------

    async def aadd_message(
        self, role: str, content: str, **kwargs: Any
    ) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.window_size:
            await self._compact_async()

    # -- Compaction ------------------------------------------------------

    def _compact_sync(self) -> None:
        overflow = self._drain_overflow()
        if not overflow:
            return
        if not self.llm:
            self._summary += "\n" + self._format_messages(overflow)
            return
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            self._summary += "\n" + self._format_messages(overflow)
        else:
            self._summary = asyncio.run(self._summarize(overflow))

    async def _compact_async(self) -> None:
        overflow = self._drain_overflow()
        if not overflow:
            return
        if not self.llm:
            self._summary += "\n" + self._format_messages(overflow)
            return
        self._summary = await self._summarize(overflow)

    def _drain_overflow(self) -> List[Dict[str, str]]:
        overflow: List[Dict[str, str]] = []
        while len(self._messages) > self.window_size:
            overflow.append(self._messages.popleft())
        return overflow

    async def _summarize(
        self, messages: List[Dict[str, str]]
    ) -> str:
        prompt = _SUMMARIZE_PROMPT.format(
            prev_summary=self._summary or "(none)",
            messages=self._format_messages(messages),
        )
        response = await self.llm.call(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.summary_max_tokens,
        )
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
            return getattr(msg, "content", "") or ""
        return str(response)

    @staticmethod
    def _format_messages(msgs: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"[{m.get('role', '?')}]: {m.get('content', '')}"
            for m in msgs
        )
