"""
SummaryMemory — condenses conversation into a running summary.

Best for: long-running agents that need full context in few tokens.
Trade-off: 1 LLM call per message; loses verbatim detail.

Requires an LLM (``BaseLLM``) for summarization.  If no LLM is
provided, falls back to keeping the last raw message as the summary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import Field, ConfigDict

from nucleusiq.memory.base import BaseMemory

if TYPE_CHECKING:
    from nucleusiq.llms.base_llm import BaseLLM


_SUMMARIZE_PROMPT = (
    "Progressively summarize the conversation so far, incorporating "
    "the new message.  Return ONLY the updated summary, nothing else.\n\n"
    "Current summary:\n{summary}\n\n"
    "New message ({role}):\n{content}\n\n"
    "Updated summary:"
)


class SummaryMemory(BaseMemory):
    """Maintains a running LLM-generated summary of the conversation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    _summary: str = ""
    _message_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        self._summary = ""
        self._message_count = 0

    @property
    def strategy_name(self) -> str:
        return "summary"

    # -- Sync core -------------------------------------------------------

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self._message_count += 1
        if not self.llm:
            self._summary = f"[{role}]: {content}"
            return
        prompt = _SUMMARIZE_PROMPT.format(
            summary=self._summary or "(empty)",
            role=role,
            content=content,
        )
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            self._summary = f"[{role}]: {content}"
        else:
            response = asyncio.run(
                self.llm.call(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.summary_max_tokens,
                )
            )
            self._summary = self._extract_text(response)

    def get_context(
        self, query: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, str]]:
        if not self._summary:
            return []
        return [{"role": "system", "content": self._summary}]

    def clear(self) -> None:
        self._summary = ""
        self._message_count = 0

    def export_state(self) -> Dict[str, Any]:
        return {
            "summary": self._summary,
            "message_count": self._message_count,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        self._summary = state.get("summary", "")
        self._message_count = state.get("message_count", 0)

    # -- Async (preferred path — agent always calls this) ----------------

    async def aadd_message(
        self, role: str, content: str, **kwargs: Any
    ) -> None:
        self._message_count += 1
        if not self.llm:
            self._summary = f"[{role}]: {content}"
            return
        prompt = _SUMMARIZE_PROMPT.format(
            summary=self._summary or "(empty)",
            role=role,
            content=content,
        )
        response = await self.llm.call(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.summary_max_tokens,
        )
        self._summary = self._extract_text(response)

    # -- Helpers ---------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
            return getattr(msg, "content", "") or ""
        return str(response)
