"""Optional merged kwargs for Claude (beta headers, top-k)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnthropicLLMParams:
    """Fine-grained Anthropic knobs merged into ``BaseAnthropic.call`` / streaming."""

    top_k: int | None = None
    anthropic_beta: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    def to_call_kwargs(self) -> dict[str, Any]:
        """Return kwargs merged upstream of ``build_create_kwargs``.

        Uses private keys prefixed with underscores so Anthropic rejects no
        parameters — :class:`~nucleusiq_anthropic.nb_anthropic.base.BaseAnthropic`
        strips ``_merged_extra_headers``.
        """

        merged: dict[str, Any] = {}
        if self.top_k is not None:
            merged["top_k"] = self.top_k
        beta = self.anthropic_beta
        if isinstance(beta, str) and beta.strip():
            merged["anthropic_beta"] = beta.strip()
        if self.extra_headers:
            merged["_merged_extra_headers"] = dict(self.extra_headers)

        return merged
