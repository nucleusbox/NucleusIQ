"""Agent-facing tools for the run-local evidence dossier."""

from __future__ import annotations

from typing import Any

from nucleusiq.agents.context.evidence import (
    EvidenceItem,
    InMemoryEvidenceDossier,
)
from nucleusiq.agents.context.policy import ContextPolicy
from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.decorators import DecoratedTool

ADD_EVIDENCE_TOOL_NAME = "add_evidence"
ADD_EVIDENCE_GAP_TOOL_NAME = "add_evidence_gap"
LIST_EVIDENCE_TOOL_NAME = "list_evidence"
SUMMARIZE_EVIDENCE_TOOL_NAME = "summarize_evidence"
EVIDENCE_COVERAGE_TOOL_NAME = "evidence_coverage"

_EVIDENCE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        ADD_EVIDENCE_TOOL_NAME,
        ADD_EVIDENCE_GAP_TOOL_NAME,
        LIST_EVIDENCE_TOOL_NAME,
        SUMMARIZE_EVIDENCE_TOOL_NAME,
        EVIDENCE_COVERAGE_TOOL_NAME,
    }
)

__all__ = [
    "ADD_EVIDENCE_TOOL_NAME",
    "ADD_EVIDENCE_GAP_TOOL_NAME",
    "LIST_EVIDENCE_TOOL_NAME",
    "SUMMARIZE_EVIDENCE_TOOL_NAME",
    "EVIDENCE_COVERAGE_TOOL_NAME",
    "build_evidence_tools",
    "is_evidence_tool_name",
]


def is_evidence_tool_name(tool_name: str | None) -> bool:
    """Return True when ``tool_name`` is an auto-injected evidence tool."""
    return tool_name in _EVIDENCE_TOOL_NAMES if tool_name else False


def _format_evidence_error(message: str) -> str:
    return f"[evidence_error: {message}]"


def _normalize_tags(tags: list | tuple | str | None) -> tuple[str, ...]:
    if tags is None:
        return ()
    if isinstance(tags, str):
        return (tags,)
    if isinstance(tags, (list, tuple)):
        return tuple(str(tag) for tag in tags if str(tag).strip())
    return ()


def _normalize_metadata(metadata: dict | None) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return {str(key): value for key, value in metadata.items()}
    return {}


def _item_to_dict(item: EvidenceItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "claim": item.claim,
        "status": item.status,
        "confidence": item.confidence,
        "tags": list(item.tags),
        "quote": item.quote,
        "source": {
            "ref": item.source.ref,
            "title": item.source.title,
            "locator": item.source.locator,
            "tool_name": item.source.tool_name,
            "metadata": dict(item.source.metadata),
        },
        "metadata": dict(item.metadata),
    }


def build_evidence_tools(dossier: InMemoryEvidenceDossier) -> list[BaseTool]:
    """Build evidence tools bound to one run-local dossier."""

    async def add_evidence(
        claim: str,
        source_ref: str,
        title: str | None = None,
        locator: str | None = None,
        tool_name: str | None = None,
        confidence: float = 1.0,
        tags: list | None = None,
        quote: str | None = None,
        status: str = "supported",
        metadata: dict | None = None,
    ) -> dict[str, Any] | str:
        """Add a supported, partial, or conflicting evidence fact."""
        if status not in {"supported", "partial", "conflict"}:
            return _format_evidence_error(
                "status must be one of: supported, partial, conflict"
            )
        try:
            item = dossier.add_evidence(
                claim=claim,
                source_ref=source_ref,
                title=title,
                locator=locator,
                tool_name=tool_name,
                confidence=confidence,
                tags=_normalize_tags(tags),
                quote=quote,
                status=status,  # type: ignore[arg-type]
                metadata=_normalize_metadata(metadata),
            )
        except Exception as exc:
            return _format_evidence_error(str(exc))
        return _item_to_dict(item)

    async def add_evidence_gap(
        question: str,
        reason: str,
        tags: list | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any] | str:
        """Record an unresolved evidence gap."""
        try:
            item = dossier.add_gap(
                question=question,
                reason=reason,
                tags=_normalize_tags(tags),
                metadata=_normalize_metadata(metadata),
            )
        except Exception as exc:
            return _format_evidence_error(str(exc))
        return _item_to_dict(item)

    async def list_evidence(
        tags: list | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]] | str:
        """List evidence facts, conflicts, and gaps."""
        if status is not None and status not in {
            "supported",
            "partial",
            "conflict",
            "gap",
        }:
            return _format_evidence_error(
                "status must be one of: supported, partial, conflict, gap"
            )
        items = dossier.list(
            tags=_normalize_tags(tags),
            status=status if status is not None else None,  # type: ignore[arg-type]
        )
        return [_item_to_dict(item) for item in items]

    async def summarize_evidence(
        max_chars: int = 4000,
        include_gaps: bool = True,
    ) -> str:
        """Return a bounded deterministic summary of the evidence dossier."""
        if not isinstance(max_chars, int):
            return _format_evidence_error("max_chars must be an integer")
        return dossier.summarize(max_chars=max_chars, include_gaps=include_gaps)

    async def evidence_coverage(required_tags: list | None = None) -> dict[str, Any]:
        """Report which required evidence tags are present or missing."""
        return dossier.coverage(_normalize_tags(required_tags)).to_dict()

    return [
        DecoratedTool(
            add_evidence,
            tool_name=ADD_EVIDENCE_TOOL_NAME,
            tool_description=(add_evidence.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
        DecoratedTool(
            add_evidence_gap,
            tool_name=ADD_EVIDENCE_GAP_TOOL_NAME,
            tool_description=(add_evidence_gap.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
        DecoratedTool(
            list_evidence,
            tool_name=LIST_EVIDENCE_TOOL_NAME,
            tool_description=(list_evidence.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
        DecoratedTool(
            summarize_evidence,
            tool_name=SUMMARIZE_EVIDENCE_TOOL_NAME,
            tool_description=(summarize_evidence.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
        DecoratedTool(
            evidence_coverage,
            tool_name=EVIDENCE_COVERAGE_TOOL_NAME,
            tool_description=(evidence_coverage.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
    ]
