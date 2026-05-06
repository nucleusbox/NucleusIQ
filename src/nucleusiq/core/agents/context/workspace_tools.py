"""Agent-facing tools for the run-local workspace.

These tools are factories, not module-level singletons. Each factory call binds
the returned tools to one execution's workspace so notes never leak across runs.
"""

from __future__ import annotations

from typing import Any

from nucleusiq.agents.context.policy import ContextPolicy
from nucleusiq.agents.context.workspace import (
    InMemoryWorkspace,
    WorkspaceEntry,
    WorkspaceLimitError,
)
from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.decorators import DecoratedTool

WRITE_WORKSPACE_NOTE_TOOL_NAME = "write_workspace_note"
WRITE_WORKSPACE_ARTIFACT_TOOL_NAME = "write_workspace_artifact"
LIST_WORKSPACE_ENTRIES_TOOL_NAME = "list_workspace_entries"
READ_WORKSPACE_ENTRY_TOOL_NAME = "read_workspace_entry"
SUMMARIZE_WORKSPACE_TOOL_NAME = "summarize_workspace"

_WORKSPACE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        WRITE_WORKSPACE_NOTE_TOOL_NAME,
        WRITE_WORKSPACE_ARTIFACT_TOOL_NAME,
        LIST_WORKSPACE_ENTRIES_TOOL_NAME,
        READ_WORKSPACE_ENTRY_TOOL_NAME,
        SUMMARIZE_WORKSPACE_TOOL_NAME,
    }
)

__all__ = [
    "WRITE_WORKSPACE_NOTE_TOOL_NAME",
    "WRITE_WORKSPACE_ARTIFACT_TOOL_NAME",
    "LIST_WORKSPACE_ENTRIES_TOOL_NAME",
    "READ_WORKSPACE_ENTRY_TOOL_NAME",
    "SUMMARIZE_WORKSPACE_TOOL_NAME",
    "build_workspace_tools",
    "is_context_management_tool_name",
    "is_workspace_tool_name",
]


def is_workspace_tool_name(tool_name: str | None) -> bool:
    """Return True when ``tool_name`` is an auto-injected workspace tool."""
    return tool_name in _WORKSPACE_TOOL_NAMES if tool_name else False


def is_context_management_tool_name(tool_name: str | None) -> bool:
    """Return True for framework-injected context-state tools.

    These tools are internal memory/context operations. They should be visible
    to the model, but they should not count against the user's external tool
    budget.
    """
    if is_workspace_tool_name(tool_name):
        return True

    try:
        from nucleusiq.agents.context.evidence_tools import is_evidence_tool_name

        if is_evidence_tool_name(tool_name):
            return True
    except Exception:
        pass

    try:
        from nucleusiq.agents.context.document_corpus_tools import (
            is_document_corpus_tool_name,
        )

        if is_document_corpus_tool_name(tool_name):
            return True
    except Exception:
        pass

    from nucleusiq.agents.context.recall_tools import is_recall_tool_name

    return is_recall_tool_name(tool_name)


def _format_workspace_error(message: str) -> str:
    return f"[workspace_error: {message}]"


def _normalize_source_refs(source_refs: list | tuple | str | None) -> tuple[str, ...]:
    if source_refs is None:
        return ()
    if isinstance(source_refs, str):
        return (source_refs,)
    if isinstance(source_refs, (list, tuple)):
        return tuple(str(ref) for ref in source_refs if str(ref).strip())
    return ()


def _normalize_metadata(metadata: dict | None) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return {str(key): value for key, value in metadata.items()}
    return {}


def _entry_to_dict(
    entry: WorkspaceEntry,
    *,
    include_content: bool = False,
    preview_chars: int = 240,
) -> dict[str, Any]:
    content = " ".join(entry.content.split())
    data: dict[str, Any] = {
        "id": entry.id,
        "kind": entry.kind,
        "title": entry.title,
        "source_refs": list(entry.source_refs),
        "content_chars": len(entry.content),
        "preview": content[:preview_chars],
        "metadata": dict(entry.metadata),
    }
    if include_content:
        data["content"] = entry.content
    return data


def build_workspace_tools(workspace: InMemoryWorkspace) -> list[BaseTool]:
    """Build workspace tools bound to one run-local workspace."""

    async def write_workspace_note(
        title: str,
        content: str,
        source_refs: list | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any] | str:
        """Write a bounded note into the current run-local workspace."""
        try:
            entry = workspace.write_note(
                title=title,
                content=content,
                source_refs=_normalize_source_refs(source_refs),
                metadata=_normalize_metadata(metadata),
            )
        except WorkspaceLimitError as exc:
            return _format_workspace_error(str(exc))
        return _entry_to_dict(entry)

    async def write_workspace_artifact(
        title: str,
        content: str,
        source_refs: list | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any] | str:
        """Write a bounded artifact into the current run-local workspace."""
        try:
            entry = workspace.write_artifact(
                title=title,
                content=content,
                source_refs=_normalize_source_refs(source_refs),
                metadata=_normalize_metadata(metadata),
            )
        except WorkspaceLimitError as exc:
            return _format_workspace_error(str(exc))
        return _entry_to_dict(entry)

    async def list_workspace_entries(
        kind: str | None = None,
    ) -> list[dict[str, Any]] | str:
        """List entries saved in the current run-local workspace."""
        if kind is not None and kind not in {"note", "artifact", "summary"}:
            return _format_workspace_error(
                "kind must be one of: note, artifact, summary"
            )
        entries = workspace.list(kind=kind if kind is not None else None)
        return [_entry_to_dict(entry) for entry in entries]

    async def read_workspace_entry(entry_id: str) -> dict[str, Any] | str:
        """Read one workspace entry by ID."""
        entry = workspace.read(entry_id)
        if entry is None:
            return _format_workspace_error(f"entry_id {entry_id!r} not found")
        return _entry_to_dict(entry, include_content=True)

    async def summarize_workspace(max_chars: int = 4000) -> str:
        """Return a bounded deterministic summary of the current workspace."""
        if not isinstance(max_chars, int):
            return _format_workspace_error("max_chars must be an integer")
        return workspace.summarize(max_chars=max_chars)

    return [
        DecoratedTool(
            write_workspace_note,
            tool_name=WRITE_WORKSPACE_NOTE_TOOL_NAME,
            tool_description=(write_workspace_note.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
        DecoratedTool(
            write_workspace_artifact,
            tool_name=WRITE_WORKSPACE_ARTIFACT_TOOL_NAME,
            tool_description=(write_workspace_artifact.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
        DecoratedTool(
            list_workspace_entries,
            tool_name=LIST_WORKSPACE_ENTRIES_TOOL_NAME,
            tool_description=(list_workspace_entries.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
        DecoratedTool(
            read_workspace_entry,
            tool_name=READ_WORKSPACE_ENTRY_TOOL_NAME,
            tool_description=(read_workspace_entry.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
        DecoratedTool(
            summarize_workspace,
            tool_name=SUMMARIZE_WORKSPACE_TOOL_NAME,
            tool_description=(summarize_workspace.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
    ]
