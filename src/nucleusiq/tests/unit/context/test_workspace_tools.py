"""L2.5 workspace tool tests.

Workspace state already exists at framework level. These tests prove the agent
can use it through bounded, run-local tools.
"""

from __future__ import annotations

import pytest
from nucleusiq.agents.context.workspace import InMemoryWorkspace
from nucleusiq.agents.context.workspace_tools import (
    READ_WORKSPACE_ENTRY_TOOL_NAME,
    SUMMARIZE_WORKSPACE_TOOL_NAME,
    WRITE_WORKSPACE_NOTE_TOOL_NAME,
    build_workspace_tools,
    is_workspace_tool_name,
)


def _tool_by_name(tools: list, name: str):
    return next(tool for tool in tools if tool.name == name)


@pytest.mark.asyncio
async def test_write_workspace_note_tool_writes_note() -> None:
    workspace = InMemoryWorkspace()
    tool = _tool_by_name(
        build_workspace_tools(workspace), WRITE_WORKSPACE_NOTE_TOOL_NAME
    )

    result = await tool.execute(
        title="TCS FY25 revenue",
        content="Captured revenue from FY25 annual report.",
        source_refs=["obs:read_report:abc123"],
        metadata={"company": "TCS"},
    )

    entries = workspace.list(kind="note")
    assert len(entries) == 1
    assert result["id"] == entries[0].id
    assert result["kind"] == "note"
    assert entries[0].source_refs == ("obs:read_report:abc123",)
    assert entries[0].metadata["company"] == "TCS"


@pytest.mark.asyncio
async def test_write_workspace_artifact_tool_writes_artifact() -> None:
    workspace = InMemoryWorkspace()
    tool = _tool_by_name(build_workspace_tools(workspace), "write_workspace_artifact")

    result = await tool.execute(
        title="Company comparison table",
        content="| company | revenue |",
    )

    entries = workspace.list(kind="artifact")
    assert len(entries) == 1
    assert result["id"] == entries[0].id
    assert result["content_chars"] == len("| company | revenue |")


@pytest.mark.asyncio
async def test_list_workspace_entries_tool_filters_by_kind() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_note(title="Note", content="note content")
    workspace.write_artifact(title="Artifact", content="artifact content")
    tool = _tool_by_name(build_workspace_tools(workspace), "list_workspace_entries")

    result = await tool.execute(kind="note")

    assert [entry["kind"] for entry in result] == ["note"]
    assert result[0]["title"] == "Note"
    assert "preview" in result[0]


@pytest.mark.asyncio
async def test_read_workspace_entry_tool_returns_entry() -> None:
    workspace = InMemoryWorkspace()
    entry = workspace.write_note(
        title="Progress",
        content="Checked TCS and Infosys.",
        source_refs=("obs:one",),
    )
    tool = _tool_by_name(
        build_workspace_tools(workspace), READ_WORKSPACE_ENTRY_TOOL_NAME
    )

    result = await tool.execute(entry_id=entry.id)

    assert result["id"] == entry.id
    assert result["content"] == "Checked TCS and Infosys."
    assert result["source_refs"] == ["obs:one"]


@pytest.mark.asyncio
async def test_summarize_workspace_tool_is_bounded() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_note(title="Long", content="alpha " * 100)
    tool = _tool_by_name(
        build_workspace_tools(workspace), SUMMARIZE_WORKSPACE_TOOL_NAME
    )

    result = await tool.execute(max_chars=80)

    assert isinstance(result, str)
    assert len(result) <= 80
    assert "Long" in result


@pytest.mark.asyncio
async def test_workspace_tool_limits_return_clear_error() -> None:
    workspace = InMemoryWorkspace(max_entries=1)
    workspace.write_note(title="Existing", content="already full")
    tool = _tool_by_name(
        build_workspace_tools(workspace), WRITE_WORKSPACE_NOTE_TOOL_NAME
    )

    result = await tool.execute(title="Too many", content="overflow")

    assert result.startswith("[workspace_error:")
    assert "limit" in result.lower()


def test_is_workspace_tool_name_identifies_framework_tools() -> None:
    assert is_workspace_tool_name(WRITE_WORKSPACE_NOTE_TOOL_NAME) is True
    assert is_workspace_tool_name("user_search") is False
    assert is_workspace_tool_name(None) is False
