"""L2 workspace state tests.

The workspace is a bounded, run-local notebook. It is not long-term memory,
not a skill system, and not persisted to disk by default.
"""

from __future__ import annotations

import pytest
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.workspace import InMemoryWorkspace, WorkspaceLimitError


def test_workspace_starts_empty() -> None:
    workspace = InMemoryWorkspace()

    assert workspace.list() == []
    assert workspace.stats().entry_count == 0
    assert workspace.stats().total_chars == 0


def test_workspace_write_and_read_note_round_trip() -> None:
    workspace = InMemoryWorkspace()

    entry = workspace.write_note(
        title="TCS FY25 revenue",
        content="TCS revenue from operations is captured from FY25 report.",
        source_refs=("obs:read_report:abc123",),
        metadata={"company": "TCS"},
    )

    loaded = workspace.read(entry.id)
    assert loaded == entry
    assert loaded is not None
    assert loaded.kind == "note"
    assert loaded.title == "TCS FY25 revenue"
    assert loaded.source_refs == ("obs:read_report:abc123",)
    assert loaded.metadata["company"] == "TCS"


def test_workspace_append_note_preserves_order() -> None:
    workspace = InMemoryWorkspace()
    entry = workspace.write_note(title="Research progress", content="Checked TCS.")

    updated = workspace.append_note(entry.id, "Checked Infosys.")

    assert updated.content == "Checked TCS.\nChecked Infosys."
    assert updated.created_at == entry.created_at
    assert updated.updated_at >= entry.updated_at
    assert workspace.list()[0].id == entry.id


def test_workspace_list_filters_by_kind() -> None:
    workspace = InMemoryWorkspace()
    note = workspace.write_note(title="Note", content="note")
    artifact = workspace.write_artifact(title="Table", content="| company | revenue |")
    summary = workspace.write_summary(title="Summary", content="summary")

    assert workspace.list(kind="note") == [note]
    assert workspace.list(kind="artifact") == [artifact]
    assert workspace.list(kind="summary") == [summary]


def test_workspace_summary_is_bounded_and_deterministic() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_note(title="A", content="alpha " * 20)
    workspace.write_artifact(title="B", content="beta " * 20)

    first = workspace.summarize(max_chars=80)
    second = workspace.summarize(max_chars=80)

    assert first == second
    assert len(first) <= 80
    assert "A" in first


def test_workspace_entries_survive_context_compaction() -> None:
    workspace = InMemoryWorkspace()
    entry = workspace.write_note(title="Keep", content="This survives compaction.")
    engine = ContextEngine(
        ContextConfig(
            max_context_tokens=8_000,
            optimal_budget=2_000,
            response_reserve=500,
            compaction_trigger=0.01,
        )
    )
    messages = [
        ChatMessage(role="system", content="system"),
        ChatMessage(role="user", content="task"),
        ChatMessage(role="assistant", content="old answer " * 5000),
        ChatMessage(role="user", content="latest"),
    ]

    # Workspace is separate from prompt messages. ContextEngine can compact
    # messages without touching workspace entries.
    import asyncio

    asyncio.run(engine.prepare(messages))

    assert workspace.read(entry.id) == entry


def test_workspace_limits_entry_and_total_size() -> None:
    workspace = InMemoryWorkspace(
        max_entries=2,
        max_entry_chars=10,
        max_total_chars=25,
    )

    first = workspace.write_note(title="A", content="x" * 50)
    workspace.write_note(title="B", content="y" * 10)

    assert first.content == "x" * 10
    with pytest.raises(WorkspaceLimitError):
        workspace.write_note(title="C", content="z")


def test_workspace_clear_removes_run_state() -> None:
    workspace = InMemoryWorkspace()
    workspace.write_note(title="A", content="alpha")

    workspace.clear()

    assert workspace.list() == []
    assert workspace.stats().entry_count == 0
