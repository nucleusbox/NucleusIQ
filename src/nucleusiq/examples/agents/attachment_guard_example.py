"""Example: AttachmentGuardPlugin -- policy-based attachment validation.

Demonstrates how to configure and use the AttachmentGuardPlugin to
enforce attachment policies before an agent processes a task.

When a policy violation occurs, ``PluginHalt`` is raised inside the
agent and caught automatically -- the agent returns ``None`` (or the
halt result) instead of executing the task.

Scenarios covered:
    1. Allow only specific attachment types (TEXT, IMAGE_URL)
    2. Block specific attachment types (PDF)
    3. Enforce per-file size limits
    4. Enforce total attachment count
    5. Restrict file extensions
    6. Combined policies (type + size + extension)
"""

from __future__ import annotations

import asyncio

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.attachments import Attachment, AttachmentType
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.plugins.builtin.attachment_guard import AttachmentGuardPlugin


def _agent(plugins: list) -> Agent:
    return Agent(
        name="GuardBot",
        role="Analyst",
        objective="Process files",
        llm=MockLLM(),
        plugins=plugins,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )


async def example_allowed_types() -> None:
    """Only TEXT and IMAGE_URL attachments are permitted."""
    print("=" * 60)
    print("1. Allowed types: TEXT and IMAGE_URL only")
    print("=" * 60)

    agent = _agent(
        [
            AttachmentGuardPlugin(
                allowed_types=[AttachmentType.TEXT, AttachmentType.IMAGE_URL],
            ),
        ]
    )

    ok = Task(
        id="t1",
        objective="Summarize this document",
        attachments=[
            Attachment(
                type=AttachmentType.TEXT, data="Revenue grew 15%.", name="report.txt"
            ),
        ],
    )
    result = await agent.execute(ok)
    print(f"  TEXT attachment -> Processed (result: {str(result)[:50]}...)")

    blocked = Task(
        id="t2",
        objective="Analyze this PDF",
        attachments=[
            Attachment(type=AttachmentType.PDF, data=b"%PDF-fake", name="report.pdf"),
        ],
    )
    result = await agent.execute(blocked)
    print(f"  PDF attachment  -> Halted (result: {result})")
    print()


async def example_blocked_types() -> None:
    """Block PDF attachments specifically."""
    print("=" * 60)
    print("2. Blocked types: PDF")
    print("=" * 60)

    agent = _agent(
        [
            AttachmentGuardPlugin(blocked_types=[AttachmentType.PDF]),
        ]
    )

    blocked = Task(
        id="t3",
        objective="Process this file",
        attachments=[
            Attachment(type=AttachmentType.PDF, data=b"fake-pdf", name="doc.pdf"),
        ],
    )
    result = await agent.execute(blocked)
    print(f"  PDF -> Halted (result: {result})")

    ok = Task(
        id="t4",
        objective="Read this text",
        attachments=[
            Attachment(type=AttachmentType.TEXT, data="Hello world", name="hello.txt"),
        ],
    )
    result = await agent.execute(ok)
    print(f"  TEXT -> Processed (result: {str(result)[:50]}...)")
    print()


async def example_size_limit() -> None:
    """Enforce a 1 KB per-file size limit."""
    print("=" * 60)
    print("3. Per-file size limit: 1 KB")
    print("=" * 60)

    agent = _agent(
        [
            AttachmentGuardPlugin(max_file_size=1024),
        ]
    )

    small = Task(
        id="t5",
        objective="Process this",
        attachments=[
            Attachment(type=AttachmentType.TEXT, data="Small file.", name="tiny.txt"),
        ],
    )
    result = await agent.execute(small)
    print(f"  11 B file  -> Processed (result: {str(result)[:50]}...)")

    big = Task(
        id="t6",
        objective="Process this large file",
        attachments=[
            Attachment(type=AttachmentType.TEXT, data="x" * 5000, name="huge.txt"),
        ],
    )
    result = await agent.execute(big)
    print(f"  5 KB file  -> Halted (result: {result})")
    print()


async def example_count_limit() -> None:
    """Enforce a maximum of 2 attachments per task."""
    print("=" * 60)
    print("4. Maximum attachment count: 2")
    print("=" * 60)

    agent = _agent(
        [
            AttachmentGuardPlugin(max_attachments=2),
        ]
    )

    task = Task(
        id="t7",
        objective="Compare these files",
        attachments=[
            Attachment(type=AttachmentType.TEXT, data="a", name="a.txt"),
            Attachment(type=AttachmentType.TEXT, data="b", name="b.txt"),
            Attachment(type=AttachmentType.TEXT, data="c", name="c.txt"),
        ],
    )
    result = await agent.execute(task)
    print(f"  3 attachments -> Halted (result: {result})")
    print()


async def example_extension_filter() -> None:
    """Allow only .txt and .csv file extensions."""
    print("=" * 60)
    print("5. Extension filter: .txt and .csv only")
    print("=" * 60)

    agent = _agent(
        [
            AttachmentGuardPlugin(allowed_extensions=[".txt", ".csv"]),
        ]
    )

    ok = Task(
        id="t8",
        objective="Process CSV",
        attachments=[
            Attachment(type=AttachmentType.TEXT, data="a,b\n1,2", name="data.csv"),
        ],
    )
    result = await agent.execute(ok)
    print(f"  data.csv -> Processed (result: {str(result)[:50]}...)")

    blocked = Task(
        id="t9",
        objective="Process executable",
        attachments=[
            Attachment(type=AttachmentType.FILE_BYTES, data=b"MZ...", name="app.exe"),
        ],
    )
    result = await agent.execute(blocked)
    print(f"  app.exe  -> Halted (result: {result})")
    print()


async def example_combined_policy() -> None:
    """Combine type, size, and extension restrictions."""
    print("=" * 60)
    print("6. Combined: type + size + count + extension")
    print("=" * 60)

    guard = AttachmentGuardPlugin(
        allowed_types=[AttachmentType.TEXT, AttachmentType.IMAGE_URL],
        max_file_size=10 * 1024,
        max_attachments=3,
        allowed_extensions=[".txt", ".csv", ".png", ".jpg"],
    )

    agent = _agent([guard])

    task = Task(
        id="t10",
        objective="Analyze these safe inputs",
        attachments=[
            Attachment(
                type=AttachmentType.TEXT, data="Revenue data", name="report.txt"
            ),
            Attachment(
                type=AttachmentType.IMAGE_URL,
                data="https://example.com/chart.png",
                name="chart.png",
            ),
        ],
    )
    result = await agent.execute(task)
    print(f"  (TEXT + IMAGE_URL) -> Processed (result: {str(result)[:50]}...)")
    print()


async def main() -> None:
    await example_allowed_types()
    await example_blocked_types()
    await example_size_limit()
    await example_count_limit()
    await example_extension_filter()
    await example_combined_policy()
    print("All attachment guard examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
