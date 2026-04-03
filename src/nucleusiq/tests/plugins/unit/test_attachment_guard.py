"""Unit tests for AttachmentGuardPlugin."""

from __future__ import annotations

import pytest
from nucleusiq.agents.attachments import Attachment, AttachmentType
from nucleusiq.agents.task import Task
from nucleusiq.plugins.base import AgentContext
from nucleusiq.plugins.builtin.attachment_guard import AttachmentGuardPlugin
from nucleusiq.plugins.errors import PluginError, PluginHalt


def _ctx(attachments: list[Attachment] | None = None) -> AgentContext:
    task = Task(id="t1", objective="test", attachments=attachments)
    return AgentContext(
        agent_name="test", task=task, state="executing", config=None, memory=None
    )


class TestAttachmentGuardTypeFilter:
    @pytest.mark.asyncio
    async def test_allowed_types_pass(self) -> None:
        guard = AttachmentGuardPlugin(allowed_types=[AttachmentType.TEXT])
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="hello", name="f.txt")])
        result = await guard.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_allowed_types_block(self) -> None:
        guard = AttachmentGuardPlugin(allowed_types=[AttachmentType.TEXT])
        ctx = _ctx([Attachment(type=AttachmentType.IMAGE_URL, data="http://img.png")])
        with pytest.raises(PluginHalt, match="not in allowed types"):
            await guard.before_agent(ctx)

    @pytest.mark.asyncio
    async def test_blocked_types_block(self) -> None:
        guard = AttachmentGuardPlugin(blocked_types=[AttachmentType.PDF])
        ctx = _ctx([Attachment(type=AttachmentType.PDF, data=b"fake", name="f.pdf")])
        with pytest.raises(PluginHalt, match="blocked by policy"):
            await guard.before_agent(ctx)

    @pytest.mark.asyncio
    async def test_blocked_types_pass(self) -> None:
        guard = AttachmentGuardPlugin(blocked_types=[AttachmentType.PDF])
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="ok", name="f.txt")])
        result = await guard.before_agent(ctx)
        assert result is None

    def test_both_allowed_and_blocked_raises(self) -> None:
        with pytest.raises(PluginError, match="Cannot specify both"):
            AttachmentGuardPlugin(
                allowed_types=[AttachmentType.TEXT],
                blocked_types=[AttachmentType.PDF],
            )


class TestAttachmentGuardSizeLimit:
    @pytest.mark.asyncio
    async def test_under_limit_passes(self) -> None:
        guard = AttachmentGuardPlugin(max_file_size=1000)
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="small", name="f.txt")])
        result = await guard.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_over_limit_blocks(self) -> None:
        guard = AttachmentGuardPlugin(max_file_size=10)
        ctx = _ctx(
            [Attachment(type=AttachmentType.TEXT, data="x" * 100, name="big.txt")]
        )
        with pytest.raises(PluginHalt, match="exceeding"):
            await guard.before_agent(ctx)


class TestAttachmentGuardCount:
    @pytest.mark.asyncio
    async def test_under_limit_passes(self) -> None:
        guard = AttachmentGuardPlugin(max_attachments=3)
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="a", name="a.txt")])
        result = await guard.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_over_limit_blocks(self) -> None:
        guard = AttachmentGuardPlugin(max_attachments=1)
        ctx = _ctx(
            [
                Attachment(type=AttachmentType.TEXT, data="a", name="a.txt"),
                Attachment(type=AttachmentType.TEXT, data="b", name="b.txt"),
            ]
        )
        with pytest.raises(PluginHalt, match="Too many attachments"):
            await guard.before_agent(ctx)


class TestAttachmentGuardExtension:
    @pytest.mark.asyncio
    async def test_allowed_extension_passes(self) -> None:
        guard = AttachmentGuardPlugin(allowed_extensions=[".txt", ".csv"])
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="ok", name="data.csv")])
        result = await guard.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_blocked_extension(self) -> None:
        guard = AttachmentGuardPlugin(allowed_extensions=[".txt"])
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="data", name="data.exe")])
        with pytest.raises(PluginHalt, match="not in allowed extensions"):
            await guard.before_agent(ctx)

    @pytest.mark.asyncio
    async def test_no_name_skips_extension_check(self) -> None:
        guard = AttachmentGuardPlugin(allowed_extensions=[".txt"])
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="data")])
        result = await guard.before_agent(ctx)
        assert result is None


class TestAttachmentGuardNoAttachments:
    @pytest.mark.asyncio
    async def test_no_attachments_passes(self) -> None:
        guard = AttachmentGuardPlugin(
            allowed_types=[AttachmentType.TEXT],
            max_file_size=100,
            max_attachments=1,
        )
        ctx = _ctx(None)
        result = await guard.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_attachments_passes(self) -> None:
        guard = AttachmentGuardPlugin(max_attachments=1)
        ctx = _ctx([])
        result = await guard.before_agent(ctx)
        assert result is None


class TestAttachmentGuardName:
    def test_plugin_name(self) -> None:
        guard = AttachmentGuardPlugin()
        assert guard.name == "attachment_guard"
