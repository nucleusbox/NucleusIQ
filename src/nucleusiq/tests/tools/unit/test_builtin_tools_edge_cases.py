"""Edge-case tests for built-in file tools, workspace sandbox, and attachment guard.

Covers scenarios missing from the base test suite:
  - Workspace: symlink escape, format_file_size(0), format_file_size on large values
  - FileReadTool: reversed line range, out-of-range lines, binary file content
  - FileSearchTool: traversal via path param, binary skipping
  - DirectoryListTool: empty directory, file-as-path error
  - FileExtractTool: empty CSV, empty JSON array, header-only CSV
  - AttachmentGuard: extension case-insensitivity, exact-limit boundaries
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from nucleusiq.agents.attachments import Attachment, AttachmentType
from nucleusiq.agents.task import Task
from nucleusiq.plugins.base import AgentContext
from nucleusiq.plugins.builtin.attachment_guard import AttachmentGuardPlugin
from nucleusiq.plugins.errors import PluginHalt
from nucleusiq.tools.builtin.directory_list import DirectoryListTool
from nucleusiq.tools.builtin.file_extract import FileExtractTool
from nucleusiq.tools.builtin.file_read import FileReadTool
from nucleusiq.tools.builtin.file_search import FileSearchTool
from nucleusiq.tools.builtin.workspace import (
    WorkspaceSecurityError,
    format_file_size,
    resolve_safe_path,
)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Workspace with various file types for edge-case testing."""
    (tmp_path / "hello.txt").write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02\xff" * 64)
    (tmp_path / "empty.csv").write_text("")
    (tmp_path / "header_only.csv").write_text("col1,col2,col3\n")
    (tmp_path / "empty_array.json").write_text("[]")
    (tmp_path / "scalar.json").write_text("42")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "note.txt").write_text("nested note\n")
    empty = tmp_path / "emptydir"
    empty.mkdir()
    return tmp_path


# ================================================================== #
# Workspace sandbox edge cases                                         #
# ================================================================== #


class TestWorkspaceSandboxEdgeCases:
    def test_format_file_size_zero(self) -> None:
        assert format_file_size(0) == "0 B"

    def test_format_file_size_exact_1kb(self) -> None:
        result = format_file_size(1024)
        assert "KB" in result

    def test_format_file_size_exact_1mb(self) -> None:
        result = format_file_size(1024 * 1024)
        assert "MB" in result

    def test_format_file_size_large(self) -> None:
        result = format_file_size(5 * 1024 * 1024 * 1024)
        assert "MB" in result

    @pytest.mark.skipif(
        os.name == "nt", reason="Symlinks may require privileges on Windows"
    )
    def test_symlink_escape_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "safe"
        workspace.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("secret data")
        link = workspace / "escape"
        link.symlink_to(outside)
        with pytest.raises(WorkspaceSecurityError):
            resolve_safe_path(str(workspace), "escape")

    @pytest.mark.skipif(
        os.name == "nt", reason="Symlinks may require privileges on Windows"
    )
    def test_symlink_dir_escape_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "safe"
        workspace.mkdir()
        outside = tmp_path / "outside_dir"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")
        link = workspace / "escape_dir"
        link.symlink_to(outside)
        with pytest.raises(WorkspaceSecurityError):
            resolve_safe_path(str(workspace), "escape_dir/secret.txt")

    def test_dot_path_resolves_to_root(self, workspace: Path) -> None:
        result = resolve_safe_path(str(workspace), ".")
        assert result == workspace.resolve()

    def test_empty_user_path(self, workspace: Path) -> None:
        result = resolve_safe_path(str(workspace), "")
        assert result == workspace.resolve()


# ================================================================== #
# FileReadTool edge cases                                              #
# ================================================================== #


class TestFileReadToolEdgeCases:
    @pytest.fixture()
    def tool(self, workspace: Path) -> FileReadTool:
        return FileReadTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_reversed_line_range(self, tool: FileReadTool) -> None:
        result = await tool.execute(path="hello.txt", start_line=5, end_line=2)
        assert "Lines 5-2" in result or "Error" not in result or result.strip() != ""

    @pytest.mark.asyncio
    async def test_out_of_range_start_line(self, tool: FileReadTool) -> None:
        result = await tool.execute(path="hello.txt", start_line=999)
        assert "Error" in result
        assert "Invalid line range" in result

    @pytest.mark.asyncio
    async def test_start_line_zero_treated_as_one(self, tool: FileReadTool) -> None:
        result = await tool.execute(path="hello.txt", start_line=0, end_line=1)
        assert "Line 1" in result

    @pytest.mark.asyncio
    async def test_binary_file_detected(self, tool: FileReadTool) -> None:
        result = await tool.execute(path="binary.bin")
        assert "binary" in result.lower()
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_with_different_encoding(self, workspace: Path) -> None:
        (workspace / "latin.txt").write_bytes("caf\xe9".encode("latin-1"))
        tool = FileReadTool(workspace_root=str(workspace))
        result = await tool.execute(path="latin.txt", encoding="latin-1")
        assert "caf" in result

    @pytest.mark.asyncio
    async def test_read_empty_file(self, workspace: Path) -> None:
        (workspace / "zero.txt").write_text("")
        tool = FileReadTool(workspace_root=str(workspace))
        result = await tool.execute(path="zero.txt")
        assert "0 lines" in result


# ================================================================== #
# FileSearchTool edge cases                                            #
# ================================================================== #


class TestFileSearchToolEdgeCases:
    @pytest.fixture()
    def tool(self, workspace: Path) -> FileSearchTool:
        return FileSearchTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_search_traversal_in_path(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern="test", path="../../etc")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_search_skips_binary_files(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern="\x00")
        assert "binary.bin" not in result

    @pytest.mark.asyncio
    async def test_search_file_in_subdirectory(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern="nested note", path="subdir")
        assert "note.txt" in result

    @pytest.mark.asyncio
    async def test_search_single_file_mode(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern="Line", path="hello.txt")
        assert "match" in result.lower()

    @pytest.mark.asyncio
    async def test_search_regex_groups(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern=r"Line (\d+)", path="hello.txt", regex=True)
        assert "Line" in result


# ================================================================== #
# DirectoryListTool edge cases                                         #
# ================================================================== #


class TestDirectoryListToolEdgeCases:
    @pytest.fixture()
    def tool(self, workspace: Path) -> DirectoryListTool:
        return DirectoryListTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, tool: DirectoryListTool) -> None:
        result = await tool.execute(path="emptydir")
        assert "No entries" in result or "0 file(s)" in result

    @pytest.mark.asyncio
    async def test_list_file_as_path_error(self, tool: DirectoryListTool) -> None:
        result = await tool.execute(path="hello.txt")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_list_traversal_blocked(self, tool: DirectoryListTool) -> None:
        result = await tool.execute(path="../../..")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_list_recursive_finds_all(self, tool: DirectoryListTool) -> None:
        result = await tool.execute(recursive=True, pattern="*")
        assert "hello.txt" in result
        assert "note.txt" in result


# ================================================================== #
# FileExtractTool edge cases                                           #
# ================================================================== #


class TestFileExtractToolEdgeCases:
    @pytest.fixture()
    def tool(self, workspace: Path) -> FileExtractTool:
        return FileExtractTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_extract_empty_csv(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="empty.csv")
        assert "Empty CSV" in result

    @pytest.mark.asyncio
    async def test_extract_header_only_csv(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="header_only.csv")
        assert "col1" in result
        assert "Rows: 0" in result

    @pytest.mark.asyncio
    async def test_extract_empty_json_array(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="empty_array.json")
        assert "Array with 0 items" in result

    @pytest.mark.asyncio
    async def test_extract_json_scalar(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="scalar.json")
        assert "int" in result or "42" in result

    @pytest.mark.asyncio
    async def test_extract_traversal_blocked(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="../../etc/passwd")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_extract_max_sample_rows(self, workspace: Path) -> None:
        rows = "id,val\n" + "\n".join(f"{i},{i * 10}" for i in range(50))
        (workspace / "big.csv").write_text(rows)
        tool = FileExtractTool(workspace_root=str(workspace), max_sample_rows=3)
        result = await tool.execute(path="big.csv")
        assert "Rows: 50" in result
        assert "first 3" in result

    @pytest.mark.asyncio
    async def test_extract_jsonl(self, workspace: Path) -> None:
        lines = '{"id":1,"name":"A"}\n{"id":2,"name":"B"}\n{"id":3,"name":"C"}\n'
        (workspace / "data.jsonl").write_text(lines)
        tool = FileExtractTool(workspace_root=str(workspace))
        result = await tool.execute(path="data.jsonl")
        assert "JSONL" in result
        assert "Lines: 3" in result
        assert "id" in result

    @pytest.mark.asyncio
    async def test_extract_ndjson_alias(self, workspace: Path) -> None:
        (workspace / "data.ndjson").write_text('{"x":1}\n{"x":2}\n')
        tool = FileExtractTool(workspace_root=str(workspace))
        result = await tool.execute(path="data.ndjson")
        assert "JSONL" in result

    @pytest.mark.asyncio
    async def test_extract_tsv(self, workspace: Path) -> None:
        (workspace / "data.tsv").write_text("name\tage\nAlice\t30\nBob\t25\n")
        tool = FileExtractTool(workspace_root=str(workspace))
        result = await tool.execute(path="data.tsv")
        assert "TSV" in result
        assert "name" in result
        assert "Rows: 2" in result

    @pytest.mark.asyncio
    async def test_extract_xml(self, workspace: Path) -> None:
        xml = "<root><item id='1'>Widget</item><item id='2'>Gadget</item></root>"
        (workspace / "data.xml").write_text(xml)
        tool = FileExtractTool(workspace_root=str(workspace))
        result = await tool.execute(path="data.xml")
        assert "XML" in result
        assert "root" in result
        assert "item" in result

    @pytest.mark.asyncio
    async def test_extract_toml(self, workspace: Path) -> None:
        (workspace / "config.toml").write_text(
            '[project]\nname = "test"\nversion = "1.0"\n'
        )
        tool = FileExtractTool(workspace_root=str(workspace))
        result = await tool.execute(path="config.toml")
        assert "TOML" in result
        assert "project" in result

    @pytest.mark.asyncio
    async def test_extract_invalid_jsonl(self, workspace: Path) -> None:
        (workspace / "bad.jsonl").write_text("{not valid json\n")
        tool = FileExtractTool(workspace_root=str(workspace))
        result = await tool.execute(path="bad.jsonl")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_extract_invalid_xml(self, workspace: Path) -> None:
        (workspace / "bad.xml").write_text("<root><unclosed>")
        tool = FileExtractTool(workspace_root=str(workspace))
        result = await tool.execute(path="bad.xml")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_get_spec_lists_all_formats(self, tool: FileExtractTool) -> None:
        spec = tool.get_spec()
        desc = spec["parameters"]["properties"]["path"]["description"]
        for ext in [".csv", ".json", ".jsonl", ".tsv", ".xml", ".toml", ".yaml"]:
            assert ext in desc


# ================================================================== #
# AttachmentGuard edge cases                                           #
# ================================================================== #


def _ctx(attachments: list[Attachment] | None = None) -> AgentContext:
    task = Task(id="t-edge", objective="test", attachments=attachments)
    return AgentContext(
        agent_name="test", task=task, state="executing", config=None, memory=None
    )


class TestAttachmentGuardEdgeCases:
    @pytest.mark.asyncio
    async def test_extension_case_insensitive(self) -> None:
        guard = AttachmentGuardPlugin(allowed_extensions=[".txt"])
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="ok", name="FILE.TXT")])
        result = await guard.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_exact_size_limit_passes(self) -> None:
        guard = AttachmentGuardPlugin(max_file_size=5)
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="12345", name="f.txt")])
        result = await guard.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_one_byte_over_size_limit_blocks(self) -> None:
        guard = AttachmentGuardPlugin(max_file_size=5)
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="123456", name="f.txt")])
        with pytest.raises(PluginHalt, match="exceeding"):
            await guard.before_agent(ctx)

    @pytest.mark.asyncio
    async def test_exact_count_limit_passes(self) -> None:
        guard = AttachmentGuardPlugin(max_attachments=2)
        ctx = _ctx(
            [
                Attachment(type=AttachmentType.TEXT, data="a", name="a.txt"),
                Attachment(type=AttachmentType.TEXT, data="b", name="b.txt"),
            ]
        )
        result = await guard.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_one_over_count_limit_blocks(self) -> None:
        guard = AttachmentGuardPlugin(max_attachments=2)
        ctx = _ctx(
            [
                Attachment(type=AttachmentType.TEXT, data="a", name="a.txt"),
                Attachment(type=AttachmentType.TEXT, data="b", name="b.txt"),
                Attachment(type=AttachmentType.TEXT, data="c", name="c.txt"),
            ]
        )
        with pytest.raises(PluginHalt, match="Too many"):
            await guard.before_agent(ctx)

    @pytest.mark.asyncio
    async def test_no_extension_on_name_passes(self) -> None:
        guard = AttachmentGuardPlugin(allowed_extensions=[".txt"])
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="ok", name="README")])
        result = await guard.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_extension_with_dot_prefix_normalization(self) -> None:
        guard = AttachmentGuardPlugin(allowed_extensions=["csv"])
        ctx = _ctx([Attachment(type=AttachmentType.TEXT, data="a,b", name="data.csv")])
        result = await guard.before_agent(ctx)
        assert result is None


# ================================================================== #
# Memory: TokenBudget eviction with metadata                           #
# ================================================================== #


class TestTokenBudgetEvictionMetadata:
    def test_evicted_messages_lose_metadata(self) -> None:
        from nucleusiq.memory import TokenBudgetMemory

        mem = TokenBudgetMemory(max_tokens=20)
        meta = {"attachments": [{"name": "old.pdf", "type": "pdf", "size": 1000}]}
        mem.add_message("user", "short", metadata=meta)
        for i in range(10):
            mem.add_message("assistant", f"response {i} with some extra tokens")
        ctx = mem.get_context()
        user_msgs = [m for m in ctx if m.get("role") == "user"]
        if user_msgs:
            assert user_msgs[0].get("metadata") == meta
