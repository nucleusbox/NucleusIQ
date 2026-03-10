"""Unit tests for built-in file tools and workspace sandbox."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from nucleusiq.tools.builtin.directory_list import DirectoryListTool
from nucleusiq.tools.builtin.file_extract import FileExtractTool
from nucleusiq.tools.builtin.file_read import FileReadTool
from nucleusiq.tools.builtin.file_search import FileSearchTool
from nucleusiq.tools.builtin.workspace import (
    WorkspaceSecurityError,
    format_file_size,
    resolve_safe_path,
)

# ================================================================== #
# Fixtures                                                             #
# ================================================================== #


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Create a temp workspace with sample files."""
    (tmp_path / "hello.txt").write_text("Hello World\nLine 2\nLine 3\n")
    (tmp_path / "data.csv").write_text(
        "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF\n"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"version": "1.0", "features": ["a", "b"], "count": 42})
    )
    (tmp_path / "items.json").write_text(
        json.dumps([{"id": 1, "name": "Widget"}, {"id": 2, "name": "Gadget"}])
    )
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("Nested file content\nrevenue: 1000\n")
    (sub / "deep.csv").write_text("col1,col2\nx,1\ny,2\n")
    return tmp_path


# ================================================================== #
# Workspace Sandbox Tests                                              #
# ================================================================== #


class TestWorkspaceSandbox:
    def test_resolve_normal_path(self, workspace: Path) -> None:
        result = resolve_safe_path(str(workspace), "hello.txt")
        assert result == workspace / "hello.txt"

    def test_resolve_nested_path(self, workspace: Path) -> None:
        result = resolve_safe_path(str(workspace), "subdir/nested.txt")
        assert result == workspace / "subdir" / "nested.txt"

    def test_block_traversal(self, workspace: Path) -> None:
        with pytest.raises(WorkspaceSecurityError, match="outside the workspace"):
            resolve_safe_path(str(workspace), "../../../etc/passwd")

    def test_block_absolute_path(self, workspace: Path) -> None:
        abs_path = (
            "/etc/passwd" if os.name != "nt" else "C:\\Windows\\System32\\cmd.exe"
        )
        with pytest.raises(WorkspaceSecurityError):
            resolve_safe_path(str(workspace), abs_path)

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        with pytest.raises(WorkspaceSecurityError, match="does not exist"):
            resolve_safe_path(str(tmp_path / "nonexistent"), "file.txt")

    def test_format_file_size_bytes(self) -> None:
        assert format_file_size(500) == "500 B"

    def test_format_file_size_kb(self) -> None:
        assert "KB" in format_file_size(5000)

    def test_format_file_size_mb(self) -> None:
        assert "MB" in format_file_size(5_000_000)


# ================================================================== #
# FileReadTool Tests                                                   #
# ================================================================== #


class TestFileReadTool:
    @pytest.fixture()
    def tool(self, workspace: Path) -> FileReadTool:
        return FileReadTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_read_full_file(self, tool: FileReadTool) -> None:
        result = await tool.execute(path="hello.txt")
        assert "Hello World" in result
        assert "3 lines" in result

    @pytest.mark.asyncio
    async def test_read_line_range(self, tool: FileReadTool) -> None:
        result = await tool.execute(path="hello.txt", start_line=2, end_line=3)
        assert "Line 2" in result
        assert "Lines 2-3 of 3" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tool: FileReadTool) -> None:
        result = await tool.execute(path="missing.txt")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_traversal_blocked(self, tool: FileReadTool) -> None:
        result = await tool.execute(path="../../etc/passwd")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_empty_path(self, tool: FileReadTool) -> None:
        result = await tool.execute()
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_large_file_truncation(self, workspace: Path) -> None:
        large = workspace / "large.txt"
        large.write_text("\n".join(f"Line {i}" for i in range(1000)))
        tool = FileReadTool(workspace_root=str(workspace), max_lines=100)
        result = await tool.execute(path="large.txt")
        assert "Showing first 100 of 1000 lines" in result
        assert "start_line/end_line" in result

    @pytest.mark.asyncio
    async def test_read_nested_file(self, tool: FileReadTool) -> None:
        result = await tool.execute(path="subdir/nested.txt")
        assert "Nested file content" in result

    def test_get_spec(self, tool: FileReadTool) -> None:
        spec = tool.get_spec()
        assert spec["name"] == "file_read"
        assert "path" in spec["parameters"]["properties"]
        assert "start_line" in spec["parameters"]["properties"]
        assert spec["parameters"]["required"] == ["path"]


# ================================================================== #
# FileSearchTool Tests                                                 #
# ================================================================== #


class TestFileSearchTool:
    @pytest.fixture()
    def tool(self, workspace: Path) -> FileSearchTool:
        return FileSearchTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_search_in_file(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern="Hello", path="hello.txt")
        assert "hello.txt:1" in result
        assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_search_in_directory(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern="revenue")
        assert "nested.txt" in result
        assert "1000" in result

    @pytest.mark.asyncio
    async def test_search_regex(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern=r"Line \d+", path="hello.txt", regex=True)
        assert "Line 2" in result

    @pytest.mark.asyncio
    async def test_search_no_match(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern="ZZZZZ_NO_MATCH", path="hello.txt")
        assert "No matches" in result

    @pytest.mark.asyncio
    async def test_search_invalid_regex(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern="[invalid", regex=True)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_search_empty_pattern(self, tool: FileSearchTool) -> None:
        result = await tool.execute()
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_search_max_results(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern="Line", path="hello.txt", max_results=1)
        assert "1 match" in result

    @pytest.mark.asyncio
    async def test_search_nonexistent_path(self, tool: FileSearchTool) -> None:
        result = await tool.execute(pattern="test", path="missing_dir")
        assert "Error" in result or "does not exist" in result

    def test_get_spec(self, tool: FileSearchTool) -> None:
        spec = tool.get_spec()
        assert spec["name"] == "file_search"
        assert "pattern" in spec["parameters"]["properties"]
        assert "regex" in spec["parameters"]["properties"]


# ================================================================== #
# DirectoryListTool Tests                                              #
# ================================================================== #


class TestDirectoryListTool:
    @pytest.fixture()
    def tool(self, workspace: Path) -> DirectoryListTool:
        return DirectoryListTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_list_root(self, tool: DirectoryListTool) -> None:
        result = await tool.execute()
        assert "hello.txt" in result
        assert "data.csv" in result
        assert "[DIR]" in result
        assert "[FILE]" in result

    @pytest.mark.asyncio
    async def test_list_subdirectory(self, tool: DirectoryListTool) -> None:
        result = await tool.execute(path="subdir")
        assert "nested.txt" in result

    @pytest.mark.asyncio
    async def test_list_with_glob(self, tool: DirectoryListTool) -> None:
        result = await tool.execute(pattern="*.csv")
        assert "data.csv" in result
        assert "hello.txt" not in result

    @pytest.mark.asyncio
    async def test_list_recursive(self, tool: DirectoryListTool) -> None:
        result = await tool.execute(recursive=True, pattern="*.txt")
        assert "hello.txt" in result
        assert "nested.txt" in result

    @pytest.mark.asyncio
    async def test_list_nonexistent_dir(self, tool: DirectoryListTool) -> None:
        result = await tool.execute(path="missing_dir")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_list_empty_result(self, tool: DirectoryListTool) -> None:
        result = await tool.execute(pattern="*.xyz_no_match")
        assert "No entries" in result

    def test_get_spec(self, tool: DirectoryListTool) -> None:
        spec = tool.get_spec()
        assert spec["name"] == "directory_list"
        assert "pattern" in spec["parameters"]["properties"]
        assert "recursive" in spec["parameters"]["properties"]


# ================================================================== #
# FileExtractTool Tests                                                #
# ================================================================== #


class TestFileExtractTool:
    @pytest.fixture()
    def tool(self, workspace: Path) -> FileExtractTool:
        return FileExtractTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_extract_csv(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="data.csv")
        assert "CSV" in result
        assert "name" in result
        assert "age" in result
        assert "city" in result
        assert "Rows: 3" in result
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_extract_csv_with_query(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="data.csv", query="ages")
        assert "Query context: ages" in result

    @pytest.mark.asyncio
    async def test_extract_json_object(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="config.json")
        assert "JSON" in result
        assert "Object with 3 keys" in result
        assert "version" in result
        assert "features" in result

    @pytest.mark.asyncio
    async def test_extract_json_array(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="items.json")
        assert "JSON" in result
        assert "Array with 2 items" in result
        assert "Widget" in result

    @pytest.mark.asyncio
    async def test_extract_unsupported_format(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="hello.txt")
        assert "Unsupported file format" in result

    @pytest.mark.asyncio
    async def test_extract_nonexistent(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="nope.csv")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_extract_empty_path(self, tool: FileExtractTool) -> None:
        result = await tool.execute()
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_extract_nested_csv(self, tool: FileExtractTool) -> None:
        result = await tool.execute(path="subdir/deep.csv")
        assert "col1" in result
        assert "col2" in result
        assert "Rows: 2" in result

    @pytest.mark.asyncio
    async def test_extract_invalid_json(self, workspace: Path) -> None:
        (workspace / "bad.json").write_text("{invalid json}")
        tool = FileExtractTool(workspace_root=str(workspace))
        result = await tool.execute(path="bad.json")
        assert "Error" in result and "Invalid JSON" in result

    def test_get_spec(self, tool: FileExtractTool) -> None:
        spec = tool.get_spec()
        assert spec["name"] == "file_extract"
        assert "path" in spec["parameters"]["properties"]
        assert "query" in spec["parameters"]["properties"]
