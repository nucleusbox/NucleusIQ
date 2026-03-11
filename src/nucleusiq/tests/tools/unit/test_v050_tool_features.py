"""Tests for v0.5.0 tool enhancements.

Covers:
- FileWriteTool (new)
- FileExtractTool query filtering (columns, key_path)
- FileSearchTool configurable binary extensions
- DirectoryListTool max_entries limit
- FileReadTool encoding auto-detection
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nucleusiq.tools.builtin.directory_list import DirectoryListTool
from nucleusiq.tools.builtin.file_extract import FileExtractTool
from nucleusiq.tools.builtin.file_read import FileReadTool
from nucleusiq.tools.builtin.file_search import (
    DEFAULT_BINARY_EXTENSIONS,
    FileSearchTool,
)
from nucleusiq.tools.builtin.file_write import FileWriteTool

# ================================================================== #
# Fixtures                                                             #
# ================================================================== #


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Create a workspace with sample files for v0.5.0 feature tests."""
    (tmp_path / "hello.txt").write_text("Hello World\nLine 2\nLine 3\n")
    (tmp_path / "data.csv").write_text(
        "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF\n"
    )
    (tmp_path / "data.tsv").write_text("name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tLA\n")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "database": {"host": "localhost", "port": 5432},
                "features": ["a", "b"],
            }
        )
    )
    (tmp_path / "items.json").write_text(
        json.dumps([{"id": 1, "name": "Widget"}, {"id": 2, "name": "Gadget"}])
    )
    (tmp_path / "sample.yaml").write_text(
        "app:\n  name: test\n  version: 1.0\nusers:\n  - alice\n  - bob\n"
    )
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("Nested file content\n")
    (sub / "code.py").write_text("print('hello')\n")
    (sub / "image.png").write_bytes(b"\x89PNG fake binary")
    (sub / "archive.zip").write_bytes(b"PK fake zip")
    return tmp_path


# ================================================================== #
# FileWriteTool                                                        #
# ================================================================== #


class TestFileWriteTool:
    @pytest.mark.asyncio
    async def test_write_new_file(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        result = await tool.execute(path="new_file.txt", content="Hello!")
        assert "Created" in result
        assert (workspace / "new_file.txt").read_text() == "Hello!"

    @pytest.mark.asyncio
    async def test_overwrite_existing_file(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        result = await tool.execute(path="hello.txt", content="Replaced")
        assert "Overwrote" in result
        assert (workspace / "hello.txt").read_text() == "Replaced"

    @pytest.mark.asyncio
    async def test_backup_on_overwrite(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace), backup=True)
        original = (workspace / "hello.txt").read_text()
        await tool.execute(path="hello.txt", content="New content")
        backup = workspace / "hello.txt.bak"
        assert backup.exists()
        assert backup.read_text() == original

    @pytest.mark.asyncio
    async def test_no_backup_when_disabled(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace), backup=False)
        await tool.execute(path="hello.txt", content="New content")
        assert not (workspace / "hello.txt.bak").exists()

    @pytest.mark.asyncio
    async def test_append_mode(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        await tool.execute(path="hello.txt", content="\nAppended!", mode="append")
        content = (workspace / "hello.txt").read_text()
        assert content.endswith("\nAppended!")
        assert content.startswith("Hello World")

    @pytest.mark.asyncio
    async def test_append_result_message(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        result = await tool.execute(path="hello.txt", content="more", mode="append")
        assert "Appended to" in result

    @pytest.mark.asyncio
    async def test_create_parent_directories(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        result = await tool.execute(path="a/b/c/deep.txt", content="deep")
        assert "Created" in result
        assert (workspace / "a" / "b" / "c" / "deep.txt").read_text() == "deep"

    @pytest.mark.asyncio
    async def test_no_create_parents_error(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        result = await tool.execute(
            path="x/y/z/file.txt", content="hi", create_parents=False
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_empty_path_error(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        result = await tool.execute(path="", content="test")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_invalid_mode_error(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        result = await tool.execute(path="f.txt", content="test", mode="delete")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_max_write_size_exceeded(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace), max_write_size=10)
        result = await tool.execute(path="big.txt", content="x" * 100)
        assert "Error" in result
        assert "limit" in result

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        result = await tool.execute(path="../escape.txt", content="bad")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_get_spec(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        spec = tool.get_spec()
        assert spec["name"] == "file_write"
        assert "path" in spec["parameters"]["properties"]
        assert "content" in spec["parameters"]["properties"]
        assert "mode" in spec["parameters"]["properties"]
        assert set(spec["parameters"]["required"]) == {"path", "content"}


# ================================================================== #
# FileExtractTool — column filtering                                   #
# ================================================================== #


class TestFileExtractColumns:
    @pytest.mark.asyncio
    async def test_csv_column_filter(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="data.csv", columns="name,city")
        assert "name" in result
        assert "city" in result
        assert "Alice" in result
        assert "NYC" in result

    @pytest.mark.asyncio
    async def test_csv_column_filter_case_insensitive(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="data.csv", columns="NAME,City")
        assert "Alice" in result
        assert "NYC" in result

    @pytest.mark.asyncio
    async def test_csv_column_filter_unknown_columns_ignored(
        self, workspace: Path
    ) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="data.csv", columns="name,unknown_col")
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_csv_no_matching_columns_shows_all(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="data.csv", columns="nonexistent")
        assert "name" in result
        assert "age" in result
        assert "city" in result

    @pytest.mark.asyncio
    async def test_tsv_column_filter(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="data.tsv", columns="name")
        assert "Alice" in result
        assert "Bob" in result

    @pytest.mark.asyncio
    async def test_columns_ignored_for_json(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="config.json", columns="name")
        assert "version" in result


# ================================================================== #
# FileExtractTool — key_path filtering                                 #
# ================================================================== #


class TestFileExtractKeyPath:
    @pytest.mark.asyncio
    async def test_json_key_path_nested(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="config.json", key_path="database.host")
        assert "localhost" in result

    @pytest.mark.asyncio
    async def test_json_key_path_top_level(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="config.json", key_path="version")
        assert "1.0" in result

    @pytest.mark.asyncio
    async def test_json_key_path_array_index(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="items.json", key_path="0.name")
        assert "Widget" in result

    @pytest.mark.asyncio
    async def test_json_key_path_not_found(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="config.json", key_path="nonexistent.deep")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_yaml_key_path(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="sample.yaml", key_path="app.name")
        assert "test" in result

    @pytest.mark.asyncio
    async def test_yaml_key_path_not_found(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="sample.yaml", key_path="missing.path")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_key_path_with_query(self, workspace: Path) -> None:
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(
            path="config.json", key_path="database", query="db settings"
        )
        assert "db settings" in result
        assert "host" in result

    @pytest.mark.asyncio
    async def test_get_spec_includes_columns_and_key_path(
        self, workspace: Path
    ) -> None:
        tool = FileExtractTool(str(workspace))
        spec = tool.get_spec()
        props = spec["parameters"]["properties"]
        assert "columns" in props
        assert "key_path" in props


# ================================================================== #
# FileSearchTool — configurable binary extensions                      #
# ================================================================== #


class TestFileSearchConfigurableExtensions:
    @pytest.mark.asyncio
    async def test_default_binary_extensions_skip(self, workspace: Path) -> None:
        tool = FileSearchTool(str(workspace))
        result = await tool.execute(pattern="fake", path="subdir")
        assert ".png" not in result

    @pytest.mark.asyncio
    async def test_include_extensions_only_py(self, workspace: Path) -> None:
        tool = FileSearchTool(str(workspace), include_extensions=[".py"])
        result = await tool.execute(pattern="hello", path="subdir")
        assert "code.py" in result
        assert "nested.txt" not in result

    @pytest.mark.asyncio
    async def test_exclude_extensions_adds_to_binary_set(self, workspace: Path) -> None:
        tool = FileSearchTool(str(workspace), exclude_extensions=[".txt"])
        result = await tool.execute(pattern="Nested", path="subdir")
        assert "No matches" in result

    @pytest.mark.asyncio
    async def test_custom_binary_extensions_override(self, workspace: Path) -> None:
        tool = FileSearchTool(str(workspace), binary_extensions=frozenset())
        result = await tool.execute(pattern="fake", path="subdir")
        assert (
            "image.png" in result
            or "archive.zip" in result
            or "No matches" not in result
        )

    @pytest.mark.asyncio
    async def test_include_takes_priority_over_skip(self, workspace: Path) -> None:
        tool = FileSearchTool(
            str(workspace),
            include_extensions=[".txt", ".py"],
            exclude_extensions=[".txt"],
        )
        result = await tool.execute(pattern="hello", path="subdir")
        assert "code.py" in result

    def test_default_binary_extensions_constant(self) -> None:
        assert ".png" in DEFAULT_BINARY_EXTENSIONS
        assert ".exe" in DEFAULT_BINARY_EXTENSIONS
        assert ".py" not in DEFAULT_BINARY_EXTENSIONS


# ================================================================== #
# DirectoryListTool — max_entries                                      #
# ================================================================== #


class TestDirectoryListMaxEntries:
    @pytest.mark.asyncio
    async def test_default_max_entries_is_200(self, workspace: Path) -> None:
        tool = DirectoryListTool(str(workspace))
        assert tool.max_entries == 200

    @pytest.mark.asyncio
    async def test_truncation_with_small_limit(self, workspace: Path) -> None:
        tool = DirectoryListTool(str(workspace), max_entries=2)
        result = await tool.execute(path=".", recursive=True)
        assert "Showing 2 of" in result

    @pytest.mark.asyncio
    async def test_no_truncation_when_under_limit(self, workspace: Path) -> None:
        tool = DirectoryListTool(str(workspace), max_entries=1000)
        result = await tool.execute(path=".")
        assert "Showing" not in result

    @pytest.mark.asyncio
    async def test_truncation_message_suggests_narrowing(self, workspace: Path) -> None:
        tool = DirectoryListTool(str(workspace), max_entries=1)
        result = await tool.execute(path=".")
        assert "narrow" in result.lower()


# ================================================================== #
# FileReadTool — encoding auto-detection                               #
# ================================================================== #


class TestFileExtractKeyPathToml:
    @pytest.mark.asyncio
    async def test_toml_key_path(self, workspace: Path) -> None:
        (workspace / "config.toml").write_text(
            '[server]\nhost = "0.0.0.0"\nport = 8080\n'
        )
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="config.toml", key_path="server.host")
        assert "0.0.0.0" in result

    @pytest.mark.asyncio
    async def test_toml_key_path_not_found(self, workspace: Path) -> None:
        (workspace / "config.toml").write_text("[server]\nport = 8080\n")
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="config.toml", key_path="server.missing")
        assert "not found" in result


class TestFileExtractKeyPathJsonl:
    @pytest.mark.asyncio
    async def test_jsonl_key_path_extracts_field(self, workspace: Path) -> None:
        (workspace / "events.jsonl").write_text(
            '{"event": "click", "user": "alice"}\n{"event": "view", "user": "bob"}\n'
        )
        tool = FileExtractTool(str(workspace))
        result = await tool.execute(path="events.jsonl", key_path="user")
        assert "alice" in result
        assert "bob" in result


class TestFileWriteToolEdgeCases:
    @pytest.mark.asyncio
    async def test_initialize_is_noop(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        await tool.initialize()

    @pytest.mark.asyncio
    async def test_write_empty_content(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        result = await tool.execute(path="empty.txt", content="")
        assert "Created" in result
        assert (workspace / "empty.txt").read_text() == ""

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_creates_file(self, workspace: Path) -> None:
        tool = FileWriteTool(str(workspace))
        result = await tool.execute(
            path="new_append.txt", content="first", mode="append"
        )
        assert "Appended to" in result
        assert (workspace / "new_append.txt").read_text() == "first"


class TestBuiltinExports:
    def test_file_write_tool_importable(self) -> None:
        from nucleusiq.tools.builtin import FileWriteTool

        assert FileWriteTool is not None

    def test_all_tools_in_init(self) -> None:
        from nucleusiq.tools.builtin import __all__

        assert "FileWriteTool" in __all__
        assert "FileReadTool" in __all__
        assert "FileSearchTool" in __all__
        assert "DirectoryListTool" in __all__
        assert "FileExtractTool" in __all__


class TestFileReadEncoding:
    @pytest.mark.asyncio
    async def test_auto_encoding_default(self, workspace: Path) -> None:
        tool = FileReadTool(str(workspace))
        result = await tool.execute(path="hello.txt")
        assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_explicit_utf8(self, workspace: Path) -> None:
        tool = FileReadTool(str(workspace))
        result = await tool.execute(path="hello.txt", encoding="utf-8")
        assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_auto_fallback_without_chardet(self, workspace: Path) -> None:
        tool = FileReadTool(str(workspace))
        result = await tool.execute(path="hello.txt", encoding="auto")
        assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_latin1_file_with_explicit_encoding(self, workspace: Path) -> None:
        latin_file = workspace / "latin1.txt"
        latin_file.write_bytes("caf\xe9".encode("latin-1"))
        tool = FileReadTool(str(workspace))
        result = await tool.execute(path="latin1.txt", encoding="latin-1")
        assert "caf" in result

    @pytest.mark.asyncio
    async def test_spec_encoding_default_is_auto(self, workspace: Path) -> None:
        tool = FileReadTool(str(workspace))
        spec = tool.get_spec()
        enc_prop = spec["parameters"]["properties"]["encoding"]
        assert enc_prop["default"] == "auto"
