"""FileReadTool — read file content with optional line ranges.

Sandboxed to a ``workspace_root`` directory.  When the file exceeds
``max_lines`` and no range is given, the tool returns the first
``max_lines`` lines plus a guidance message telling the LLM to request
specific ranges.

Safety features:
    * **Binary detection** — checks the first 512 bytes for null chars
      and refuses to read binary files (returns a warning instead).
    * **Max file size** — configurable upper limit (default 10 MB) to
      prevent reading huge files into memory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.builtin.workspace import (
    WorkspaceSecurityError,
    format_file_size,
    resolve_safe_path,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_LINES = 500
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
_BINARY_CHECK_BYTES = 512


def _is_binary(file_path: Any) -> bool:
    """Return ``True`` if the first 512 bytes of *file_path* contain null chars."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(_BINARY_CHECK_BYTES)
        return b"\x00" in chunk
    except Exception:
        return False


class FileReadTool(BaseTool):
    """Read file content, optionally restricting to a line range.

    Parameters accepted by ``execute()``:
        path (str): File path relative to *workspace_root*.
        start_line (int, optional): 1-based first line to read.
        end_line (int, optional): 1-based last line to read (inclusive).
        encoding (str, optional): File encoding (default ``utf-8``).
    """

    def __init__(
        self,
        workspace_root: str,
        *,
        max_lines: int = DEFAULT_MAX_LINES,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        name: str = "file_read",
        description: str = (
            "Read the contents of a file. Supports optional start_line/end_line "
            "parameters (1-based, inclusive) for reading specific sections of large files."
        ),
    ) -> None:
        super().__init__(name=name, description=description)
        self.workspace_root = workspace_root
        self.max_lines = max_lines
        self.max_file_size = max_file_size

    async def initialize(self) -> None:
        return

    async def execute(self, **kwargs: Any) -> str:
        path: str = kwargs.get("path", "")
        start_line: int | None = kwargs.get("start_line")
        end_line: int | None = kwargs.get("end_line")
        encoding: str = kwargs.get("encoding", "utf-8")

        if not path:
            return "Error: 'path' parameter is required."

        try:
            resolved = resolve_safe_path(self.workspace_root, path)
        except WorkspaceSecurityError as exc:
            return f"Error: {exc}"

        if not resolved.is_file():
            return f"Error: '{path}' is not a file or does not exist."

        size = resolved.stat().st_size

        if size > self.max_file_size:
            return (
                f"Error: '{path}' is {format_file_size(size)}, exceeding the "
                f"{format_file_size(self.max_file_size)} limit. Use FileSearchTool "
                f"to find relevant sections, then read with start_line/end_line."
            )

        if _is_binary(resolved):
            return (
                f"Error: '{path}' appears to be a binary file "
                f"({format_file_size(size)}). FileReadTool only reads text files."
            )

        try:
            text = resolved.read_text(encoding=encoding, errors="replace")
        except Exception as exc:
            return f"Error reading file: {exc}"

        lines = text.splitlines(keepends=True)
        total = len(lines)

        if start_line is not None or end_line is not None:
            s = max((start_line or 1) - 1, 0)
            e = min(end_line or total, total)
            if s >= e:
                return f"Error: Invalid line range ({start_line}-{end_line}). File has {total} lines."
            selected = lines[s:e]
            header = (
                f"File: {path} | Lines {s + 1}-{e} of {total} "
                f"| Size: {format_file_size(size)}\n"
            )
            numbered = [f"{s + i + 1:>6}| {ln}" for i, ln in enumerate(selected)]
            return header + "".join(numbered)

        if total <= self.max_lines:
            header = f"File: {path} | {total} lines | Size: {format_file_size(size)}\n"
            return header + text

        truncated = lines[: self.max_lines]
        header = (
            f"File: {path} | Showing first {self.max_lines} of {total} lines "
            f"| Size: {format_file_size(size)}\n"
        )
        guidance = (
            f"\n--- Truncated (file has {total} total lines). "
            f"Use start_line/end_line to read specific sections. ---\n"
        )
        return header + "".join(truncated) + guidance

    def get_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-based first line to read (optional).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-based last line to read, inclusive (optional).",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8).",
                        "default": "utf-8",
                    },
                },
                "required": ["path"],
            },
        }
