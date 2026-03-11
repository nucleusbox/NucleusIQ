"""DirectoryListTool — list directory contents with glob filtering.

Sandboxed to a ``workspace_root`` directory.  Returns file/directory
entries with sizes so the LLM can decide what to read or search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.builtin.workspace import (
    WorkspaceSecurityError,
    format_file_size,
    resolve_safe_path,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 200


class DirectoryListTool(BaseTool):
    """List files and directories with optional glob filtering.

    Parameters accepted by ``execute()``:
        path (str): Directory path relative to *workspace_root*.
            Defaults to ``"."`` (workspace root).
        pattern (str, optional): Glob pattern to filter entries
            (default ``"*"``).
        recursive (bool, optional): Search recursively (default ``False``).

    Constructor options:
        max_entries: Maximum number of entries to return (default 200).
            Prevents huge output on large directory trees that would
            waste LLM context tokens.
    """

    def __init__(
        self,
        workspace_root: str,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        name: str = "directory_list",
        description: str = (
            "List files and directories at a given path. Supports glob "
            "pattern filtering and recursive listing."
        ),
    ) -> None:
        super().__init__(name=name, description=description)
        self.workspace_root = workspace_root
        self.max_entries = max_entries

    async def initialize(self) -> None:
        return

    async def execute(self, **kwargs: Any) -> str:
        path: str = kwargs.get("path", ".")
        pattern: str = kwargs.get("pattern", "*")
        recursive: bool = kwargs.get("recursive", False)

        try:
            resolved = resolve_safe_path(self.workspace_root, path)
        except WorkspaceSecurityError as exc:
            return f"Error: {exc}"

        if not resolved.is_dir():
            return f"Error: '{path}' is not a directory or does not exist."

        root = resolve_safe_path(self.workspace_root, ".")

        if recursive:
            entries = sorted(resolved.rglob(pattern))
        else:
            entries = sorted(resolved.glob(pattern))

        safe_entries: list[Path] = []
        for entry in entries:
            try:
                entry.resolve().relative_to(root)
                safe_entries.append(entry)
            except ValueError:
                continue

        if not safe_entries:
            return f"No entries matching '{pattern}' in {path}."

        total_found = len(safe_entries)
        truncated = total_found > self.max_entries
        if truncated:
            safe_entries = safe_entries[: self.max_entries]

        lines: list[str] = []
        dir_count = 0
        file_count = 0

        for entry in safe_entries:
            try:
                rel = entry.relative_to(root)
            except ValueError:
                rel = entry.name

            if entry.is_dir():
                lines.append(f"  [DIR]  {rel}/")
                dir_count += 1
            else:
                size = format_file_size(entry.stat().st_size)
                lines.append(f"  [FILE] {rel}  ({size})")
                file_count += 1

        header = f"Directory: {path} | {file_count} file(s), {dir_count} dir(s)"
        if recursive:
            header += " (recursive)"
        header += "\n"

        result = header + "\n".join(lines)
        if truncated:
            result += (
                f"\n\n--- Showing {self.max_entries} of {total_found} entries. "
                f"Use a more specific path or pattern to narrow results. ---"
            )
        return result

    def get_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory path relative to workspace root. "
                            "Defaults to '.' (workspace root)."
                        ),
                        "default": ".",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter entries (default: '*').",
                        "default": "*",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Search recursively through subdirectories (default: false).",
                        "default": False,
                    },
                },
                "required": [],
            },
        }
