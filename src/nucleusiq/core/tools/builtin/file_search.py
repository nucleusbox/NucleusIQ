"""FileSearchTool — search text or regex patterns in files/directories.

Sandboxed to a ``workspace_root`` directory.  Returns matching lines
with ``file:line_number`` references so the LLM can request specific
ranges via ``FileReadTool``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict

from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.builtin.workspace import (
    WorkspaceSecurityError,
    resolve_safe_path,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 50


class FileSearchTool(BaseTool):
    """Search for text or regex patterns inside files.

    Parameters accepted by ``execute()``:
        pattern (str): Text or regex pattern to search for.
        path (str): File or directory path relative to *workspace_root*.
            Defaults to ``"."`` (entire workspace).
        regex (bool, optional): Treat *pattern* as a regex (default ``False``).
        max_results (int, optional): Maximum matches to return (default 50).
    """

    def __init__(
        self,
        workspace_root: str,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        name: str = "file_search",
        description: str = (
            "Search for a text or regex pattern inside files. "
            "Returns matching lines with file path and line number references."
        ),
    ) -> None:
        super().__init__(name=name, description=description)
        self.workspace_root = workspace_root
        self.max_results = max_results

    async def initialize(self) -> None:
        return

    async def execute(self, **kwargs: Any) -> str:
        pattern: str = kwargs.get("pattern", "")
        path: str = kwargs.get("path", ".")
        use_regex: bool = kwargs.get("regex", False)
        max_results: int = kwargs.get("max_results", self.max_results)

        if not pattern:
            return "Error: 'pattern' parameter is required."

        try:
            resolved = resolve_safe_path(self.workspace_root, path)
        except WorkspaceSecurityError as exc:
            return f"Error: {exc}"

        if use_regex:
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                return f"Error: Invalid regex pattern: {exc}"
        else:
            compiled = None

        matches: list[str] = []

        if resolved.is_file():
            self._search_file(resolved, pattern, compiled, matches, max_results)
        elif resolved.is_dir():
            self._search_directory(resolved, pattern, compiled, matches, max_results)
        else:
            return f"Error: '{path}' does not exist."

        if not matches:
            return f"No matches found for '{pattern}' in {path}."

        header = f"Found {len(matches)} match(es) for '{pattern}':\n\n"
        truncation = ""
        if len(matches) >= max_results:
            truncation = f"\n--- Results capped at {max_results}. Narrow your search or path. ---\n"

        return header + "\n".join(matches) + truncation

    def _search_file(
        self,
        file_path,
        pattern: str,
        compiled: re.Pattern | None,
        matches: list[str],
        limit: int,
    ) -> None:
        root = resolve_safe_path(self.workspace_root, ".")
        try:
            rel = file_path.relative_to(root)
        except ValueError:
            rel = file_path.name

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        for line_no, line in enumerate(text.splitlines(), start=1):
            if len(matches) >= limit:
                return
            hit = compiled.search(line) if compiled else (pattern in line)
            if hit:
                matches.append(f"  {rel}:{line_no}  {line.rstrip()}")

    def _search_directory(
        self,
        dir_path,
        pattern: str,
        compiled: re.Pattern | None,
        matches: list[str],
        limit: int,
    ) -> None:
        _BINARY_EXTENSIONS = frozenset(
            {
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".bmp",
                ".ico",
                ".pdf",
                ".zip",
                ".tar",
                ".gz",
                ".7z",
                ".rar",
                ".exe",
                ".dll",
                ".so",
                ".dylib",
                ".bin",
                ".mp3",
                ".mp4",
                ".wav",
                ".avi",
                ".mov",
                ".woff",
                ".woff2",
                ".ttf",
                ".eot",
                ".pyc",
                ".pyo",
                ".class",
                ".o",
            }
        )
        for dirpath, _dirnames, filenames in os.walk(dir_path):
            for fname in sorted(filenames):
                if len(matches) >= limit:
                    return
                fpath = resolve_safe_path(
                    self.workspace_root,
                    os.path.join(
                        os.path.relpath(
                            dirpath, resolve_safe_path(self.workspace_root, ".")
                        ),
                        fname,
                    ),
                )
                if fpath.suffix.lower() in _BINARY_EXTENSIONS:
                    continue
                self._search_file(fpath, pattern, compiled, matches, limit)

    def get_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "File or directory path relative to workspace root. "
                            "Defaults to '.' (entire workspace)."
                        ),
                        "default": ".",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Treat pattern as a regular expression (default: false).",
                        "default": False,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return (default: 50).",
                        "default": 50,
                    },
                },
                "required": ["pattern"],
            },
        }
