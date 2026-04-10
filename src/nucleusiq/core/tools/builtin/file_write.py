"""FileWriteTool — write or append content to files in the workspace.

Sandboxed to a ``workspace_root`` directory via ``resolve_safe_path``.
The tool **does not** enforce ``HumanApprovalPlugin`` — that is the
user's choice.  If they want an approval gate, they register the
plugin on the agent; this tool is unaware of the plugin layer.

Safety features:
    * Path sandbox — all writes are confined to ``workspace_root``.
    * Backup on overwrite — when ``backup=True`` (default), writes a
      ``.bak`` copy before overwriting an existing file.
    * Max file size — rejects writes whose content exceeds a
      configurable limit (default 5 MB) to prevent runaway generation.
    * Parent directory auto-creation — creates intermediate directories
      as needed (like ``mkdir -p``).
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.builtin.workspace import (
    WorkspaceSecurityError,
    format_file_size,
    resolve_safe_path,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_WRITE_SIZE = 5 * 1024 * 1024  # 5 MB


class FileWriteTool(BaseTool):
    """Write or append text content to a file in the workspace.

    Parameters accepted by ``execute()``:
        path (str): File path relative to *workspace_root*.
        content (str): Text content to write.
        mode (str, optional): ``"write"`` (default) to create/overwrite,
            ``"append"`` to add to end of file.
        encoding (str, optional): File encoding (default ``utf-8``).
        create_parents (bool, optional): Create intermediate directories
            if they don't exist (default ``True``).

    Constructor options:
        backup: Create a ``.bak`` copy before overwriting an existing
            file (default ``True``).
        max_write_size: Maximum allowed content length in bytes
            (default 5 MB).
    """

    def __init__(
        self,
        workspace_root: str,
        *,
        backup: bool = True,
        max_write_size: int = DEFAULT_MAX_WRITE_SIZE,
        name: str = "file_write",
        description: str = (
            "Write or append text content to a file. Creates parent "
            "directories automatically. Supports write and append modes."
        ),
    ) -> None:
        super().__init__(name=name, description=description)
        self.workspace_root = workspace_root
        self.backup = backup
        self.max_write_size = max_write_size

    async def initialize(self) -> None:
        return

    async def execute(self, **kwargs: Any) -> str:
        path: str = kwargs.get("path", "")
        content: str = kwargs.get("content", "")
        mode: str = kwargs.get("mode", "write")
        encoding: str = kwargs.get("encoding", "utf-8")
        create_parents: bool = kwargs.get("create_parents", True)

        if not path:
            return "Error: 'path' parameter is required."

        if mode not in ("write", "append"):
            return f"Error: 'mode' must be 'write' or 'append', got '{mode}'."

        content_bytes = len(content.encode(encoding, errors="replace"))
        if content_bytes > self.max_write_size:
            return (
                f"Error: Content size ({format_file_size(content_bytes)}) exceeds "
                f"the {format_file_size(self.max_write_size)} limit."
            )

        try:
            resolved = resolve_safe_path(self.workspace_root, path)
        except WorkspaceSecurityError as exc:
            return f"Error: {exc}"

        if create_parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        elif not resolved.parent.is_dir():
            return f"Error: Parent directory does not exist for '{path}'."

        existed = resolved.is_file()
        if existed and mode == "write" and self.backup:
            backup_path = resolved.with_suffix(resolved.suffix + ".bak")
            try:
                shutil.copy2(resolved, backup_path)
            except Exception as exc:
                logger.warning("Failed to create backup of '%s': %s", path, exc)

        try:
            if mode == "append":
                with open(resolved, "a", encoding=encoding, errors="replace") as f:
                    f.write(content)
            else:
                resolved.write_text(content, encoding=encoding, errors="replace")
        except Exception as exc:
            return f"Error writing file: {exc}"

        action = (
            "Appended to"
            if mode == "append"
            else ("Overwrote" if existed else "Created")
        )
        size = resolved.stat().st_size
        return f"{action} '{path}' ({format_file_size(size)})."

    def get_spec(self) -> dict[str, Any]:
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
                    "content": {
                        "type": "string",
                        "description": "Text content to write to the file.",
                    },
                    "mode": {
                        "type": "string",
                        "description": (
                            "'write' to create/overwrite (default), "
                            "'append' to add to end of file."
                        ),
                        "enum": ["write", "append"],
                        "default": "write",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8).",
                        "default": "utf-8",
                    },
                    "create_parents": {
                        "type": "boolean",
                        "description": (
                            "Create intermediate directories if they don't "
                            "exist (default: true)."
                        ),
                        "default": True,
                    },
                },
                "required": ["path", "content"],
            },
        }
