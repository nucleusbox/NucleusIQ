"""Workspace sandboxing for built-in file tools.

All built-in file tools resolve user-supplied paths through
``resolve_safe_path`` before performing any I/O.  This prevents
path traversal attacks (``../``), symlink escapes, and absolute
path injection.
"""

from __future__ import annotations

from pathlib import Path

from nucleusiq.errors import NucleusIQError


class WorkspaceSecurityError(NucleusIQError):
    """Raised when a path escapes the workspace sandbox.

    Attributes:
        path: The offending path that was attempted.
        workspace_root: The workspace root it tried to escape.
    """

    def __init__(
        self,
        message: str = "",
        *,
        path: str | None = None,
        workspace_root: str | None = None,
    ) -> None:
        self.path = path
        self.workspace_root = workspace_root
        super().__init__(message)


def resolve_safe_path(workspace_root: str | Path, user_path: str) -> Path:
    """Resolve *user_path* relative to *workspace_root* safely.

    Guarantees:
        * The returned path is inside *workspace_root* (after symlink
          resolution).
        * ``..`` segments, absolute paths, and symlink escapes are
          blocked.

    Returns:
        Resolved ``Path`` that is guaranteed to be under *workspace_root*.

    Raises:
        WorkspaceSecurityError: If the resolved path escapes the workspace.
    """
    root = Path(workspace_root).resolve()

    if not root.is_dir():
        raise WorkspaceSecurityError(
            f"Workspace root does not exist or is not a directory: {root}"
        )

    candidate = (root / user_path).resolve()

    try:
        candidate.relative_to(root)
    except ValueError:
        raise WorkspaceSecurityError(
            f"Path '{user_path}' resolves outside the workspace root. "
            f"Access is restricted to: {root}"
        )

    return candidate


def format_file_size(size_bytes: int) -> str:
    """Human-readable file size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"
