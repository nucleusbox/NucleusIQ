"""Built-in file tools for NucleusIQ agents.

All tools inherit from ``BaseTool`` and are sandboxed to a
``workspace_root`` directory.  They plug directly into Standard
and Autonomous mode tool loops with zero extra wiring.

Usage::

    from nucleusiq.tools.builtin import (
        FileReadTool,
        FileWriteTool,
        FileSearchTool,
        DirectoryListTool,
        FileExtractTool,
    )

    agent = Agent(
        llm=llm,
        tools=[
            FileReadTool(workspace_root="./data"),
            FileWriteTool(workspace_root="./data"),
            FileSearchTool(workspace_root="./data"),
            DirectoryListTool(workspace_root="./data"),
            FileExtractTool(workspace_root="./data"),
        ],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
    )
"""

from nucleusiq.tools.builtin.directory_list import DirectoryListTool
from nucleusiq.tools.builtin.file_extract import (
    FileExtractTool,
    register_extract_format,
)
from nucleusiq.tools.builtin.file_read import FileReadTool
from nucleusiq.tools.builtin.file_search import FileSearchTool
from nucleusiq.tools.builtin.file_write import FileWriteTool
from nucleusiq.tools.builtin.workspace import (
    WorkspaceSecurityError,
    resolve_safe_path,
)

__all__ = [
    "DirectoryListTool",
    "FileExtractTool",
    "FileReadTool",
    "FileSearchTool",
    "FileWriteTool",
    "WorkspaceSecurityError",
    "register_extract_format",
    "resolve_safe_path",
]
