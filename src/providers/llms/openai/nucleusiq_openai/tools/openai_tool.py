"""
OpenAI-specific tools factory.

OpenAI supports multiple tool types through the Responses API:
1. Function calling  - Custom code execution (BaseTool handles this)
2. Web search        - Native tool  (web_search_preview)
3. Code interpreter  - Native tool
4. File search       - Native tool  (search uploaded files)
5. Image generation  - Native tool  (GPT Image)
6. Remote MCP        - Native tool  (Model Context Protocol)
7. Computer use      - Native tool  (computer_use_preview)

Usage — create via the OpenAITool factory and pass to an Agent:
    web   = OpenAITool.web_search()
    code  = OpenAITool.code_interpreter()
    agent = Agent(tools=[web, code], ...)

Tool type identifiers are stored as class-level constants on OpenAITool
so they can be overridden in one place when OpenAI ships new type strings:
    OpenAITool.WEB_SEARCH_TYPE = "web_search_preview_2025_09_01"
"""

from __future__ import annotations

from typing import Any

from nucleusiq.tools.base_tool import BaseTool

# ---------------------------------------------------------------------------
# Registry of native tool types that require the Responses API.
# BaseOpenAI._has_native_tools() uses this set to decide which backend to use.
# ---------------------------------------------------------------------------
NATIVE_TOOL_TYPES: frozenset = frozenset(
    {
        # Web search — preview names used by OpenAI
        "web_search_preview",
        "web_search_preview_2025_03_11",
        # Code interpreter
        "code_interpreter",
        # File search (vector stores)
        "file_search",
        # Image generation (GPT Image)
        "image_generation",
        # Model Context Protocol
        "mcp",
        # Computer use — preview names used by OpenAI
        "computer_use_preview",
        "computer_use_preview_2025_03_11",
    }
)


class _OpenAINativeTool(BaseTool):
    """
    Internal wrapper around an OpenAI native tool specification.

    Created exclusively by :class:`OpenAITool` factory methods.  Native tools
    are executed **server-side** by OpenAI's Responses API — the local
    :class:`Executor` never calls ``execute()`` on them.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        tool_type: str,
        tool_spec: dict[str, Any],
    ):
        super().__init__(name=name, description=description, version=None)
        self.tool_type = tool_type
        self.tool_spec = tool_spec
        self.is_native = True

    async def initialize(self) -> None:
        """No initialization needed for native tools."""
        pass

    async def execute(self, **kwargs: Any) -> Any:
        """
        Native tools are executed server-side by OpenAI.

        Raising here protects against accidental local invocation.
        """
        raise NotImplementedError(
            f"OpenAI native tool '{self.name}' (type={self.tool_type}) is "
            f"executed server-side by OpenAI's Responses API.  "
            f"It must not be called via Executor.execute()."
        )

    def get_spec(self) -> dict[str, Any]:
        """
        Return the native tool spec — already in OpenAI Responses API format.

        ``BaseOpenAI._convert_tool_spec()`` detects the ``"type"`` key and
        returns the spec as-is (pass-through).
        """
        return self.tool_spec


class OpenAITool:
    """
    Factory class for creating OpenAI-specific tools.

    **Adaptive design** — tool type identifiers live as class attributes so
    they act as a single source of truth.  When OpenAI changes a type string
    (e.g. ``web_search_preview`` → ``web_search``), update the constant here
    and every tool created afterward picks up the change automatically.

    Usage::

        web_search       = OpenAITool.web_search()
        code_interpreter = OpenAITool.code_interpreter()
        file_search      = OpenAITool.file_search(vector_store_ids=["vs_123"])
        image_gen        = OpenAITool.image_generation()
        mcp              = OpenAITool.mcp(server_label="dmcp", ...)
        computer         = OpenAITool.computer_use()

        agent = Agent(tools=[web_search, code_interpreter], ...)
    """

    # ------------------------------------------------------------------ #
    # Tool type identifiers — single source of truth.                     #
    # Override these class attrs when OpenAI ships new type strings.       #
    # ------------------------------------------------------------------ #
    WEB_SEARCH_TYPE: str = "web_search_preview"
    CODE_INTERPRETER_TYPE: str = "code_interpreter"
    FILE_SEARCH_TYPE: str = "file_search"
    IMAGE_GENERATION_TYPE: str = "image_generation"
    MCP_TYPE: str = "mcp"
    COMPUTER_USE_TYPE: str = "computer_use_preview"

    # -------------------------------------------------------------- #
    # Factory methods                                                  #
    # -------------------------------------------------------------- #

    @staticmethod
    def web_search() -> BaseTool:
        """
        Create OpenAI's web search tool.

        Returns:
            BaseTool instance whose spec uses the Responses API.
        """
        return _OpenAINativeTool(
            name="web_search",
            description="Search the web for current information using OpenAI's built-in search",
            tool_type=OpenAITool.WEB_SEARCH_TYPE,
            tool_spec={"type": OpenAITool.WEB_SEARCH_TYPE},
        )

    @staticmethod
    def code_interpreter(
        *,
        container: dict[str, Any] | None = None,
    ) -> BaseTool:
        """
        Create OpenAI's code interpreter tool.

        Args:
            container: Container configuration for the sandbox.
                Defaults to ``{"type": "auto"}`` which lets OpenAI
                manage container lifecycle automatically.
                Pass ``{"type": "auto", "memory_limit": "4g"}`` for
                a higher memory limit, or a pre-created container ID.

        Returns:
            BaseTool instance for code interpreter.
        """
        spec: dict[str, Any] = {
            "type": OpenAITool.CODE_INTERPRETER_TYPE,
            "container": container or {"type": "auto"},
        }
        return _OpenAINativeTool(
            name="code_interpreter",
            description="Execute Python code in a secure container using OpenAI's built-in code interpreter",
            tool_type=OpenAITool.CODE_INTERPRETER_TYPE,
            tool_spec=spec,
        )

    @staticmethod
    def file_search(vector_store_ids: list[str] | None = None) -> BaseTool:
        """
        Create OpenAI's file search tool.

        Args:
            vector_store_ids: Optional list of vector store IDs to search.

        Returns:
            BaseTool instance for file search.
        """
        tool_spec: dict[str, Any] = {"type": OpenAITool.FILE_SEARCH_TYPE}
        if vector_store_ids:
            tool_spec["vector_store_ids"] = vector_store_ids

        return _OpenAINativeTool(
            name="file_search",
            description="Search the contents of uploaded files for context when generating a response",
            tool_type=OpenAITool.FILE_SEARCH_TYPE,
            tool_spec=tool_spec,
        )

    @staticmethod
    def image_generation() -> BaseTool:
        """
        Create OpenAI's image generation tool (GPT Image).

        Returns:
            BaseTool instance for image generation.
        """
        return _OpenAINativeTool(
            name="image_generation",
            description="Generate or edit images using GPT Image",
            tool_type=OpenAITool.IMAGE_GENERATION_TYPE,
            tool_spec={"type": OpenAITool.IMAGE_GENERATION_TYPE},
        )

    @staticmethod
    def mcp(
        server_label: str,
        server_description: str,
        *,
        server_url: str | None = None,
        connector_id: str | None = None,
        require_approval: str | dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        authorization: str | None = None,
    ) -> BaseTool:
        """
        Create OpenAI's MCP (Model Context Protocol) tool.

        Supports both remote MCP servers and OpenAI connectors.

        Args:
            server_label: Label/name for the MCP server or connector
            server_description: Description of what the server/connector does
            server_url: URL of the remote MCP server — required for remote MCP
            connector_id: ID of the OpenAI connector — required for connectors
            require_approval: ``"never"`` | ``"always"`` | dict with per-tool settings
            allowed_tools: Tool names to import (filters for cost/latency)
            authorization: OAuth access token for authenticated servers

        Returns:
            BaseTool instance for MCP server or connector.

        Examples::

            # Remote MCP server
            mcp_tool = OpenAITool.mcp(
                server_label="dmcp",
                server_description="A D&D MCP server for dice rolling.",
                server_url="https://dmcp-server.deno.dev/sse",
                require_approval="never",
            )

            # OpenAI Connector
            calendar = OpenAITool.mcp(
                server_label="google_calendar",
                server_description="Access Google Calendar events.",
                connector_id="connector_googlecalendar",
                authorization="ya29.A0AS3H6...",
                require_approval="never",
            )
        """
        if not server_url and not connector_id:
            raise ValueError(
                "Either server_url (for remote MCP) or connector_id "
                "(for connector) must be provided"
            )
        if server_url and connector_id:
            raise ValueError(
                "Cannot specify both server_url and connector_id. Use one or the other."
            )

        tool_spec: dict[str, Any] = {
            "type": OpenAITool.MCP_TYPE,
            "server_label": server_label,
            "server_description": server_description,
        }

        if server_url:
            tool_spec["server_url"] = server_url
        if connector_id:
            tool_spec["connector_id"] = connector_id
        if require_approval is not None:
            tool_spec["require_approval"] = require_approval
        if allowed_tools:
            tool_spec["allowed_tools"] = allowed_tools
        if authorization:
            tool_spec["authorization"] = authorization

        return _OpenAINativeTool(
            name=f"mcp_{server_label}",
            description=server_description,
            tool_type=OpenAITool.MCP_TYPE,
            tool_spec=tool_spec,
        )

    @staticmethod
    def connector(
        connector_id: str,
        server_label: str,
        server_description: str,
        *,
        authorization: str,
        require_approval: str | dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> BaseTool:
        """
        Convenience method for creating OpenAI connector tools.

        This is a shortcut for ``OpenAITool.mcp()`` with *connector_id*.

        Available connectors::

            (
                connector_dropbox,
                connector_gmail,
                connector_googlecalendar,
            )
            (
                connector_googledrive,
                connector_microsoftteams,
            )
            connector_outlookcalendar, connector_outlookemail, connector_sharepoint

        Args:
            connector_id: ID of the OpenAI connector
            server_label: Label/name for the connector
            server_description: Description of what the connector does
            authorization: OAuth access token (required)
            require_approval: ``"never"`` | ``"always"`` | dict
            allowed_tools: Tool names to import

        Returns:
            BaseTool instance for the connector.
        """
        return OpenAITool.mcp(
            server_label=server_label,
            server_description=server_description,
            connector_id=connector_id,
            authorization=authorization,
            require_approval=require_approval,
            allowed_tools=allowed_tools,
        )

    @staticmethod
    def computer_use() -> BaseTool:
        """
        Create OpenAI's computer use tool for agentic workflows.

        Returns:
            BaseTool instance for computer use.
        """
        return _OpenAINativeTool(
            name="computer_use",
            description="Create agentic workflows that enable a model to control a computer interface",
            tool_type=OpenAITool.COMPUTER_USE_TYPE,
            tool_spec={"type": OpenAITool.COMPUTER_USE_TYPE},
        )
