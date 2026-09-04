"""``nucleusiq-mcp`` — Model Context Protocol adapter for NucleusIQ.

Quick start::

    from nucleusiq.agents.agent import Agent
    from nucleusiq_mcp import MCPTool

    agent = Agent(
        name="researcher",
        llm=llm,
        prompt=prompt,
        tools=[
            MCPTool("https://mcp.slack.com/api", auth="xoxb-..."),
            MCPTool("npx -y @modelcontextprotocol/server-github"),
        ],
    )
    await agent.initialize()

See ``docs/design/MCP_INTEGRATION_DESIGN.md`` for the full design.
"""

from __future__ import annotations

from nucleusiq_mcp.auth import (
    BearerAuth,
    CustomHeadersAuth,
    EnvAuth,
    MCPAuth,
    OAuthAuth,
    build_auth,
)
from nucleusiq_mcp.bound_tool import MCPBoundTool, format_result
from nucleusiq_mcp.config import MCPServerConfig, MCPTransport, infer_transport
from nucleusiq_mcp.decorators import MCPToolset, mcp_tool_filter
from nucleusiq_mcp.exceptions import (
    MCPAuthError,
    MCPConnectionError,
    MCPError,
    MCPProtocolError,
    MCPTimeoutError,
    MCPToolError,
)
from nucleusiq_mcp.mcp_tool import MCPTool
from nucleusiq_mcp.models import MCPContent, MCPToolResult, MCPToolSpec
from nucleusiq_mcp.schema_adapter import MCPSchemaAdapter
from nucleusiq_mcp.session import MCPSession

__version__ = "0.1.1"

__all__ = [
    "__version__",
    # ---- Public entry point ----
    "MCPTool",
    # ---- Auth strategies ----
    "MCPAuth",
    "BearerAuth",
    "CustomHeadersAuth",
    "EnvAuth",
    "OAuthAuth",
    "build_auth",
    # ---- Configuration ----
    "MCPServerConfig",
    "MCPTransport",
    "infer_transport",
    # ---- Lower-level (advanced) ----
    "MCPSession",
    "MCPBoundTool",
    "MCPSchemaAdapter",
    "format_result",
    # ---- Domain models ----
    "MCPContent",
    "MCPToolResult",
    "MCPToolSpec",
    # ---- Decorator-style filters ----
    "MCPToolset",
    "mcp_tool_filter",
    # ---- Exceptions ----
    "MCPError",
    "MCPConnectionError",
    "MCPAuthError",
    "MCPTimeoutError",
    "MCPProtocolError",
    "MCPToolError",
]
