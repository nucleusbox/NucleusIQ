# nucleusiq-mcp

[![PyPI version](https://img.shields.io/pypi/v/nucleusiq-mcp?color=brightgreen)](https://pypi.org/project/nucleusiq-mcp/)
[![PyPI downloads](https://img.shields.io/pypi/dm/nucleusiq-mcp?label=downloads%2Fmonth)](https://pypistats.org/packages/nucleusiq-mcp)
[![Python versions](https://img.shields.io/pypi/pyversions/nucleusiq-mcp)](https://pypi.org/project/nucleusiq-mcp/)

**Model Context Protocol (MCP) adapter for the [NucleusIQ](https://github.com/nucleusbox/NucleusIQ) AI agent framework.**

Connect any MCP server (stdio, Streamable HTTP, SSE) as a NucleusIQ
`BaseTool` with one line. Built on the official
[`mcp`](https://pypi.org/project/mcp/) Python SDK.

## Status

**`0.1.1`** — **Development Status :: 5 - Production/Stable**. Security floor
`mcp>=1.28.1`. The public API (`MCPTool`, `MCPAuth`, exceptions) is under
semver; lower-level types may still evolve before `1.0.0`. Floor
`nucleusiq>=0.7.12`. **235 unit tests passing.**

## Install

```bash
pip install nucleusiq-mcp
```

This pulls in `nucleusiq` (core) and `mcp` (>=1.28.1, <2) automatically.

## Quick start

```python
import asyncio
from nucleusiq.agents.agent import Agent
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq_mcp import MCPTool, EnvAuth

async def main():
    agent = Agent(
        name="researcher",
        prompt=ZeroShotPrompt().configure(system="You are a researcher."),
        llm=...,  # any NucleusIQ LLM provider
        tools=[
            # Remote HTTP server with a bearer token
            MCPTool("https://mcp.slack.com/api", auth="xoxb-..."),

            # Local stdio server (e.g. GitHub MCP)
            MCPTool(
                "npx -y @modelcontextprotocol/server-github",
                env={"GITHUB_TOKEN": "ghp_..."},
            ),

            # Env-based bearer
            MCPTool("https://mcp.example.com/api", auth=EnvAuth("MY_TOKEN")),
        ],
    )
    await agent.initialize()
    # All MCP tools are now discoverable by the LLM as regular tools.

asyncio.run(main())
```

## Supported transports

| Transport         | When to use                                | Auto-detected from |
|-------------------|--------------------------------------------|--------------------|
| `STDIO`           | Local subprocess via stdin/stdout JSON-RPC | non-URL strings    |
| `STREAMABLE_HTTP` | Modern HTTP transport (default for URLs)   | `http(s)://` URLs  |
| `SSE`             | Legacy SSE transport (opt-in)              | never (explicit only) |

```python
from nucleusiq_mcp import MCPTool, MCPTransport
MCPTool("https://x.com/sse", transport=MCPTransport.SSE)
```

## Authentication

All four strategies satisfy a common `MCPAuth` protocol:

```python
from nucleusiq_mcp import (
    BearerAuth,         # static bearer token
    EnvAuth,            # lazily read from env (rotation-friendly)
    CustomHeadersAuth,  # any headers (X-API-Key, mTLS proxies, ...)
    OAuthAuth,          # OAuth 2.1 + PKCE via the official mcp SDK
)
```

Auth coercion sugar:

```python
MCPTool("...", auth="raw-bearer-token")           # → BearerAuth
MCPTool("...", auth={"X-API-Key": "secret"})      # → CustomHeadersAuth
MCPTool("...", auth=EnvAuth("MY_TOKEN"))          # explicit
MCPTool("...", auth=OAuthAuth(provider=...))      # full OAuth flow
```

## Filtering & renaming

```python
from nucleusiq_mcp import MCPTool, mcp_tool_filter, MCPToolset

@mcp_tool_filter(name="read_only")
def read_only(spec):
    """Only read-only tools."""
    return spec.name.startswith(("get_", "list_", "search_"))

agent.tools = [
    MCPTool(
        "https://mcp.example.com/api",
        auth="...",
        include_tools=["search", "summarize"],   # whitelist
        exclude_tools=["delete_user"],           # blacklist
        tool_filter=read_only,                   # predicate (or MCPToolset)
        rename={"search": "find"},               # static rename map
        prefix="ex",                             # → "ex_find", ...
        on_collision="auto_prefix",              # "auto_prefix" | "skip" | "raise"
    )
]
```

## Error handling

```python
from nucleusiq_mcp import (
    MCPError,           # base for all adapter errors
    MCPConnectionError, # transport / network failure
    MCPAuthError,       # 401 / 403 / OAuth failure
    MCPTimeoutError,    # RPC exceeded timeout
    MCPProtocolError,   # malformed server response
    MCPToolError,       # server returned isError=true (= ToolExecutionError)
)
```

All inherit from `nucleusiq.tools.ToolError`, so existing
`except ToolError` blocks continue to work.

## Retries

`call_tool` retries up to `max_retries` (default 2) on transient errors
(429, 503, timeouts, connection resets) using NucleusIQ's shared rate-limit
backoff policy (`nucleusiq.llms.retry_policy`). This means MCP retries are
observable through the same telemetry as LLM provider retries.

```python
from nucleusiq_mcp import MCPSession, MCPServerConfig

# Lower-level usage:
cfg = MCPServerConfig.build("https://...", auth="...")
session = MCPSession(cfg, max_retries=5)
async with session:
    tools = await session.list_tools()
    result = await session.call_tool("search", {"q": "rust"})
```

## SOLID alignment

| Principle | Realisation |
|-----------|-------------|
| **SRP** | `MCPServerConfig` (config only), `MCPSession` (RPC only), `MCPBoundTool` (BaseTool adapter only), `MCPSchemaAdapter` (schema translation only) |
| **OCP** | New transports → enum value + branch in `_open_transport`. New auth → new `MCPAuth` implementation. New filters → callable / `MCPToolset` composition. |
| **LSP** | `MCPTool` is a perfect `ExpandableTool`; `MCPBoundTool` is a perfect `BaseTool`; `MCPToolError` is a perfect `ToolExecutionError`. |
| **ISP** | `ExpandableTool` has 3 methods; `MCPAuth` has 2. No fat interfaces. |
| **DIP** | Session depends on `MCPAuth` (protocol), not concrete strategies. `MCPTool` depends on the `ExpandableTool` protocol for core integration. |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/unit
```

Current coverage: **99%** (target: ≥90%).

## License

MIT.
