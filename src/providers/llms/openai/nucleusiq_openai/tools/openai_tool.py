"""
OpenAI-specific tools factory.

OpenAI supports multiple tool types:
1. Function calling - Custom code execution (BaseTool handles this)
2. Web search - Native tool
3. Remote MCP servers - Native tool (Model Context Protocol)
4. File search - Native tool (search uploaded files)
5. Image generation - Native tool (GPT Image)
6. Code interpreter - Native tool
7. Computer use - Native tool (agentic workflows)

Use OpenAITool as a factory class:
    - OpenAITool.web_search()
    - OpenAITool.code_interpreter()
    - OpenAITool.file_search(vector_store_ids=[...])
    - OpenAITool.image_generation()
    - OpenAITool.mcp(server_name="...", server_url="...")
    - OpenAITool.computer_use()
"""

from typing import Any, Dict, Optional, List, Union
from nucleusiq.tools.base_tool import BaseTool


class _OpenAINativeTool(BaseTool):
    """
    Internal class for OpenAI native tools.
    
    This is created by OpenAITool factory methods.
    """
    
    def __init__(
        self,
        *,
        name: str,
        description: str,
        tool_type: str,
        tool_spec: Dict[str, Any],
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
        Native tools don't use execute() - OpenAI handles them directly.
        This should never be called.
        """
        raise NotImplementedError(
            "OpenAI native tools don't use execute(). They're handled by OpenAI's API directly."
        )
    
    def get_spec(self) -> Dict[str, Any]:
        """
        Return the native tool spec (already in OpenAI format).
        
        This returns the spec directly since it's already in OpenAI format.
        BaseOpenAI._convert_tool_spec() will detect the "type" key and return as-is.
        """
        return self.tool_spec


class OpenAITool:
    """
    Factory class for creating OpenAI-specific tools.
    
    Usage:
        # Web search
        web_search = OpenAITool.web_search()
        
        # Code interpreter
        code_interpreter = OpenAITool.code_interpreter()
        
        # File search
        file_search = OpenAITool.file_search(vector_store_ids=["vs_123"])
        
        # Image generation
        image_gen = OpenAITool.image_generation()
        
        # MCP server
        mcp = OpenAITool.mcp(
            server_label="dmcp",
            server_description="A Dungeons and Dragons MCP server",
            server_url="https://dmcp-server.deno.dev/sse",
            require_approval="never",
        )
        
        # Computer use
        computer = OpenAITool.computer_use()
        
        # Use in Agent
        agent = Agent(
            tools=[web_search, code_interpreter, file_search],
            ...
        )
    """
    
    @staticmethod
    def web_search() -> BaseTool:
        """
        Create OpenAI's web search tool.
        
        Returns:
            BaseTool instance for web search
        """
        return _OpenAINativeTool(
            name="web_search_preview",
            description="Search the web for current information using OpenAI's built-in search",
            tool_type="web_search_preview",
            tool_spec={"type": "web_search_preview"},
        )
    
    @staticmethod
    def code_interpreter() -> BaseTool:
        """
        Create OpenAI's code interpreter tool.
        
        Returns:
            BaseTool instance for code interpreter
        """
        return _OpenAINativeTool(
            name="code_interpreter",
            description="Execute Python code in a secure container using OpenAI's built-in code interpreter",
            tool_type="code_interpreter",
            tool_spec={"type": "code_interpreter"},
        )
    
    @staticmethod
    def file_search(vector_store_ids: Optional[List[str]] = None) -> BaseTool:
        """
        Create OpenAI's file search tool.
        
        Args:
            vector_store_ids: Optional list of vector store IDs to search
        
        Returns:
            BaseTool instance for file search
        """
        tool_spec: Dict[str, Any] = {"type": "file_search"}
        if vector_store_ids:
            tool_spec["vector_store_ids"] = vector_store_ids
        
        return _OpenAINativeTool(
            name="file_search",
            description="Search the contents of uploaded files for context when generating a response",
            tool_type="file_search",
            tool_spec=tool_spec,
        )
    
    @staticmethod
    def image_generation() -> BaseTool:
        """
        Create OpenAI's image generation tool (GPT Image).
        
        Returns:
            BaseTool instance for image generation
        """
        return _OpenAINativeTool(
            name="image_generation",
            description="Generate or edit images using GPT Image",
            tool_type="image_generation",
            tool_spec={"type": "image_generation"},
        )
    
    @staticmethod
    def mcp(
        server_label: str,
        server_description: str,
        *,
        server_url: Optional[str] = None,
        connector_id: Optional[str] = None,
        require_approval: Optional[Union[str, Dict[str, Any]]] = None,
        allowed_tools: Optional[List[str]] = None,
        authorization: Optional[str] = None,
    ) -> BaseTool:
        """
        Create OpenAI's MCP (Model Context Protocol) tool.
        
        Supports both remote MCP servers and OpenAI connectors.
        
        Args:
            server_label: Label/name for the MCP server or connector (e.g., "dmcp", "google_calendar")
            server_description: Description of what the server/connector does
            server_url: URL of the remote MCP server (e.g., "https://dmcp-server.deno.dev/sse")
                       Required for remote MCP servers, not used for connectors
            connector_id: ID of the OpenAI connector (e.g., "connector_googlecalendar")
                         Required for connectors, not used for remote MCP servers
            require_approval: Approval requirement. Can be:
                - "never": Skip approvals for all tools
                - "always": Require approval for all tools
                - Dict: {"never": {"tool_names": ["tool1", "tool2"]}} - Skip approvals for specific tools
            allowed_tools: Optional list of tool names to import from the server
                          (filters tools to reduce cost and latency)
            authorization: Optional OAuth access token for authenticated servers/connectors
        
        Returns:
            BaseTool instance for MCP server or connector
        
        Examples:
            # Remote MCP server
            mcp_tool = OpenAITool.mcp(
                server_label="dmcp",
                server_description="A Dungeons and Dragons MCP server to assist with dice rolling.",
                server_url="https://dmcp-server.deno.dev/sse",
                require_approval="never",
            )
            
            # With tool filtering
            mcp_tool = OpenAITool.mcp(
                server_label="dmcp",
                server_description="D&D dice rolling server.",
                server_url="https://dmcp-server.deno.dev/sse",
                require_approval="never",
                allowed_tools=["roll"],  # Only import the "roll" tool
            )
            
            # With OAuth authentication
            mcp_tool = OpenAITool.mcp(
                server_label="stripe",
                server_description="Stripe payment processing.",
                server_url="https://mcp.stripe.com",
                authorization="$STRIPE_OAUTH_ACCESS_TOKEN",
            )
            
            # OpenAI Connector (Google Calendar)
            calendar_connector = OpenAITool.mcp(
                server_label="google_calendar",
                server_description="Access Google Calendar events.",
                connector_id="connector_googlecalendar",
                authorization="ya29.A0AS3H6...",
                require_approval="never",
            )
        """
        if not server_url and not connector_id:
            raise ValueError("Either server_url (for remote MCP) or connector_id (for connector) must be provided")
        if server_url and connector_id:
            raise ValueError("Cannot specify both server_url and connector_id. Use one or the other.")
        
        tool_spec: Dict[str, Any] = {
            "type": "mcp",
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
            tool_type="mcp",
            tool_spec=tool_spec,
        )
    
    @staticmethod
    def connector(
        connector_id: str,
        server_label: str,
        server_description: str,
        *,
        authorization: str,
        require_approval: Optional[Union[str, Dict[str, Any]]] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> BaseTool:
        """
        Convenience method for creating OpenAI connector tools.
        
        This is a shortcut for OpenAITool.mcp() with connector_id.
        
        Available connectors:
        - connector_dropbox
        - connector_gmail
        - connector_googlecalendar
        - connector_googledrive
        - connector_microsoftteams
        - connector_outlookcalendar
        - connector_outlookemail
        - connector_sharepoint
        
        Args:
            connector_id: ID of the OpenAI connector (e.g., "connector_googlecalendar")
            server_label: Label/name for the connector (e.g., "google_calendar")
            server_description: Description of what the connector does
            authorization: OAuth access token (required for connectors)
            require_approval: Approval requirement ("never", "always", or dict)
            allowed_tools: Optional list of tool names to import
        
        Returns:
            BaseTool instance for the connector
        
        Example:
            calendar = OpenAITool.connector(
                connector_id="connector_googlecalendar",
                server_label="google_calendar",
                server_description="Access Google Calendar events.",
                authorization="ya29.A0AS3H6...",
                require_approval="never",
            )
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
            BaseTool instance for computer use
        """
        return _OpenAINativeTool(
            name="computer_use",
            description="Create agentic workflows that enable a model to control a computer interface",
            tool_type="computer_use",
            tool_spec={"type": "computer_use"},
        )

