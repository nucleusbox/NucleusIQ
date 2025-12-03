"""
Tests for OpenAI Tools (OpenAITool factory methods).

Tests cover:
- All OpenAI native tool types (web_search, code_interpreter, file_search, etc.)
- MCP tool creation and validation
- Connector creation
- Tool spec generation
"""

import pytest
from typing import Dict, Any, List
from nucleusiq.providers.llms.openai.tools import OpenAITool
from nucleusiq.core.tools import BaseTool


class TestOpenAIToolFactory:
    """Test OpenAITool factory methods for creating native tools."""
    
    def test_web_search_tool(self):
        """Test creating web search tool."""
        tool = OpenAITool.web_search()
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "web_search_preview"
        assert hasattr(tool, 'is_native')
        assert tool.is_native is True
        
        spec = tool.get_spec()
        assert spec == {"type": "web_search_preview"}
    
    def test_code_interpreter_tool(self):
        """Test creating code interpreter tool."""
        tool = OpenAITool.code_interpreter()
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "code_interpreter"
        assert tool.is_native is True
        
        spec = tool.get_spec()
        assert spec == {"type": "code_interpreter"}
    
    def test_file_search_tool_no_vector_stores(self):
        """Test creating file search tool without vector stores."""
        tool = OpenAITool.file_search()
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "file_search"
        assert tool.is_native is True
        
        spec = tool.get_spec()
        assert spec == {"type": "file_search"}
    
    def test_file_search_tool_with_vector_stores(self):
        """Test creating file search tool with vector stores."""
        vector_store_ids = ["vs_123", "vs_456"]
        tool = OpenAITool.file_search(vector_store_ids=vector_store_ids)
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "file_search"
        
        spec = tool.get_spec()
        assert spec == {
            "type": "file_search",
            "vector_store_ids": vector_store_ids
        }
    
    def test_image_generation_tool(self):
        """Test creating image generation tool."""
        tool = OpenAITool.image_generation()
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "image_generation"
        assert tool.is_native is True
        
        spec = tool.get_spec()
        assert spec == {"type": "image_generation"}
    
    def test_computer_use_tool(self):
        """Test creating computer use tool."""
        tool = OpenAITool.computer_use()
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "computer_use"
        assert tool.is_native is True
        
        spec = tool.get_spec()
        assert spec == {"type": "computer_use"}


class TestMCPTool:
    """Test MCP tool creation and validation."""
    
    def test_mcp_tool_remote_server(self):
        """Test creating MCP tool for remote server."""
        tool = OpenAITool.mcp(
            server_label="dmcp",
            server_description="A D&D dice rolling server",
            server_url="https://dmcp-server.deno.dev/sse",
        )
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "mcp_dmcp"
        assert tool.is_native is True
        
        spec = tool.get_spec()
        assert spec == {
            "type": "mcp",
            "server_label": "dmcp",
            "server_description": "A D&D dice rolling server",
            "server_url": "https://dmcp-server.deno.dev/sse",
        }
    
    def test_mcp_tool_with_require_approval_never(self):
        """Test MCP tool with require_approval='never'."""
        tool = OpenAITool.mcp(
            server_label="test",
            server_description="Test server",
            server_url="https://test.com",
            require_approval="never",
        )
        
        spec = tool.get_spec()
        assert spec["require_approval"] == "never"
    
    def test_mcp_tool_with_require_approval_always(self):
        """Test MCP tool with require_approval='always'."""
        tool = OpenAITool.mcp(
            server_label="test",
            server_description="Test server",
            server_url="https://test.com",
            require_approval="always",
        )
        
        spec = tool.get_spec()
        assert spec["require_approval"] == "always"
    
    def test_mcp_tool_with_require_approval_dict(self):
        """Test MCP tool with require_approval as dict."""
        require_approval = {
            "never": {
                "tool_names": ["roll", "search"]
            }
        }
        tool = OpenAITool.mcp(
            server_label="test",
            server_description="Test server",
            server_url="https://test.com",
            require_approval=require_approval,
        )
        
        spec = tool.get_spec()
        assert spec["require_approval"] == require_approval
    
    def test_mcp_tool_with_allowed_tools(self):
        """Test MCP tool with allowed_tools filter."""
        allowed_tools = ["roll", "search"]
        tool = OpenAITool.mcp(
            server_label="test",
            server_description="Test server",
            server_url="https://test.com",
            allowed_tools=allowed_tools,
        )
        
        spec = tool.get_spec()
        assert spec["allowed_tools"] == allowed_tools
    
    def test_mcp_tool_with_authorization(self):
        """Test MCP tool with OAuth authorization."""
        auth_token = "ya29.test_token"
        tool = OpenAITool.mcp(
            server_label="test",
            server_description="Test server",
            server_url="https://test.com",
            authorization=auth_token,
        )
        
        spec = tool.get_spec()
        assert spec["authorization"] == auth_token
    
    def test_mcp_tool_with_all_parameters(self):
        """Test MCP tool with all optional parameters."""
        tool = OpenAITool.mcp(
            server_label="dmcp",
            server_description="D&D dice rolling server",
            server_url="https://dmcp-server.deno.dev/sse",
            require_approval="never",
            allowed_tools=["roll"],
            authorization="token123",
        )
        
        spec = tool.get_spec()
        assert spec == {
            "type": "mcp",
            "server_label": "dmcp",
            "server_description": "D&D dice rolling server",
            "server_url": "https://dmcp-server.deno.dev/sse",
            "require_approval": "never",
            "allowed_tools": ["roll"],
            "authorization": "token123",
        }
    
    def test_mcp_tool_missing_server_url_and_connector_id(self):
        """Test MCP tool validation - missing both server_url and connector_id."""
        with pytest.raises(ValueError, match="Either server_url.*or connector_id"):
            OpenAITool.mcp(
                server_label="test",
                server_description="Test",
                # Missing both server_url and connector_id
            )
    
    def test_mcp_tool_both_server_url_and_connector_id(self):
        """Test MCP tool validation - both server_url and connector_id provided."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            OpenAITool.mcp(
                server_label="test",
                server_description="Test",
                server_url="https://test.com",
                connector_id="connector_test",
            )
    
    def test_mcp_tool_connector_id(self):
        """Test MCP tool with connector_id."""
        tool = OpenAITool.mcp(
            server_label="google_calendar",
            server_description="Google Calendar connector",
            connector_id="connector_googlecalendar",
            authorization="ya29.token",
        )
        
        spec = tool.get_spec()
        assert spec == {
            "type": "mcp",
            "server_label": "google_calendar",
            "server_description": "Google Calendar connector",
            "connector_id": "connector_googlecalendar",
            "authorization": "ya29.token",
        }
        assert "server_url" not in spec


class TestConnectorTool:
    """Test connector tool creation."""
    
    def test_connector_tool_google_calendar(self):
        """Test creating Google Calendar connector."""
        tool = OpenAITool.connector(
            connector_id="connector_googlecalendar",
            server_label="google_calendar",
            server_description="Access Google Calendar events",
            authorization="ya29.token",
        )
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "mcp_google_calendar"
        
        spec = tool.get_spec()
        assert spec["connector_id"] == "connector_googlecalendar"
        assert spec["authorization"] == "ya29.token"
        assert "server_url" not in spec
    
    def test_connector_tool_with_require_approval(self):
        """Test connector with require_approval."""
        tool = OpenAITool.connector(
            connector_id="connector_gmail",
            server_label="gmail",
            server_description="Gmail connector",
            authorization="ya29.token",
            require_approval="never",
        )
        
        spec = tool.get_spec()
        assert spec["require_approval"] == "never"
    
    def test_connector_tool_with_allowed_tools(self):
        """Test connector with allowed_tools."""
        tool = OpenAITool.connector(
            connector_id="connector_dropbox",
            server_label="dropbox",
            server_description="Dropbox connector",
            authorization="ya29.token",
            allowed_tools=["read_file", "list_files"],
        )
        
        spec = tool.get_spec()
        assert spec["allowed_tools"] == ["read_file", "list_files"]
    
    def test_connector_tool_all_connectors(self):
        """Test all available connectors."""
        connectors = [
            "connector_dropbox",
            "connector_gmail",
            "connector_googlecalendar",
            "connector_googledrive",
            "connector_microsoftteams",
            "connector_outlookcalendar",
            "connector_outlookemail",
            "connector_sharepoint",
        ]
        
        for connector_id in connectors:
            tool = OpenAITool.connector(
                connector_id=connector_id,
                server_label=connector_id.replace("connector_", ""),
                server_description=f"{connector_id} connector",
                authorization="ya29.token",
            )
            
            spec = tool.get_spec()
            assert spec["connector_id"] == connector_id
            assert spec["authorization"] == "ya29.token"


class TestNativeToolExecution:
    """Test that native tools raise NotImplementedError on execute()."""
    
    def test_native_tool_execute_raises_error(self):
        """Test that native tools raise NotImplementedError on execute()."""
        tool = OpenAITool.web_search()
        
        with pytest.raises(NotImplementedError, match="don't use execute"):
            import asyncio
            asyncio.run(tool.execute())
    
    @pytest.mark.asyncio
    async def test_native_tool_execute_async(self):
        """Test native tool execute() in async context."""
        tool = OpenAITool.code_interpreter()
        
        with pytest.raises(NotImplementedError, match="don't use execute"):
            await tool.execute()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

