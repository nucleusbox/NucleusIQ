# NucleusIQ Test Suite

This directory contains comprehensive tests for all NucleusIQ features.

## Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures and configuration
├── test_openai_tools.py          # Tests for OpenAITool factory methods
├── test_tool_conversion.py       # Tests for tool spec conversion
├── test_custom_mcp_tool.py       # Tests for custom MCP tool (BaseTool)
├── test_agent_with_tools.py      # Tests for Agent integration with tools
├── test_base_openai.py           # Tests for BaseOpenAI LLM
└── ...                           # Other existing tests
```

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest tests/test_openai_tools.py
```

### Run with verbose output:
```bash
pytest -v
```

### Run with coverage:
```bash
pytest --cov=src/nucleusiq --cov-report=html
```

### Run specific test class:
```bash
pytest tests/test_openai_tools.py::TestOpenAIToolFactory
```

### Run specific test:
```bash
pytest tests/test_openai_tools.py::TestOpenAIToolFactory::test_web_search_tool
```

## Test Categories

### 1. OpenAI Tools (`test_openai_tools.py`)
- ✅ All OpenAI native tool types (web_search, code_interpreter, file_search, etc.)
- ✅ MCP tool creation and validation
- ✅ Connector creation
- ✅ Tool spec generation
- ✅ Error handling and validation

### 2. Tool Conversion (`test_tool_conversion.py`)
- ✅ BaseTool to OpenAI function calling format
- ✅ Native tools pass-through
- ✅ Mixed tool lists
- ✅ Parameter schema conversion

### 3. Custom MCP Tool (`test_custom_mcp_tool.py`)
- ✅ Custom MCP tool as BaseTool
- ✅ Tool initialization
- ✅ Tool execution
- ✅ Integration with OpenAI LLM

### 4. Agent Integration (`test_agent_with_tools.py`)
- ✅ Agent with BaseTool instances
- ✅ Agent with native OpenAI tools
- ✅ Agent with mixed tools
- ✅ Tool execution flow

## Environment Variables

Some tests require environment variables:

- `OPENAI_API_KEY`: Required for integration tests with OpenAI API
  - Tests will skip if not set (using `skip_if_no_openai_key` fixture)

## Test Coverage

The test suite covers:

1. **Tool Creation**
   - All OpenAITool factory methods
   - MCP tool with all parameters
   - Connector tools
   - Custom MCP tools (BaseTool)

2. **Tool Conversion**
   - BaseTool → OpenAI function calling format
   - Native tools → Pass-through
   - Mixed tool lists

3. **Agent Integration**
   - Tool registration
   - Tool spec conversion
   - Tool execution (with MockLLM)

4. **Error Handling**
   - Validation errors
   - Missing parameters
   - Invalid configurations

## Writing New Tests

When adding new features, follow these guidelines:

1. **Use descriptive test names**: `test_<feature>_<scenario>`
2. **Group related tests**: Use test classes
3. **Use fixtures**: For common setup (see `conftest.py`)
4. **Mock external dependencies**: Use MockLLM for LLM tests
5. **Test edge cases**: Invalid inputs, missing parameters, etc.

## Example Test

```python
import pytest
from nucleusiq.providers.llms.openai.tools import OpenAITool

class TestMyFeature:
    def test_my_feature_basic(self):
        """Test basic functionality."""
        tool = OpenAITool.web_search()
        assert tool.name == "web_search_preview"
    
    @pytest.mark.asyncio
    async def test_my_feature_async(self):
        """Test async functionality."""
        # Test async code
        pass
```

## Continuous Integration

Tests are run automatically in CI/CD pipelines. All tests must pass before merging.

## Troubleshooting

### Tests failing with import errors:
- Make sure you're in the project root directory
- Check that `src/` is in Python path
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Tests requiring API keys:
- Set environment variables before running tests
- Or use `pytest -m "not integration"` to skip integration tests

### MockLLM not working:
- Check that MockLLM is properly configured
- Verify function calling responses are set up correctly

