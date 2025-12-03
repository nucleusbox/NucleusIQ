# NucleusIQ Test Suite

## Overview

Comprehensive test suite for NucleusIQ framework, organized by component for better readability and maintainability.

## Test Structure

```
tests/
├── agents/          # Agent-related tests
│   ├── test_agent.py              # Core Agent class tests
│   └── test_agent_precedence.py   # Prompt precedence tests
├── prompts/         # Prompt-related tests
│   ├── test_zero_shot_prompt.py
│   ├── test_few_shot_prompt.py
│   ├── test_chain_of_thought_prompt.py
│   ├── test_auto_chain_of_thought_prompt.py
│   ├── test_retrieval_augmented_generation_prompt.py
│   ├── test_meta_prompt.py
│   ├── test_prompt_composer.py
│   ├── test_metadata_tags.py
│   ├── test_output_parser.py
│   ├── test_partial_variables.py
│   ├── test_partial_variables_various_types.py
│   └── test_unrecognized_fields.py
├── tools/           # Tool-related tests
│   ├── test_openai_tools.py       # OpenAI tool factory tests
│   ├── test_tool_conversion.py    # Tool conversion tests
│   └── test_custom_mcp_tool.py    # Custom MCP tool tests
├── llms/            # LLM-related tests
│   └── test_base_openai.py        # BaseOpenAI tests
├── conftest.py      # Pytest configuration and fixtures
└── README.md        # This file
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Tests by Category

```bash
# Agent tests
pytest tests/agents/

# Prompt tests
pytest tests/prompts/

# Tool tests
pytest tests/tools/

# LLM tests
pytest tests/llms/
```

### Run Specific Test File

```bash
pytest tests/agents/test_agent.py
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src/nucleusiq --cov-report=html
```

## Test Categories

### Agent Tests (`tests/agents/`)

**test_agent.py:**
- Agent initialization
- Agent execution (with/without planning)
- Agent planning (basic and LLM-based)
- State transitions
- Error handling
- Task and Plan integration
- Tool execution

**test_agent_precedence.py:**
- Prompt precedence over role/objective
- Fallback to role/objective when prompt is None
- Warning messages when override occurs
- Planning context behavior
- Narrative field optional

### Prompt Tests (`tests/prompts/`)

- **Zero-Shot Prompting**: Basic prompt formatting
- **Few-Shot Prompting**: Example-based prompting
- **Chain-of-Thought**: Step-by-step reasoning
- **Auto Chain-of-Thought**: Automatic reasoning chain generation
- **Retrieval-Augmented Generation**: RAG prompt formatting
- **Meta-Prompting**: Dynamic prompt generation
- **Prompt Composer**: Flexible prompt composition
- **Metadata and Tags**: Prompt metadata handling
- **Output Parsing**: Result processing
- **Partial Variables**: Variable substitution
- **Unrecognized Fields**: Error handling

### Tool Tests (`tests/tools/`)

- **OpenAI Tools**: Factory methods for all OpenAI tool types
  - Web search
  - Code interpreter
  - File search
  - Image generation
  - MCP servers and connectors
  - Computer use
- **Tool Conversion**: BaseTool to LLM-specific format conversion
- **Custom MCP Tools**: BaseTool-based MCP implementations

### LLM Tests (`tests/llms/`)

- **BaseOpenAI**: OpenAI client functionality
  - Async/sync modes
  - Error handling
  - Retry logic
  - Tool conversion

## Test Coverage

### Current Coverage

- ✅ Agent initialization and execution
- ✅ Task and Plan classes
- ✅ Planning integration
- ✅ Prompt precedence
- ✅ All prompt types
- ✅ All OpenAI tool types
- ✅ Tool conversion
- ✅ Custom MCP tools
- ✅ Agent-tool integration
- ✅ State management
- ✅ Error handling

### Test Count

- **Agent Tests**: 30+ tests
- **Prompt Tests**: 50+ tests
- **Tool Tests**: 30+ tests
- **LLM Tests**: 10+ tests
- **Total**: 120+ tests

## Fixtures

### Common Fixtures (`conftest.py`)

- `openai_api_key`: OpenAI API key from environment
- `skip_if_no_openai_key`: Skip tests if API key not available
- `mock_openai_client`: Mock OpenAI client for unit tests

### Agent Fixtures

- `mock_llm`: MockLLM instance for testing
- `agent_with_prompt`: Agent with prompt configured
- `agent_without_prompt`: Agent without prompt (uses role/objective)

## Writing New Tests

### Test Naming Convention

- Test files: `test_<component>_<feature>.py`
- Test classes: `Test<Component><Feature>`
- Test methods: `test_<what_is_being_tested>`

### Example Test Structure

```python
"""
Tests for <Component> <Feature>.

Tests verify:
- Feature 1
- Feature 2
- Edge cases
"""

import pytest
from nucleusiq.<module> import <Component>


class TestComponentFeature:
    """Test <Component> <Feature>."""
    
    @pytest.fixture
    def component(self):
        """Create component instance."""
        return <Component>(...)
    
    @pytest.mark.asyncio
    async def test_feature_behavior(self, component):
        """Test that feature works correctly."""
        result = await component.method()
        assert result == expected
```

## Continuous Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch
- Scheduled nightly runs

## Test Reports

Test reports are generated in:
- `test_report.html`: HTML test report
- `TEST_REPORT.md`: Markdown test report

Generate reports:
```bash
pytest tests/ --html=test_report.html --self-contained-html
```

## Notes

- All tests use `pytest-asyncio` for async test support
- Mock LLM is used for most tests to avoid API calls
- Integration tests require `OPENAI_API_KEY` environment variable
- Tests are organized by component for better maintainability

---

*Last Updated: After test reorganization and precedence implementation*
