# Contributing to NucleusIQ

Thank you for your interest in contributing to NucleusIQ! We welcome contributions from the community and are excited to work with you.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)
- [Getting Help](#getting-help)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to info@nucleusbox.com.

## Getting Started

1. **Fork the Repository**
   - Click the "Fork" button on GitHub to create your own copy of the repository.

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/NucleusIQ.git
   cd NucleusIQ
   ```

3. **Add Upstream Remote**
   ```bash
   git remote add upstream https://github.com/nucleusbox/NucleusIQ.git
   ```

## Development Setup

### Prerequisites

- **Python 3.8+** (Python 3.12 recommended)
- **Git**
- **pip** (Python package manager)

### Setup Steps

1. **Create Virtual Environment**
   ```bash
   python -m venv nb
   # On Windows:
   nb\Scripts\activate
   # On Unix/MacOS:
   source nb/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Development Dependencies**
   ```bash
   pip install pytest pytest-asyncio pytest-html pytest-cov black flake8 mypy
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here  # Optional, for integration tests
   ```

5. **Verify Installation**
   ```bash
   # Run tests to verify everything works
   $env:PYTHONPATH="src"; pytest tests/ -v
   # On Unix/MacOS:
   PYTHONPATH=src pytest tests/ -v
   ```

## Project Structure

```
NucleusIQ/
├── src/
│   ├── nucleusiq/              # Main package
│   │   ├── agents/             # Agent classes and configuration
│   │   ├── core/               # Core components (tools, memory)
│   │   ├── llms/               # LLM base classes and mock
│   │   ├── prompts/            # Prompt engineering techniques
│   │   ├── providers/          # LLM and DB providers
│   │   └── utilities/          # Utility functions
│   └── examples/               # Example code
│       ├── agents/             # Agent examples
│       ├── prompts/            # Prompt examples
│       └── tools/              # Tool examples
├── tests/                      # Test suite
│   ├── conftest.py            # Pytest fixtures
│   └── test_*.py              # Test files
├── docs/                       # Documentation
│   ├── strategy/              # Strategy documents
│   └── TOOL_DESIGN.md         # Tool design documentation
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
├── setup.py                   # Package setup
└── README.md                  # Project README
```

### Key Directories

- **`src/nucleusiq/agents/`**: Core agent implementation
- **`src/nucleusiq/core/tools/`**: Base tool classes (LLM-agnostic)
- **`src/nucleusiq/providers/llms/`**: LLM provider implementations
- **`src/nucleusiq/providers/llms/openai/tools/`**: OpenAI-specific tools
- **`tests/`**: Comprehensive test suite (185+ tests)
- **`src/examples/`**: Working examples for users

## Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Follow the code style guidelines
   - Add tests for new features
   - Update documentation as needed

3. **Keep Your Branch Updated**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Test Your Changes**
   ```bash
   # Run all tests
   $env:PYTHONPATH="src"; pytest tests/ -v
   
   # Run specific test file
   $env:PYTHONPATH="src"; pytest tests/test_your_feature.py -v
   
   # Run with coverage
   $env:PYTHONPATH="src"; pytest tests/ --cov=src/nucleusiq --cov-report=html
   ```

5. **Commit Your Changes**
   - Follow commit message guidelines (see below)
   - Make atomic commits (one logical change per commit)

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Open a PR on GitHub
   - Fill out the PR template
   - Link related issues
   - Wait for review and address feedback

## Code Style Guidelines

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** for function signatures
- Maximum line length: **100 characters** (soft limit)
- Use **4 spaces** for indentation (no tabs)

### Code Formatting

We use **Black** for code formatting:

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

### Linting

We use **flake8** for linting:

```bash
# Run linter
flake8 src/ tests/
```

### Type Checking

We use **mypy** for type checking:

```bash
# Type check
mypy src/
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `BaseAgent`, `OpenAITool`)
- **Functions/Methods**: `snake_case` (e.g., `execute_task`, `get_spec`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_MODEL`)
- **Private methods**: `_leading_underscore` (e.g., `_parse_response`)

### Import Organization

```python
# Standard library imports
import asyncio
import json
from typing import Dict, List, Optional

# Third-party imports
import pydantic
from openai import OpenAI

# Local imports
from nucleusiq.agents.agent import Agent
from nucleusiq.core.tools.base_tool import BaseTool
```

### Docstrings

Use Google-style docstrings:

```python
def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a task using the agent's tools and LLM.
    
    Args:
        task: Dictionary containing task objective and optional context.
            Must include 'objective' key.
    
    Returns:
        Dictionary containing execution result with 'output' and 'status' keys.
    
    Raises:
        ValueError: If task is missing required 'objective' key.
        RuntimeError: If agent is not initialized.
    """
    pass
```

### Pydantic Models

- Use **Pydantic V2** style (`ConfigDict` instead of `class Config:`)
- Use `model_config = ConfigDict(...)` for configuration

```python
from pydantic import BaseModel, ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    
    name: str
    value: int
```

## Testing

### Test Requirements

- **All new features must include tests**
- **Test coverage should not decrease**
- **All tests must pass before PR submission**

### Running Tests

```bash
# Run all tests
$env:PYTHONPATH="src"; pytest tests/ -v

# Run specific test file
$env:PYTHONPATH="src"; pytest tests/test_agent.py -v

# Run specific test class
$env:PYTHONPATH="src"; pytest tests/test_agent.py::TestAgent -v

# Run specific test method
$env:PYTHONPATH="src"; pytest tests/test_agent.py::TestAgent::test_agent_execute -v

# Run with coverage
$env:PYTHONPATH="src"; pytest tests/ --cov=src/nucleusiq --cov-report=html

# Run with HTML report
$env:PYTHONPATH="src"; pytest tests/ --html=test_report.html --self-contained-html
```

### Writing Tests

- Use **pytest** as the testing framework
- Use **pytest-asyncio** for async tests
- Group related tests in classes
- Use descriptive test names: `test_<feature>_<scenario>`
- Use fixtures from `conftest.py` for common setup

**Example:**

```python
import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.llms.mock_llm import MockLLM

class TestAgentFeature:
    @pytest.mark.asyncio
    async def test_agent_feature_basic(self, mock_llm):
        """Test basic feature functionality."""
        agent = Agent(llm=mock_llm)
        result = await agent.execute({"objective": "test"})
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_agent_feature_error_handling(self, mock_llm):
        """Test error handling."""
        agent = Agent(llm=mock_llm)
        with pytest.raises(ValueError):
            await agent.execute({})  # Missing objective
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Mock External Services**: Use `MockLLM` instead of real API calls in tests

See [tests/README.md](tests/README.md) for more details.

## Documentation

### Code Documentation

- **All public classes and methods must have docstrings**
- Use Google-style docstrings
- Include type hints in function signatures
- Document complex algorithms and design decisions

### Example Documentation

```python
class BaseTool(ABC):
    """
    Abstract base class for all tools in NucleusIQ.
    
    Tools are LLM-agnostic and can be used with any LLM provider.
    LLM providers are responsible for converting tool specs to their native format.
    
    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
    """
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with provided arguments.
        
        Args:
            **kwargs: Tool-specific arguments
        
        Returns:
            Tool execution result
        """
        pass
```

### README Updates

- Update `README.md` if adding new features
- Update `src/examples/README.md` if adding new examples
- Update `tests/README.md` if adding new test patterns

### Documentation Files

- **`docs/TOOL_DESIGN.md`**: Tool architecture and design
- **`ROADMAP.md`**: Project roadmap and future plans
- **`CHANGELOG.md`**: List of changes (if maintained)

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(agents): add memory persistence to Agent class

Implement persistent memory storage using SQLite backend.
Agents can now remember conversations across sessions.

Closes #123
```

```
fix(tools): handle None values in tool execution

Prevent AttributeError when tool receives None arguments.
Add validation in BaseTool.execute().

Fixes #456
```

```
docs(readme): update installation instructions

Add Windows-specific activation commands for virtual environment.
```

## Pull Request Process

### Before Submitting

1. ✅ **All tests pass**: `pytest tests/ -v`
2. ✅ **Code is formatted**: `black src/ tests/`
3. ✅ **No linting errors**: `flake8 src/ tests/`
4. ✅ **Type checking passes**: `mypy src/`
5. ✅ **Documentation updated**: Docstrings and README
6. ✅ **Branch is up to date**: Rebased on `main`

### PR Template

When creating a PR, include:

1. **Description**: What changes are made and why
2. **Type**: Feature, Bug Fix, Documentation, etc.
3. **Related Issues**: Link to related issues
4. **Testing**: How to test the changes
5. **Checklist**: Confirm all requirements met

### PR Review Process

1. **Automated Checks**: CI/CD runs tests and linting
2. **Code Review**: Maintainers review code quality and design
3. **Feedback**: Address review comments
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to `main`

### PR Guidelines

- **Keep PRs focused**: One feature or fix per PR
- **Keep PRs small**: Easier to review and merge
- **Write clear descriptions**: Explain what and why
- **Respond to feedback**: Be open to suggestions
- **Update documentation**: Keep docs in sync with code

## Areas for Contribution

We welcome contributions in these areas:

### High Priority

- 🚀 **Additional LLM Providers**: Ollama, Groq, Gemini, Anthropic
- 🗄️ **Vector Database Integrations**: Pinecone, Chroma, Weaviate
- 💾 **Memory System**: Persistent memory storage and retrieval
- 🔄 **Multi-Agent Orchestration**: Agent coordination and workflows
- 📊 **Observability Dashboard**: Monitoring and debugging tools

### Medium Priority

- 🧪 **More Test Coverage**: Increase test coverage for edge cases
- 📚 **Documentation**: Improve guides, tutorials, and API docs
- 🎨 **Examples**: More real-world examples and use cases
- 🐛 **Bug Fixes**: Fix issues reported in GitHub Issues
- ⚡ **Performance**: Optimize existing code

### Low Priority

- 🎨 **UI/UX**: Web interface or CLI improvements
- 🌐 **Internationalization**: Multi-language support
- 🔌 **Plugins**: Plugin system for extensibility
- 📦 **Packaging**: Improve distribution and installation

### How to Find Issues

- **Good First Issues**: Look for `good first issue` label
- **Help Wanted**: Look for `help wanted` label
- **Bug Reports**: Check open issues
- **Feature Requests**: Review discussions and roadmap

## Getting Help

### Resources

- **Documentation**: Check `README.md` and `docs/` folder
- **Examples**: See `src/examples/` for usage examples
- **Issues**: Search existing GitHub Issues
- **Discussions**: Use GitHub Discussions for questions

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and discussions
- **Email**: info@nucleusbox.com (for sensitive matters)

### Before Asking for Help

1. ✅ Check existing documentation
2. ✅ Search GitHub Issues and Discussions
3. ✅ Review examples in `src/examples/`
4. ✅ Try to reproduce the issue
5. ✅ Provide minimal reproducible example

## Recognition

Contributors will be:

- **Listed in CONTRIBUTORS.md** (if maintained)
- **Mentioned in release notes** for significant contributions
- **Thanked in PR comments** and discussions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (Apache 2.0).

---

Thank you for contributing to NucleusIQ! 🎉

Your contributions help make NucleusIQ better for everyone. We appreciate your time and effort.
