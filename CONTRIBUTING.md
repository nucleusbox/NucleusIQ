# Contributing to NucleusIQ

Thanks for your interest in contributing! This guide will help you get started.

## Quick Links

- [Issue tracker](https://github.com/nucleusbox/NucleusIQ/issues)
- [Installation guide](INSTALLATION.md)
- [Release process](RELEASE.md)

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git

### Clone and set up

```bash
git clone https://github.com/nucleusbox/NucleusIQ.git
cd NucleusIQ
```

### Install for development

**Core framework:**

```bash
cd src/nucleusiq
uv sync          # or: pip install -e ".[dev]"
```

**OpenAI provider:**

```bash
cd src/providers/llms/openai
uv sync          # or: pip install -e ".[dev]"
```

## Branching Strategy

We use **GitHub Flow** -- simple and trunk-based.

```
main        ── single source of truth, always releasable
feature/*   ── new features (branch from main, PR into main)
fix/*       ── bug fixes (branch from main, PR into main)
```

### Workflow

1. **Fork** the repository (external contributors) or create a branch (maintainers)
2. **Branch from `main`**:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/my-feature
   ```
3. Make your changes
4. Push and **open a PR into `main`**

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is in `ruff.toml` at the project root.

```bash
# Check for lint errors
ruff check src/ --config ruff.toml

# Auto-fix what's possible
ruff check src/ --config ruff.toml --fix

# Check formatting
ruff format src/ --config ruff.toml --check

# Apply formatting
ruff format src/ --config ruff.toml
```

Key rules:
- Line length: 120 characters
- Target: Python 3.10+
- Import sorting is enforced (isort-compatible)

## Testing

Tests live alongside each package:

```bash
# Core tests
cd src/nucleusiq
python -m pytest tests/ -q --import-mode=importlib

# OpenAI provider tests
cd src/providers/llms/openai
python -m pytest tests/ -q --import-mode=importlib
```

### Writing tests

- Place tests in the `tests/` directory of the relevant package
- Name test files `test_*.py`
- Use `pytest` fixtures and parametrize where appropriate
- Mock external APIs (LLM calls, network requests)
- Aim for meaningful coverage, not 100% line coverage

## Making a Pull Request

1. **One concern per PR** — don't mix features with refactors
2. **Write a clear description** — explain what and why, not just how
3. **Keep it small** — smaller PRs get reviewed faster
4. **Add tests** — for bug fixes, add a test that fails without your fix
5. **Update docs** — if you change public API, update docstrings and relevant docs
6. **Update CHANGELOG.md** — for any user-facing change, add a line under `[Unreleased]`

### CI checks

Every PR runs these checks automatically:

| Check | What it does |
|-------|-------------|
| Core tests | pytest on Python 3.10, 3.11, 3.12, 3.13 |
| OpenAI tests | pytest on Python 3.10, 3.12 |
| Lint | ruff check + ruff format |
| Import check | Verifies public exports work |
| Build | Builds sdist + wheel for both packages |
| Security | pip-audit for known vulnerabilities |

All checks must pass before merging.

## Project Structure

```
src/
  nucleusiq/                      # Core framework
    core/
      agents/                     # Agent system, execution modes
      prompts/                    # Prompt engineering techniques
      tools/                      # Tool interface
      memory/                     # Memory strategies
      llms/                       # LLM base classes
    plugins/                      # Plugin system + built-ins
    tests/                        # Core tests
  providers/
    llms/openai/                  # OpenAI provider
      nucleusiq_openai/           # Package source
      tests/                      # Provider tests
notebooks/agents/                 # Example notebooks
docs/                             # Documentation
```

## Reporting Issues

- Use the [bug report](https://github.com/nucleusbox/NucleusIQ/issues/new?template=bug_report.yml) template for bugs
- Use the [feature request](https://github.com/nucleusbox/NucleusIQ/issues/new?template=feature_request.yml) template for ideas
- Include version numbers, Python version, and OS
- Provide minimal reproducible examples when possible

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
