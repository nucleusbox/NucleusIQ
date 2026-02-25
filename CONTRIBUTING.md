# Contributing to NucleusIQ

Welcome! Thank you for your interest in contributing. Whether you're fixing bugs, adding features, improving documentation, or sharing feedback, your involvement helps make NucleusIQ better for everyone.

## Ways to Contribute

### Report Bugs

Found a bug? Help us fix it:

1. **Search** [existing issues](https://github.com/nucleusbox/NucleusIQ/issues) to check if it's already reported
2. **Open a new issue** using the [Bug Report](https://github.com/nucleusbox/NucleusIQ/issues/new?template=bug_report.yml) template
3. Include a minimal, reproducible example -- maintainers can't fix what they can't reproduce
4. Be patient while we triage -- we'll follow up if we need more info

### Suggest Features

Have an idea?

1. **Search** [existing issues](https://github.com/nucleusbox/NucleusIQ/issues?q=label%3Aenhancement) for similar requests
2. **Open a new issue** using the [Feature Request](https://github.com/nucleusbox/NucleusIQ/issues/new?template=feature_request.yml) template
3. Describe the use case and why it would be valuable -- code examples are welcome

### Improve Documentation

Documentation improvements are always welcome. Fix typos, clarify explanations, add examples -- every bit helps. No issue needed, just open a PR.

### Contribute Code

Looking for something to work on? Check issues labeled:
- [`good first issue`](https://github.com/nucleusbox/NucleusIQ/labels/good%20first%20issue) -- great for new contributors
- [`help wanted`](https://github.com/nucleusbox/NucleusIQ/labels/help%20wanted) -- we'd love your help on these

If you start working on an issue, comment on it so others know. This avoids duplicate work.

---

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

---

## Branching

We use **GitHub Flow** -- one main branch, feature branches, pull requests.

```
main                           -- single source of truth, always releasable
username/short-description     -- your working branch (PR into main)
```

### Branch naming

Use `username/short-description` so it's clear who owns the branch:

```
brijesh/streaming-support
brijesh/fix-validation-bug
alice/add-gemini-provider
release/v0.2.0
```

### Workflow

1. **Fork** the repo (external contributors) or create a branch (maintainers)
2. **Branch from `main`**:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b yourname/my-feature
   ```
3. Make your changes
4. Push and **open a PR into `main`**

---

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is in `ruff.toml` at the project root.

```bash
# Lint
ruff check src/ --config ruff.toml

# Auto-fix
ruff check src/ --config ruff.toml --fix

# Format
ruff format src/ --config ruff.toml
```

Key rules:
- Line length: 88 characters
- Target: Python 3.10+
- Import sorting is enforced (isort-compatible)

---

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

---

## Making a Pull Request

1. **One concern per PR** -- don't mix features with refactors
2. **Write a clear description** -- explain what and why, not just how
3. **Keep it small** -- smaller PRs get reviewed faster
4. **Add tests** -- for bug fixes, add a test that fails without your fix
5. **Update docs** -- if you change public API, update docstrings and relevant docs
6. **Update CHANGELOG.md** -- for any user-facing change, add a line under `[Unreleased]`

### CI checks

Every PR runs these automatically:

| Check | What it does |
|-------|-------------|
| Core tests | pytest on Python 3.10, 3.11, 3.12, 3.13 |
| OpenAI tests | pytest on Python 3.10, 3.12 |
| Lint | ruff check + ruff format |
| Import check | Verifies public exports work |
| Build | Builds sdist + wheel for both packages |
| Security | pip-audit for known vulnerabilities |

All checks must pass before merging.

---

## Use of AI Tools

AI tools can help you contribute faster, and we encourage their responsible use. However, AI assistance must be paired with meaningful human effort, judgment, and understanding of the codebase.

**We will close PRs that appear to be low-effort, AI-generated content** without meaningful human review or contextual understanding. If the effort to create a PR is less than the effort for maintainers to review it, it should not be submitted.

---

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

---

## Quick Links

- [Issue tracker](https://github.com/nucleusbox/NucleusIQ/issues)
- [Installation guide](INSTALLATION.md)
- [Changelog](CHANGELOG.md)
- [Release process](RELEASE.md)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
