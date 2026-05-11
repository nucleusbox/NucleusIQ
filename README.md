<p align="center">
  <img src="assets/images/nucleusiq-logo.png" alt="NucleusIQ logo" width="400" />
</p>

[![CI](https://github.com/nucleusbox/NucleusIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/nucleusbox/NucleusIQ/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

| Package | PyPI | Version | Python |
|---------|------|---------|--------|
| **nucleusiq** | [![PyPI](https://img.shields.io/pypi/v/nucleusiq)](https://pypi.org/project/nucleusiq/) | ![Version](https://img.shields.io/pypi/v/nucleusiq?label=) | [![Python](https://img.shields.io/pypi/pyversions/nucleusiq)](https://pypi.org/project/nucleusiq/) |
| **nucleusiq-openai** | [![PyPI](https://img.shields.io/pypi/v/nucleusiq-openai)](https://pypi.org/project/nucleusiq-openai/) | ![Version](https://img.shields.io/pypi/v/nucleusiq-openai?label=) | [![Python](https://img.shields.io/pypi/pyversions/nucleusiq-openai)](https://pypi.org/project/nucleusiq-openai/) |
| **nucleusiq-gemini** | [![PyPI](https://img.shields.io/pypi/v/nucleusiq-gemini)](https://pypi.org/project/nucleusiq-gemini/) | ![Version](https://img.shields.io/pypi/v/nucleusiq-gemini?label=) | [![Python](https://img.shields.io/pypi/pyversions/nucleusiq-gemini)](https://pypi.org/project/nucleusiq-gemini/) |
| **nucleusiq-groq** | [![PyPI](https://img.shields.io/pypi/v/nucleusiq-groq)](https://pypi.org/project/nucleusiq-groq/) | ![Pre-release](https://img.shields.io/pypi/v/nucleusiq-groq?label=) | [![Python](https://img.shields.io/pypi/pyversions/nucleusiq-groq)](https://pypi.org/project/nucleusiq-groq/) |
| **nucleusiq-ollama** | [![PyPI](https://img.shields.io/pypi/v/nucleusiq-ollama)](https://pypi.org/project/nucleusiq-ollama/) | ![Pre-release](https://img.shields.io/pypi/v/nucleusiq-ollama?label=) | [![Python](https://img.shields.io/pypi/pyversions/nucleusiq-ollama)](https://pypi.org/project/nucleusiq-ollama/) |

---

## What Is NucleusIQ?

NucleusIQ is an **open-source, agent-first Python framework** for building AI agents that work in real environments - beyond demos - without creating a one-off system you will regret maintaining.

**In one line:**

> NucleusIQ helps developers build AI agents like software systems: maintainable, testable, provider-portable, and ready for real-world integration.

NucleusIQ is built on a simple belief:

> An agent is not a single model call. An agent is a managed runtime with memory, tools, policy, streaming, structure, and responsibilities.

---
## NucleusIQ philosophy:

A shared doctrine for what NucleusIQ stands for, why it exists, and how it should evolve over time.

See **[NucleusIQ_Philosphy.md](NucleusIQ_Philosphy.md)**.

## Quick Start

```bash
# Install core + OpenAI provider
pip install nucleusiq nucleusiq-openai

# Or with Gemini provider
pip install nucleusiq nucleusiq-gemini

# Or with Groq provider (beta; see provider README in-repo)
pip install nucleusiq nucleusiq-groq

# Or with Ollama (local LLMs; alpha on PyPI once published)
pip install nucleusiq nucleusiq-ollama

# Or with uv
uv pip install nucleusiq nucleusiq-openai
```

```python
import asyncio
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq_openai import BaseOpenAI

agent = Agent(
    name="analyst",
    llm=BaseOpenAI(model="gpt-4o-mini"),
    config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
)

result = asyncio.run(agent.execute("What is the capital of France?"))
print(result)
```

See [INSTALLATION.md](INSTALLATION.md) for full setup instructions (pip, uv, development mode).

## What's Inside

| Component | What it does |
|-----------|-------------|
| **3 Execution Modes** | `DIRECT` (single call), `STANDARD` (tool loop), `AUTONOMOUS` (orchestration + validation + retry) |
| **Streaming** | `execute_stream()` — real-time token-by-token output with tool call visibility across all modes |
| **7 Prompt Techniques** | ZeroShot, FewShot, ChainOfThought, AutoCoT, RAG, PromptComposer, MetaPrompt |
| **Multimodal Attachments** | 7 attachment types (text, PDF, images, files) with provider-native optimisation |
| **Built-in File Tools** | `FileReadTool`, `FileSearchTool`, `DirectoryListTool`, `FileExtractTool` — sandboxed to workspace |
| **Tool System** | `BaseTool` interface + `@tool` decorator + provider native tools (OpenAI: code_interpreter, file_search, web_search; Gemini: Google Search, Code Execution, URL Context, Maps) |
| **Memory** | 5 strategies (full history, sliding window, summary, summary+window, token budget) with file-aware metadata |
| **Plugins** | 10 built-in: call limits, retry, fallback, PII guard, human approval, tool guard, attachment guard, context window, result validator |
| **Usage Tracking** | Token usage per call with purpose tagging (main, planning, tool loop, critic, refiner) and cost estimation |
| **Structured Output** | Schema-based output parsing with Pydantic, dataclass, TypedDict support |
| **Provider Portability** | Swap providers (OpenAI, Gemini, Groq, Ollama, …) with one line — same agent code, same tools, same plugins |

## Execution Modes

NucleusIQ agents use the **Gearbox Strategy** — three execution modes that scale from simple chat to autonomous reasoning:

| Capability | Direct | Standard | Autonomous |
|---|---|---|---|
| Memory | Yes | Yes | Yes |
| Plugins | Yes | Yes | Yes |
| Tools | Yes (max 25) | Yes (max 80) | Yes (max 300) |
| Tool loop | Yes | Yes | Yes |
| Task decomposition | No | No | Yes |
| Independent verification (Critic) | No | No | Yes |
| Targeted correction (Refiner) | No | No | Yes |
| Validation pipeline | No | No | Yes |

Tool limits are configurable via `AgentConfig(max_tool_calls=N)`. The framework validates tool count at agent creation and raises a clear error if the limit is exceeded.

```python
# Direct: fast Q&A, simple lookups (max 25 tool calls)
AgentConfig(execution_mode=ExecutionMode.DIRECT)

# Standard: multi-step tool workflows (max 80 tool calls) — default
AgentConfig(execution_mode=ExecutionMode.STANDARD)

# Autonomous: orchestration + Critic/Refiner verification (max 300 tool calls)
AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS)
```

See the [PE Due Diligence notebook](notebooks/agents/pe_due_diligence.ipynb) for a real-world demo of Autonomous mode achieving **100% accuracy** on 8 complex financial analyses with external validation.

## Packages

| Package | Description | Install |
|---------|-------------|---------|
| [`nucleusiq`](https://pypi.org/project/nucleusiq/) | Core framework | `pip install nucleusiq` |
| [`nucleusiq-openai`](https://pypi.org/project/nucleusiq-openai/) | OpenAI provider | `pip install nucleusiq-openai` |
| [`nucleusiq-gemini`](https://pypi.org/project/nucleusiq-gemini/) | Google Gemini provider | `pip install nucleusiq-gemini` |
| [`nucleusiq-groq`](https://pypi.org/project/nucleusiq-groq/) | Groq inference (**beta** `0.1.0b1`; Chat Completions via official `groq` SDK) | `pip install nucleusiq-groq` · [README](src/providers/inference/groq/README.md) · [Design](docs/design/GROQ_PROVIDER.md) |
| [`nucleusiq-ollama`](https://pypi.org/project/nucleusiq-ollama/) | Ollama inference (**alpha** `0.1.0a1`; local/remote via official `ollama` SDK) | `pip install nucleusiq-ollama` · [README](src/providers/inference/ollama/README.md) · [Design](docs/design/OLLAMA_PROVIDER.md) |

## Project Structure

```
src/
  nucleusiq/core/              # Core framework (agents, prompts, tools, memory, plugins)
  providers/llms/openai/       # OpenAI provider
  providers/llms/gemini/       # Gemini provider
  providers/inference/groq/    # Groq provider (nucleusiq-groq)
  providers/inference/ollama/  # Ollama provider (nucleusiq-ollama)
notebooks/agents/              # Example notebooks
```

## Testing

```bash
# Monorepo: core setuptools packages + Hatch provider roots (OpenAI, Gemini, Groq, Ollama)
python scripts/verify_core_package_layout.py

# Core tests (1,795 passing)
cd src/nucleusiq && python -m pytest tests/ -q

# OpenAI provider tests (224 passing)
cd src/providers/llms/openai && python -m pytest tests/ -q

# Gemini provider unit tests (221 passing)
cd src/providers/llms/gemini && python -m pytest tests/unit/ -q

# Groq provider tests (requires dev group / uv; ≥90% coverage gate)
cd src/providers/inference/groq && uv run pytest -q

# Ollama provider tests (≥95% coverage gate; 100% line coverage on package)
cd src/providers/inference/ollama && uv run pytest -q

# Gemini integration tests (requires GEMINI_API_KEY)
cd src/providers/llms/gemini && python -m pytest tests/integration/ -q
```

## Documentation

- **Published docs** — https://nucleusbox.github.io/nucleusiq-docs/
- **Docs repository** — https://github.com/nucleusbox/nucleusiq-docs
- [INSTALLATION.md](INSTALLATION.md) — Setup instructions (pip, uv, development)
- [CHANGELOG.md](CHANGELOG.md) — Release notes
- [RELEASE.md](RELEASE.md) — Release process and branching strategy
- [File handling guide](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/file-handling/) — Attachment vs Tool vs Both decision guide

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b yourname/my-feature main`
3. Make your changes and add tests
4. Submit a pull request to `main`

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

## License

[MIT](LICENSE)
