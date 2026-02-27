# NucleusIQ

[![CI](https://github.com/nucleusbox/NucleusIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/nucleusbox/NucleusIQ/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nucleusiq)](https://pypi.org/project/nucleusiq/)
[![Python](https://img.shields.io/pypi/pyversions/nucleusiq)](https://pypi.org/project/nucleusiq/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
| **Tool System** | `BaseTool` interface + OpenAI native tools (function, code_interpreter, file_search, web_search, MCP) |
| **Memory** | 5 strategies: full history, sliding window, summary, summary+window, token budget |
| **Plugins** | 9 built-in: call limits, retry, fallback, PII guard, human approval, tool guard, context window, result validator |
| **Structured Output** | Schema-based output parsing with Pydantic, dataclass, TypedDict support |

## Execution Modes

NucleusIQ agents use the **Gearbox Strategy** — three execution modes that scale from simple chat to autonomous reasoning:

| Capability | Direct | Standard | Autonomous |
|---|---|---|---|
| Memory | Yes | Yes | Yes |
| Plugins | Yes | Yes | Yes |
| Tools | Yes (max 5) | Yes (max 30) | Yes (max 100) |
| Tool loop | Yes | Yes | Yes |
| Task decomposition | No | No | Yes |
| Independent verification (Critic) | No | No | Yes |
| Targeted correction (Refiner) | No | No | Yes |
| Validation pipeline | No | No | Yes |

Tool limits are configurable via `AgentConfig(max_tool_calls=N)`. The framework validates tool count at agent creation and raises a clear error if the limit is exceeded.

```python
# Direct: fast Q&A, simple lookups (max 5 tool calls)
AgentConfig(execution_mode=ExecutionMode.DIRECT)

# Standard: multi-step tool workflows (max 30 tool calls) — default
AgentConfig(execution_mode=ExecutionMode.STANDARD)

# Autonomous: orchestration + Critic/Refiner verification (max 100 tool calls)
AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS)
```

See the [PE Due Diligence notebook](notebooks/agents/pe_due_diligence.ipynb) for a real-world demo of Autonomous mode achieving **100% accuracy** on 8 complex financial analyses with external validation.

## Packages

| Package | Description | Install |
|---------|-------------|---------|
| [`nucleusiq`](https://pypi.org/project/nucleusiq/) | Core framework | `pip install nucleusiq` |
| [`nucleusiq-openai`](https://pypi.org/project/nucleusiq-openai/) | OpenAI provider | `pip install nucleusiq-openai` |

## Project Structure

```
src/
  nucleusiq/core/          # Core framework (agents, prompts, tools, memory, plugins)
  providers/llms/openai/   # OpenAI provider
notebooks/agents/          # Example notebooks
docs/                      # Documentation
```

## Testing

```bash
# Core tests (1207 passing, 2 skipped)
cd src/nucleusiq && python -m pytest tests/ -q

# OpenAI provider tests (116 passing)
cd src/providers/llms/openai && python -m pytest tests/ -q
```

## Documentation

- [INSTALLATION.md](INSTALLATION.md) — Setup instructions (pip, uv, development)
- [CHANGELOG.md](CHANGELOG.md) — Release notes
- [RELEASE.md](RELEASE.md) — Release process and branching strategy
- [docs/IMPLEMENTATION_TRACKER.md](docs/IMPLEMENTATION_TRACKER.md) — Detailed implementation status

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b yourname/my-feature main`
3. Make your changes and add tests
4. Submit a pull request to `main`

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

## License

[MIT](LICENSE)
