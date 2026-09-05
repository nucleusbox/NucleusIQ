<p align="center">
  <a href="https://github.com/nucleusbox/NucleusIQ">
    <img src="assets/images/nucleusiq-logo.png" alt="NucleusIQ logo" width="420" />
  </a>
</p>

<h3 align="center">Agents that survive production.</h3>

<p align="center">
  <em>Open-source, agent-first Python framework for building AI agents that hold up under real, long, tool-heavy work — reliable, observable, and portable across LLM providers. Not a coding copilot. Not a chatbot. The framework you build serious agents with.</em>
</p>

<!-- Project status badges -->
<p align="center">
  <a href="https://github.com/nucleusbox/NucleusIQ/actions/workflows/ci.yml"><img src="https://github.com/nucleusbox/NucleusIQ/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/nucleusbox/NucleusIQ/actions/workflows/codeql.yml"><img src="https://github.com/nucleusbox/NucleusIQ/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/nucleusiq/"><img src="https://img.shields.io/pypi/pyversions/nucleusiq" alt="Python versions"></a>
  <a href="https://pypistats.org/packages/nucleusiq"><img src="https://img.shields.io/pypi/dm/nucleusiq?label=downloads%2Fmonth" alt="PyPI downloads per month"></a>
  <a href="https://nucleusbox.github.io/nucleusiq-docs/"><img src="https://img.shields.io/badge/docs-nucleusbox.github.io-blue" alt="Docs"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs welcome"></a>
</p>

<!-- Per-package PyPI badges (all first-party packages are stable as of v0.7.13) -->
<p align="center">
  <a href="https://pypi.org/project/nucleusiq/"><img src="https://img.shields.io/pypi/v/nucleusiq?label=nucleusiq&color=brightgreen" alt="nucleusiq"></a>
  <a href="https://pypi.org/project/nucleusiq-openai/"><img src="https://img.shields.io/pypi/v/nucleusiq-openai?label=openai&color=brightgreen" alt="nucleusiq-openai"></a>
  <a href="https://pypi.org/project/nucleusiq-gemini/"><img src="https://img.shields.io/pypi/v/nucleusiq-gemini?label=gemini&color=brightgreen" alt="nucleusiq-gemini"></a>
  <a href="https://pypi.org/project/nucleusiq-anthropic/"><img src="https://img.shields.io/pypi/v/nucleusiq-anthropic?label=anthropic&color=brightgreen" alt="nucleusiq-anthropic"></a>
  <a href="https://pypi.org/project/nucleusiq-groq/"><img src="https://img.shields.io/pypi/v/nucleusiq-groq?label=groq&color=brightgreen" alt="nucleusiq-groq"></a>
  <a href="https://pypi.org/project/nucleusiq-ollama/"><img src="https://img.shields.io/pypi/v/nucleusiq-ollama?label=ollama&color=brightgreen" alt="nucleusiq-ollama"></a>
  <a href="https://pypi.org/project/nucleusiq-openai-compatible/"><img src="https://img.shields.io/pypi/v/nucleusiq-openai-compatible?label=openai-compatible&color=brightgreen" alt="nucleusiq-openai-compatible"></a>
  <a href="https://pypi.org/project/nucleusiq-mcp/"><img src="https://img.shields.io/pypi/v/nucleusiq-mcp?label=mcp&color=brightgreen" alt="nucleusiq-mcp"></a>
</p>

<!-- Community badges -->
<p align="center">
  <a href="https://github.com/nucleusbox/NucleusIQ/stargazers"><img src="https://img.shields.io/github/stars/nucleusbox/NucleusIQ?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/nucleusbox/NucleusIQ/network/members"><img src="https://img.shields.io/github/forks/nucleusbox/NucleusIQ?style=social" alt="GitHub forks"></a>
  <a href="https://github.com/nucleusbox/NucleusIQ/issues"><img src="https://img.shields.io/github/issues/nucleusbox/NucleusIQ" alt="GitHub issues"></a>
  <a href="https://github.com/nucleusbox/NucleusIQ/pulls"><img src="https://img.shields.io/github/issues-pr/nucleusbox/NucleusIQ" alt="GitHub PRs"></a>
</p>

<!-- Section navigation -->
<p align="center">
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-whats-inside">Features</a> ·
  <a href="#-execution-modes">Execution Modes</a> ·
  <a href="#-packages--ecosystem">Packages</a> ·
  <a href="https://nucleusbox.github.io/nucleusiq-docs/">Docs</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="https://nucleusbox.github.io/nucleusiq-docs/reference/release-notes/v0.7.12/">Release Notes</a> ·
  <a href="#-agent-coverage-map">Agent coverage</a>
</p>

---

> If NucleusIQ helps you build maintainable Python agents, please ⭐ **star the repo** so other engineers can find it while the project is still early.

## 🗺️ Agent coverage map

**[NucleusIQ Agent Coverage — what we ship today vs planned](https://www.nucleusbox.com/nucleusiq-agent-coverage/)** (May 2026, audited on **v0.7.12**)

Interactive scorecard on [Nucleusbox](https://www.nucleusbox.com/): every major agent type (research, multi-agent, coding, and more) with **support today** vs **planned (v0.8–v0.9)** — radars, category tables, 28-pattern matrix, and gap analysis in one page.

| Today (shipped) | Planned (roadmap) |
|-----------------|-------------------|
| 3 Gearbox modes (Direct / Standard / Autonomous) | Agent-as-Tool, structured sub-agent handoff |
| Context engine + telemetry | A2A (thin adapter) |
| Parallel sub-agents + Critic/Refiner | Public benchmark scorecards & Context Report |
| File tools + MCP 0.1.0 Stable | Optional shell tool for coding/terminal benches |

**Best fit now:** research and analysis agents. **Honest gaps:** no LLM-callable AgentTool yet, no published benchmark proof, sub-agent synthesis capped at 2K chars.

## Why Star NucleusIQ?

- **Agent-first Python runtime** — build with `Agent`, `Task`, `@tool`, and typed results instead of chains, graphs, or a custom DSL.
- **Stable provider ecosystem** — OpenAI, Gemini, Anthropic, Groq, Ollama, any OpenAI-compatible server (vLLM, SGLang, TGI, llama.cpp, LM Studio, NIM), MCP, and Mock LLM.
- **Context management before overflow** — NucleusIQ can compact / mask / recall before the LLM API rejects an oversized prompt.
- **Production posture** — 4,382 tests across the monorepo, provider-agnostic observability, usage tracking, plugins, and clear execution modes.

---

## ✨ What's New

> **`nucleusiq` 0.7.13** — September 2026
> Coordinated release adding a generic OpenAI-compatible provider and declaring provider identity in core:
> **`nucleusiq-openai-compatible` 0.1.0 Stable** — bring-your-own-model / bring-your-own-key for any Chat Completions server (vLLM, SGLang, TGI, llama.cpp, LM Studio, NVIDIA NIM, OpenRouter, Together, Fireworks, Azure OpenAI v1) ·
> **`nucleusiq` 0.7.13** — `BaseLLM.PROVIDER_NAME` so identity is declared, not guessed from the class name ·
> **`nucleusiq-openai` 0.7.1** — Responses API usage now reports prompt / completion tokens (streaming and non-streaming) ·
> **`nucleusiq-mcp` 0.1.1** — security floor `mcp>=1.28.1` ·
> Gemini / Anthropic / Groq / Ollama maintenance bumps (`0.3.1` / `0.2.1` / `0.1.1` / `0.2.1`).
>
> - 🧩 **Self-hosted and OpenAI-compatible clouds** — one adapter, no model-name heuristics. Capabilities come from an engine preset (`vllm`, `sglang`, `tgi`, `llamacpp`, `lmstudio`, `ollama`, `generic`) or an explicit override.
> - 🪪 **Declared provider identity** — `get_provider_from_llm()` reads `PROVIDER_NAME` first. Without it, `OpenAICompatibleLLM` would match as `"openai"` and send OpenAI-cloud structured output to a self-hosted server.
> - 🔧 **Live-endpoint fixes** — flat tool-call sanitization, streaming `tool_calls` on COMPLETE, and inbound `response_format` routed through the structured-output policy (vLLM tool-suppression workaround).
>
> Previous coordinated line: [v0.7.12](CHANGELOG.md#0712--2026-05-26) (first stable provider matrix + native-tool observability). See [CHANGELOG.md](CHANGELOG.md) for the full notes.

---

## What Is NucleusIQ?

NucleusIQ is an **open-source, agent-first Python framework** for building AI agents that **survive production** — long runs, messy tool output, and real stakes — instead of falling apart the moment they leave the demo.

**In one line:**

> Every AI agent works in the demo. Almost none survive production. NucleusIQ is the framework for the ones that have to — reliable, observable, and provider-portable.

**Why "survive production"?** Most agents work in a demo, then forget what they read, choke when tool output piles up, or fail silently with no explanation. NucleusIQ manages the context window before the model breaks, runs across any LLM provider, and returns a scorecard of exactly what happened on every run.

NucleusIQ is built on a simple belief:

> An agent is not a single model call. An agent is a managed runtime with memory, tools, policy, streaming, structure, and responsibilities.

### NucleusIQ Philosophy

A shared doctrine for what NucleusIQ stands for, why it exists, and how it should evolve over time.

See **[NucleusIQ Philosophy](https://nucleusbox.github.io/nucleusiq-docs/reference/nucleusiq-philosophy/)**.

---

## 🚀 Quick Start

### Fastest path

```bash
pip install nucleusiq nucleusiq-openai
export OPENAI_API_KEY=sk-...
```

### Hello agent

```python
import asyncio

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq_openai import BaseOpenAI


async def main() -> None:
    agent = Agent(
        name="analyst",
        prompt=ZeroShotPrompt().configure(
            system="You are a concise assistant. Answer in one short sentence.",
        ),
        llm=BaseOpenAI(model_name="gpt-4o-mini"),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    await agent.initialize()
    result = await agent.execute(
        Task(id="hello-1", objective="What is the capital of France?"),
    )
    print(result.output)


asyncio.run(main())
```

See the [Quickstart docs](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/quickstart/) for provider setup, `.env` loading, tools, streaming, and structured output.

### Install other stable packages

```bash
# Google Gemini
pip install nucleusiq nucleusiq-gemini

# Anthropic Claude
pip install nucleusiq nucleusiq-anthropic

# Groq inference
pip install nucleusiq nucleusiq-groq

# Ollama for local / remote models (native Ollama API)
pip install nucleusiq nucleusiq-ollama

# Any OpenAI-compatible server — vLLM, SGLang, TGI, llama.cpp, LM Studio, NIM, …
pip install nucleusiq nucleusiq-openai-compatible

# MCP tool adapter — plug any MCP server in as a tool
pip install nucleusiq-mcp nucleusiq-anthropic   # or any provider

# uv works too
uv pip install nucleusiq nucleusiq-openai
```

### Hello agent + MCP tools

```python
import asyncio

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq_anthropic import BaseAnthropic
from nucleusiq_mcp import MCPTool


async def main() -> None:
    agent = Agent(
        name="researcher",
        prompt=ZeroShotPrompt().configure(
            system="You are a careful research assistant. Cite source ids when available.",
        ),
        llm=BaseAnthropic(model_name="claude-sonnet-4-5-20250929", async_mode=True),
        tools=[
            # Transport is auto-detected from URL / command; auth is auto-wired.
            MCPTool("npx -y @modelcontextprotocol/server-github"),
            MCPTool("https://mcp.slack.com/api", auth="xoxb-..."),
        ],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD, enable_tracing=True),
    )

    await agent.initialize()  # connects to MCP servers and discovers tools
    result = await agent.execute(
        Task(
            id="repo-summary",
            objective="Summarise the last 5 issues in nucleusbox/NucleusIQ.",
        ),
    )
    print(result.output)


asyncio.run(main())
```

See [INSTALLATION.md](INSTALLATION.md) for full setup instructions (pip, uv, development mode).

---

## 🧩 What's Inside

| Component | What it does |
|-----------|-------------|
| **3 Execution Modes** | `DIRECT` (single call), `STANDARD` (tool loop), `AUTONOMOUS` (orchestration + validation + retry) |
| **Streaming** | `execute_stream()` — real-time token-by-token output with tool call visibility across all modes |
| **7 Prompt Techniques** | ZeroShot, FewShot, ChainOfThought, AutoCoT, RAG, PromptComposer, MetaPrompt |
| **Multimodal Attachments** | 7 attachment types (text, PDF, images, files) with provider-native optimisation |
| **Built-in File Tools** | `FileReadTool`, `FileSearchTool`, `DirectoryListTool`, `FileExtractTool` — sandboxed to workspace |
| **Tool System** | `BaseTool` interface + `@tool` decorator + provider native tools (OpenAI: code_interpreter, file_search, web_search; Gemini: Google Search, Code Execution, URL Context, Maps; Anthropic: web_search, web_fetch, code_execution + extended thinking) |
| **MCP Tool Adapter** | Connect any **Model Context Protocol** server (Slack, GitHub, Postgres, Stripe, …) as native tools — stdio + Streamable HTTP + SSE; OAuth/Bearer/Env auth |
| **Memory** | 5 strategies (full history, sliding window, summary, summary+window, token budget) with file-aware metadata |
| **Plugins** | 10 built-in: call limits, retry, fallback, PII guard, human approval, tool guard, attachment guard, context window, result validator |
| **Usage Tracking** | Token usage per call with purpose tagging (main, planning, tool loop, critic, refiner) and cost estimation |
| **Structured Output** | Schema-based output parsing with Pydantic, dataclass, TypedDict support |
| **Observability** | `ExecutionTracer` records every model call + tool call with `source` attribution (e.g. `mcp://server=github`) |
| **Provider Portability** | Swap providers (OpenAI, Gemini, Anthropic, Groq, Ollama, OpenAI-compatible, …) with one line — same agent code, same tools, same plugins |

---

## 🧠 Context Management

Tool-heavy agents fail when every tool result stays in the active prompt forever. NucleusIQ treats context as a managed runtime resource:

- `ContextEngine.prepare()` runs **before** LLM calls, not after the provider rejects an oversized prompt.
- `ContextLedger` tracks prompt regions (system, user, assistant, tool calls, tool results) so the framework can compact the right thing first.
- Large tool results can be masked / offloaded while staying recoverable through recall.
- `AgentResult.context_telemetry` reports peak utilization, compaction events, tokens saved, and estimated savings.

See the [context management guide](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/context-management/) and the [observability guide](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/observability/).

Try the runnable [Agent Engineering Challenge 01](notebooks/agents/agent_engineering_challenge/) to test context pressure, noisy tool outputs, and evidence quality on a concrete task.

---

## ⚙️ Execution Modes

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

---

## 📦 Packages & Ecosystem

NucleusIQ ships as a **core framework + thin provider/tool packages**. Install only what you need — every package can be added or removed independently.

### Core

| Package | Status | Version | Description |
|---|---|---|---|
| [`nucleusiq`](https://pypi.org/project/nucleusiq/) | 🟢 Stable | `0.7.13` | Core framework: agents, prompts, tools, memory, plugins, modes, tracing |

### LLM Providers

| Package | Status | Version | Description |
|---|---|---|---|
| [`nucleusiq-openai`](https://pypi.org/project/nucleusiq-openai/) | 🟢 Stable | `0.7.1` | OpenAI (gpt-4o, o-series); Responses API + Chat Completions; native `code_interpreter`, `file_search`, `web_search`; Responses usage mapped to `prompt_tokens` / `completion_tokens` · [README](src/providers/llms/openai/README.md) |
| [`nucleusiq-gemini`](https://pypi.org/project/nucleusiq-gemini/) | 🟢 Stable | `0.3.1` | Google Gemini; native Google Search + Code Execution emitted as `ToolCallRecord(executed_by="provider")`; URL Context, Maps grounding · [README](src/providers/llms/gemini/README.md) |
| [`nucleusiq-anthropic`](https://pypi.org/project/nucleusiq-anthropic/) | 🟢 Stable | `0.2.1` | Anthropic Claude (Messages API); **native server tools** (`AnthropicTool.web_search()` / `web_fetch()` / `code_execution()` w/ auto-`anthropic-beta`), **prompt caching** (`cache_tools` / `cache_system`), **extended thinking** (`thinking="low"\|"medium"\|"high"\|"max"`), **server-tool observability** · [README](src/providers/llms/anthropic/README.md) |

### Inference Backends

| Package | Status | Version | Description |
|---|---|---|---|
| [`nucleusiq-groq`](https://pypi.org/project/nucleusiq-groq/) | 🟢 Stable | `0.1.1` | Groq inference (Chat Completions) via official `groq` SDK; hosted-tool observability stub (`message.executed_tools` → `server_tool_calls`) · [README](src/providers/inference/groq/README.md) · [Guide](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/groq-provider/) |
| [`nucleusiq-ollama`](https://pypi.org/project/nucleusiq-ollama/) | 🟢 Stable | `0.2.1` | Local/remote Ollama via official `ollama` SDK; **vision wire** for OpenAI-style multimodal messages; structured output, `think` pass-through · [README](src/providers/inference/ollama/README.md) · [Guide](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/ollama-provider/) |
| [`nucleusiq-openai-compatible`](https://pypi.org/project/nucleusiq-openai-compatible/) | 🟢 Stable | `0.1.0` | **BYOM / BYOK** — any OpenAI Chat Completions server (vLLM, SGLang, TGI, llama.cpp, LM Studio, NIM, OpenRouter, Together, …). Declared capabilities, no model-name heuristics. Requires `nucleusiq>=0.7.13` · [README](src/providers/inference/openai_compatible/README.md) |

### Tool Adapters

| Package | Status | Version | Description |
|---|---|---|---|
| [`nucleusiq-mcp`](https://pypi.org/project/nucleusiq-mcp/) | 🟢 Stable | `0.1.1` | **Model Context Protocol** adapter — turn any MCP server (Slack, GitHub, Postgres, Stripe, …) into NucleusIQ tools; stdio + Streamable HTTP + SSE; OAuth 2.1 / Bearer / Env auth · [README](src/providers/tools/mcp/README.md) · [Guide](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/mcp-integration/) |

**Maturity legend:** 🟢 Stable (production-ready, SemVer guarantees). Future pre-release packages may use 🟡 Beta / 🟠 Alpha while they mature.

---

## 🗂️ Project Structure

```
src/
  nucleusiq/core/                # Core framework (agents, prompts, tools, memory, plugins, modes, tracing)
  providers/
    llms/
      openai/                    # nucleusiq-openai
      gemini/                    # nucleusiq-gemini
      anthropic/                 # nucleusiq-anthropic
    inference/
      groq/                      # nucleusiq-groq
      ollama/                    # nucleusiq-ollama
      openai_compatible/         # nucleusiq-openai-compatible (vLLM, SGLang, TGI, …)
    tools/
      mcp/                       # nucleusiq-mcp (Model Context Protocol adapter)
notebooks/agents/                # Example notebooks (PE due diligence, MCP showcase, …)
docs/                            # Internal design/strategy docs (published docs live in nucleusiq-docs)
scripts/                         # Repo-wide tooling (e.g. verify_core_package_layout.py)
```

---

## 🧪 Testing

```bash
# Monorepo: verify core setuptools packages + all Hatch provider/tool wheel roots
# (including nucleusiq-openai-compatible). Fails if a new provider is added on
# disk and not registered in the script.
python scripts/verify_core_package_layout.py

# Isolated import of each provider from its declared dependencies alone
# (separate CI job: dependency-completeness)
python scripts/verify_dependency_completeness.py

# Core tests (2,607 passing; 8 live OpenAI memory tests deselected)
cd src/nucleusiq && python -m pytest tests/ -q

# OpenAI provider tests (239 passing)
cd src/providers/llms/openai && python -m pytest tests/ -q

# Gemini provider unit tests (292 passing)
cd src/providers/llms/gemini && python -m pytest tests/unit/ -q

# Anthropic provider tests (>=95% coverage gate)
cd src/providers/llms/anthropic && python -m pytest tests/ -q

# Groq provider tests (requires dev group / uv; >=90% coverage gate)
cd src/providers/inference/groq && uv run pytest -q

# Ollama provider tests (>=95% coverage gate)
cd src/providers/inference/ollama && uv run pytest -q

# OpenAI-compatible provider tests (>=95% coverage gate; 675 passing)
cd src/providers/inference/openai_compatible && uv run pytest -q -m "not live"

# MCP tool adapter — unit (235 passing; >=90% gate)
cd src/providers/tools/mcp && python -m pytest tests/unit/ -q -m "not integration"

# MCP tool adapter — live integration (requires Node.js + npx)
cd src/providers/tools/mcp && python -m pytest tests/integration/ -m integration -v

# Gemini integration tests (requires GEMINI_API_KEY)
cd src/providers/llms/gemini && python -m pytest tests/integration/ -q
```

---

## 📚 Documentation

- **Published docs** — https://nucleusbox.github.io/nucleusiq-docs/
- **Docs repository** — https://github.com/nucleusbox/nucleusiq-docs
- [INSTALLATION.md](INSTALLATION.md) — Setup instructions (pip, uv, development)
- [CHANGELOG.md](CHANGELOG.md) — Release notes (**v0.7.13** is current)
- [RELEASE.md](RELEASE.md) — Release process and branching strategy
- [v0.7.12 release notes](https://nucleusbox.github.io/nucleusiq-docs/reference/release-notes/v0.7.12/) — previous coordinated stable line
- [Provider guides](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/) — OpenAI, Gemini, Anthropic, Groq, Ollama, MCP
- [OpenAI-compatible provider](src/providers/inference/openai_compatible/README.md) — self-hosted / BYOM (docs site page pending)
- [MCP integration guide](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/mcp-integration/) — MCP adapter usage
- [File handling guide](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/file-handling/) — Attachment vs Tool vs Both decision guide

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b yourname/my-feature main`
3. Make your changes and add tests
4. Submit a pull request to `main`

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details, coding standards, and the dev-setup walkthrough.

### Get in touch

- 🐛 Bugs & feature requests — [GitHub Issues](https://github.com/nucleusbox/NucleusIQ/issues)
- 💬 Questions & ideas — [GitHub Discussions](https://github.com/nucleusbox/NucleusIQ/discussions)
- ⭐ If NucleusIQ is useful to you, please consider starring the repo — it helps a lot.

---

## 📄 License

[MIT](LICENSE) © Nucleusbox
