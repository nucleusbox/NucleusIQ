# Changelog

All notable changes to NucleusIQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-02-24

### Changed

- Migrated repository to `nucleusbox` GitHub organization
- Simplified branching strategy to GitHub Flow (single `main` branch)
- Upgraded issue templates to YAML forms (bug report, feature request, question)
- Added CONTRIBUTING.md with full development guide
- Streamlined RELEASE.md and removed obsolete FIRST_RELEASE_TODO.md

---

## [0.1.0] — 2026-02-24

**Initial public release** of the NucleusIQ framework and OpenAI provider.

### Packages

| Package | Version | PyPI |
|---------|---------|------|
| `nucleusiq` | 0.1.0 | [nucleusiq](https://pypi.org/project/nucleusiq/) |
| `nucleusiq-openai` | 0.1.0 | [nucleusiq-openai](https://pypi.org/project/nucleusiq-openai/) |

### Agent System

- **3 Execution Modes** via Strategy Pattern:
  - `DIRECT` — single LLM call, no tools
  - `STANDARD` — LLM + tool-calling loop
  - `AUTONOMOUS` — orchestration with parallel execution, external validation, structured retry, and progress tracking
- **Autonomous Mode** with `ValidationPipeline` (3-layer validation: tool checks → plugin validators → optional LLM review), `ProgressTracker`, and `Decomposer` for complex task parallelization
- **`ResultValidatorPlugin`** — abstract base class for domain-specific external validation (the framework orchestrates, the LLM executes, external signals validate)
- **Planning System** — `Planner` facade with `PlanCreator`, `PlanExecutor`, `PlanParser`, and `PlanPromptStrategy` (Protocol + DI)
- **ReAct Agent** — Reasoning + Acting pattern implementation
- **Structured Output** — schema-based output parsing and validation
- **`AgentConfig`** — Pydantic configuration with execution mode, retry settings, and sub-agent limits

### Prompt Engineering

- **7 Prompt Techniques**: `ZeroShot`, `FewShot`, `ChainOfThought`, `AutoChainOfThought`, `RetrievalAugmentedGeneration`, `PromptComposer`, `MetaPrompt`
- **`PromptFactory`** — create prompts by technique name via `PromptTechnique` enum

### Tool System

- **`BaseTool`** — LLM-agnostic tool interface with JSON schema generation
- **`BaseTool.from_function()`** — create tools from plain Python functions
- **OpenAI native tools**: `function`, `code_interpreter`, `file_search`, `web_search`, `mcp`, `connector` (via `OpenAITool`)

### Memory System

- **5 Memory Strategies** via `MemoryFactory`:
  - `FullHistoryMemory` — keep all messages
  - `SlidingWindowMemory` — keep last N messages
  - `SummaryMemory` — summarize older messages via LLM
  - `SummaryWindowMemory` — sliding window + summary of dropped messages
  - `TokenBudgetMemory` — keep messages within token budget

### Plugin System

- **`BasePlugin`** ABC with typed request models (`ModelRequest`, `ToolRequest`, `AgentContext`)
- **`PluginManager`** — chain-of-responsibility hook pipeline
- **Decorator API** — `@before_agent`, `@after_agent`, `@before_model`, `@after_model`, `@wrap_model_call`, `@wrap_tool_call`
- **9 Built-in Plugins**:

| Plugin | Purpose |
|--------|---------|
| `ModelCallLimitPlugin` | Limits LLM call count per execution |
| `ToolCallLimitPlugin` | Limits tool call count |
| `ToolRetryPlugin` | Retries failed tools with exponential backoff |
| `ModelFallbackPlugin` | Tries fallback models on primary failure |
| `PIIGuardPlugin` | Detects/redacts/masks/blocks PII |
| `HumanApprovalPlugin` | Human approval gate with `ApprovalHandler` pattern |
| `ContextWindowPlugin` | Trims messages to fit context window |
| `ToolGuardPlugin` | Tool whitelist/blacklist |
| `ResultValidatorPlugin` | Abstract base for domain-specific result validation |

### LLM Provider — OpenAI (`nucleusiq-openai`)

- **Chat Completions API** — full support with tool calling
- **Responses API** — automatic routing based on tool types
- **`OpenAILLMParams`** — type-safe parameters with typo detection and merge chain (LLM defaults < AgentConfig < per-execute overrides)
- **6 Native Tool Types** — function, code_interpreter, file_search, web_search_preview, mcp, connector
- **Structured Output** — JSON schema enforcement via `response_format`

### Testing

- **1358 tests passing** (1242 core + 116 OpenAI provider, 2 skipped)
- 98% plugin system branch coverage

### Documentation & Examples

- `notebooks/agents/pe_due_diligence.ipynb` — end-to-end autonomous agent demo with 8 PE due diligence scenarios
- 17 core examples + 28 OpenAI provider examples

---

## [Unreleased]

### Planned for v0.2.0

- Streaming support (`execute_stream()`)
- Usage / token tracking
- Agent-level streaming with tool calls

### Planned for v0.3.0

- Multimodal inputs (vision, audio)
- Gemini provider (`nucleusiq-gemini`)
- Ollama provider (`nucleusiq-ollama`)

[0.2.0]: https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.2.0
[0.1.0]: https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.1.0
[Unreleased]: https://github.com/nucleusbox/NucleusIQ/compare/v0.2.0...HEAD
