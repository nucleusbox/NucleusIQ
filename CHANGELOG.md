# Changelog

All notable changes to NucleusIQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] — 2026-02-27

### Added

- **End-to-end streaming** via `Agent.execute_stream()` — async generator yielding `StreamEvent` objects with real-time token-by-token output across all 3 execution modes
- **`StreamEvent` + `StreamEventType`** data model (`core/streaming/events.py`) — 8 event types: `TOKEN`, `TOOL_CALL_START`, `TOOL_CALL_END`, `LLM_CALL_START`, `LLM_CALL_END`, `THINKING`, `COMPLETE`, `ERROR`
- **`BaseLLM.call_stream()`** — abstract streaming contract with non-streaming fallback; `MockLLM.call_stream()` for testing
- **`BaseOpenAI.call_stream()`** — OpenAI provider streaming for both Chat Completions and Responses API backends
- **`stream_adapters.py`** — adapter layer converting raw OpenAI SDK chunks/SSE events into framework `StreamEvent` objects
- **Streaming in all execution modes** — `DirectMode.run_stream()`, `StandardMode.run_stream()`, `AutonomousMode.run_stream()` with reusable `_streaming_tool_call_loop()` in base mode
- **Usage telemetry** in `_LLMResponse` — `usage` (prompt/completion/reasoning tokens), `id`, `model`, `created`, `service_tier`, `system_fingerprint`
- **Streaming example** — `examples/agents/streaming_example.py` demonstrating all 3 modes
- **21 cross-milestone integration tests** — full-stack streaming from Agent to MockLLM
- 221 new tests across streaming, coverage boost, and edge cases

### Fixed

- **Chat Completions streaming** — accumulate all chunks instead of returning only first delta
- **Responses API streaming** — handle SSE event iterator instead of awaiting single response
- **Multimodal content normalization** — `_messages_to_responses_input()` now preserves content arrays for vision/audio/file inputs (previously stringified them)
- **Metadata extraction** — filter non-primitive types from `_extract_response_metadata` to prevent test mock leakage

### Changed

- Bumped `nucleusiq` to 0.3.0, `nucleusiq-openai` to 0.3.0
- `nucleusiq-openai` now requires `nucleusiq>=0.3.0`
- `Agent.execute()` internals refactored — extracted `_resolve_mode()` and `_setup_execution()` (shared with `execute_stream()`)
- `ChatCompletionsPayload.build()` uses `model_fields` set lookup instead of brittle `hasattr(cls, k)`

### Testing

- **1,544 tests passing** (1,382 core + 162 OpenAI provider, 2 skipped)
- **97% coverage** on both packages
- All files above 90% coverage; previously sub-90% files boosted to 95%+

---

## [0.2.0] — 2026-02-25

### Added

- **Configurable tool limits per execution mode**: Direct (5), Standard (30), Autonomous (100) — configurable via `AgentConfig.max_tool_calls`
- **Tool support in DirectMode** — up to 5 tool calls (previously no tools)
- **Critic/Refiner integration in AutonomousMode** — replaces simple LLM review (Layer 3) and generic retry with independent verification and targeted correction
- **Tool limit validation** — agent raises `ValueError` at execution time if more tools are configured than the mode allows
- **`AgentConfig.get_effective_max_tool_calls()`** — centralized method for mode-aware tool limits
- 198 new tests for tool limits, DirectMode tool support, and Critic/Refiner flow

### Removed

- **Deprecated `planning/` module** — `PlanCreator`, `PlanExecutor`, `PlanParser`, `Planner`, `PlanPromptStrategy`, `schema` (~1,200 lines). Autonomous mode uses `Decomposer` for task breakdown instead.
- Removed `AgentConfig` fields: `use_planning`, `planning_max_tokens`, `planning_timeout`
- Removed 428 planning-related tests (`test_planning_coverage.py`)

### Changed

- Migrated repository to `nucleusbox` GitHub organization
- Simplified branching strategy to GitHub Flow (single `main` branch)
- Upgraded issue templates to YAML forms (bug report, feature request, question)
- Added CONTRIBUTING.md with full development guide
- Streamlined RELEASE.md and removed obsolete FIRST_RELEASE_TODO.md
- StandardMode now uses `AgentConfig.get_effective_max_tool_calls()` (default 30) instead of internal constant
- Updated README with execution modes comparison table
- Updated all examples and docs to remove planning references

### Testing

- **1,323 tests passing** (1,207 core + 116 OpenAI provider, 2 skipped)

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

### Planned for v0.4.0

- Agent Types: ReAct integration into mode system, Chain-of-Thought as config flag
- Framework-level multimodal inputs (Task.attachments for images, files)

### Planned for v0.5.0

- Gemini provider (`nucleusiq-gemini`)
- Ollama provider (`nucleusiq-ollama`)
- Embeddings API
- UsageTracker (token + cost estimation)

[0.3.0]: https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.3.0
[0.2.0]: https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.2.0
[0.1.0]: https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.1.0
[Unreleased]: https://github.com/nucleusbox/NucleusIQ/compare/v0.3.0...HEAD
