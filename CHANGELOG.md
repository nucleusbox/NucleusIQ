# Changelog

All notable changes to NucleusIQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.7.4](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.4) — 2026-04-03

### Added

- **`observability` package** — dedicated `nucleusiq.agents.observability` package with SRP file layout:
  - `protocol.py` — `ExecutionTracerProtocol` (`@runtime_checkable`)
  - `default_tracer.py` — `DefaultExecutionTracer` (in-memory, `__slots__` optimised)
  - `noop_tracer.py` — `NoOpTracer` (Null Object, zero overhead)
  - `record_builders.py` — `build_tool_call_record`, `build_llm_call_record`, `build_llm_call_record_from_stream`
  - `_response_parser.py` — `extract_tool_calls` (OpenAI + Gemini), `safe_int`, `usage_dict_from_response`
- **Agent wiring** — each `execute()` / `execute_stream()` run resets a fresh `DefaultExecutionTracer` on the agent; `BaseExecutionMode.call_llm` / `call_tool` / `_streaming_tool_call_loop` record timings and outcomes; `AgentResult` now receives `tool_calls`, `llm_calls`, `plugin_events`, `memory_snapshot`, `autonomous`, and `warnings` from the tracer where populated.
- **All 3 execution modes traced** — Direct, Standard, and Autonomous (including Critic LLM calls with `purpose="critic"`). Known gap: `Decomposer.analyze()` bypasses the tracer (will be wired in v0.7.6).
- **Tests** — `tests/unit/test_execution_tracer.py`, `tests/agents/unit/test_agent_tracer_integration.py`, `tests/agents/unit/test_autonomous_tracer_integration.py`.
- **`integration_test/run_integration.py`** — version check for `0.7.4` plus tracer smoke assertions.
- **`AgentConfig.enable_tracing`** — `bool = False`. When off (default), `AgentResult` trace fields are empty tuples with zero overhead. When on, `DefaultExecutionTracer` captures all LLM/tool calls, durations, and warnings.
- **Pyrefly static type checking (Meta, MIT)** — integrated across all packages as Python's "compile-time" safety layer:
  - `[tool.pyrefly]` config in `pyproject.toml` for core, OpenAI, and Gemini.
  - All **121 type errors** fixed across core (71), OpenAI (33), Gemini (17) — null safety guards, undeclared dynamic attributes, type annotation corrections, override signature fixes.
  - Declared `_last_messages` and `_execution_progress` as `PrivateAttr` on `Agent`.
  - Fixed `_plugin_manager` type to `PluginManager | None`.
  - Fixed `build_call_kwargs` return type, `responses_api` tuple types, `ChatCompletionsPayload` construction.
  - Pyrefly added to `dependency-groups.lint` (dev-only, not shipped in wheel).
  - CI `type-check` job gates the build: installs with `editable_mode=strict` then runs `pyrefly check` for all 3 packages.
- **`core/errors/` package** — converted from single `errors.py` to a proper package: `base.py` defines `NucleusIQError` (cycle-free), `__init__.py` provides lazy `__getattr__` re-exports of all 40+ error types. All 9 subsystem error modules updated to import from `nucleusiq.errors.base`. Backward-compatible: `from nucleusiq.errors import NucleusIQError` still works.
- **`core/agents/usage/` package** — extracted `usage_tracker.py` and `pricing.py` from `components/` into a dedicated `usage/` package with public `__init__.py` re-exports. Old shim files deleted (no backward-compatibility wrappers — clean break).
- **Exhaustive error wiring** — every `raise ValueError` / `raise RuntimeError` in production code audited and replaced with proper custom error types:
  - **Agent modes**: `AgentExecutionError` in `standard_mode.run()`, `direct_mode.run()` (replaces bare `except Exception` string returns)
  - **Agent lifecycle**: `AgentExecutionError` and `AgentTimeoutError` in `base_agent.py` retry loop (replaces `RuntimeError`)
  - **LLM validation**: `LLMError` in `base_mode.validate_response()` and `react_agent.py` (replaces `ValueError`)
  - **Tools**: `ToolValidationError` in `decorators.py` and `base_tool.py` (replaces `TypeError`); `ToolExecutionError` in `tool_retry.py`
  - **Plugins**: `PluginError` in all 6 built-in plugins (`context_window`, `tool_guard`, `pii_guard`, `attachment_guard`, `human_approval`, `model_fallback`); `PluginExecutionError` in `validation.py`
  - **Prompts**: `PromptTemplateError` and `PromptConfigError` across `base.py`, `prompt_composer.py`, `meta_prompt.py`, `auto_chain_of_thought.py`, `few_shot.py`, `chain_of_thought.py`, `retrieval_augmented_generation.py` (all runtime methods; Pydantic validators correctly remain `ValueError`)
  - **Structured output**: `StructuredOutputError` in `config.py`; `SchemaValidationError` in `parser.py`; `StructuredOutputError`/`SchemaParseError` in provider parsers
  - **Attachments**: `AttachmentProcessingError` for base64 failures; `AttachmentUnsupportedError` in OpenAI provider
  - **Provider auth**: `AuthenticationError` for missing API keys in both OpenAI and Gemini providers
  - **Provider retry**: `ContentFilterError` and `ContextLengthError` mapped in both OpenAI and Gemini `retry.py` modules
  - **Provider tools**: `ToolValidationError` for OpenAI MCP tool config validation

### Changed

- **`_setup_execution`** — usage tracker and execution tracer are reset immediately after plugin counter reset (before `BEFORE_AGENT`), so halted or failed setups do not leak prior-run tracer data. Tracer creation gated on `AgentConfig.enable_tracing`.
- All internal imports updated to canonical paths (`nucleusiq.agents.usage.*`, `nucleusiq.errors.base`).
- `components/usage_tracker.py` and `components/pricing.py` shim files **deleted** — all imports now use canonical `nucleusiq.agents.usage.*` paths.
- `standard_mode.run()` and `direct_mode.run()` now raise `AgentExecutionError` with mode context instead of returning error strings.
- `base_agent.py` retry loop raises `AgentExecutionError`/`AgentTimeoutError` instead of `RuntimeError`/`TimeoutError`.
- OpenAI provider: `AuthenticationError` instead of `ValueError` for missing API key; `AttachmentUnsupportedError` for unknown attachment types; `ToolValidationError` for MCP config; `StructuredOutputError`/`SchemaParseError` for structured output parsing; `ContentFilterError`/`ContextLengthError` in retry.
- Gemini provider: `AuthenticationError` instead of `ValueError` for missing API key; `StructuredOutputError`/`SchemaParseError` for structured output parsing; `ContentFilterError`/`ContextLengthError` in retry.

### Developer Support

- **Pyrefly type checking** — CI pipeline now includes a `type-check` job using [Pyrefly](https://pyrefly.org/) (Meta, MIT license, Rust-based, 1.85M lines/sec). Catches undefined names, null safety violations, type mismatches, and override signature inconsistencies at "compile time" — before tests or deployment.
  - CI flow: `ruff check` → `ruff format` → **`pyrefly check`** → `pytest` → `import-check` → `security` → `build`
  - Added to `dependency-groups.lint` alongside `ruff` (dev-only, not in wheel).
  - Requires `editable_mode=strict` install for `setuptools` `package-dir` mapping resolution.
  - Provider configs use `ignore-missing-imports` for cross-package deps (`nucleusiq.*`, `google.*`) — these resolve at CI time when all packages are installed; locally they gracefully fall back to `Any`.

### Packages

| Package | Version | Note |
|---------|---------|------|
| `nucleusiq` | **0.7.4** | ExecutionTracer, configurable tracing, error/usage package restructure, exhaustive error wiring |
| `nucleusiq-openai` | **0.6.1** | Custom error types wired, ContentFilter/ContextLength mapped (requires `nucleusiq>=0.7.4`) |
| `nucleusiq-gemini` | **0.2.2** | Custom error types wired, ContentFilter/ContextLength mapped (requires `nucleusiq>=0.7.4`) |

---

## [0.7.3](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.3) — 2026-04-02

### Fixed

- **Gemini `function_response.name` cannot be empty** — tool result messages (`role="tool"`) now carry `name=tc.name` in `ChatMessage` across all execution modes (standard, direct, base streaming). Previously, `name` was `None`, causing Gemini API 400 errors on the second turn of any tool-calling conversation.
- **Gemini `function_response.response` must be dict** — `response_normalizer.py` now wraps non-dict payloads (e.g. `json.dumps("string")`) in `{"result": ...}` to satisfy the `google-genai` SDK's Pydantic validation.
- **Defense-in-depth name inference** — if `name` is still missing from an incoming tool message dict, the Gemini response normalizer infers it from the prior assistant message's `tool_calls` by matching `tool_call_id`.
- **Gemini `$ref`/`$defs` inlining** — `_clean_schema` in structured output builder now inlines `$ref` references (matching OpenAI's cleaner quality). Previously, `$defs` were silently dropped, producing broken schemas for nested Pydantic models.

### Added

- **Tools + structured output guard** — `BaseGemini.call()` detects when both `tools` and `response_format` are set, logs a warning, and drops JSON schema mode. Gemini API rejects `response_mime_type: application/json` combined with function calling; this guard prevents the 400 error.
- **`OutputSchema` tuple handling** — `BaseGemini.call()` now unpacks the `(provider_format, schema)` tuple from core's `StructuredOutputHandler`, matching the OpenAI provider's behavior.
- **Gemini integration tests: `test_gemini_tool_round_trip.py`** — full multi-turn tool loop with tools resent on second call, JSON string content round-trip, multiple tool calls, structured output + tools guard test.
- **OpenAI integration test scaffold** — `tests/integration/` directory created with `conftest.py` + `test_openai_tool_round_trip.py` mirroring Gemini's integration test structure. Uses `@pytest.mark.integration` (previously defined but never applied in OpenAI tests).
- **Nested Pydantic model unit tests** — `test_nested_pydantic_model_refs_inlined` and `test_deeply_nested_refs_inlined` verify `$ref` inlining works for 2+ levels of model nesting.

### Changed

- **Notebook: `research_analyst_tcs.ipynb`** — rewritten as a complete framework showcase demonstrating all core features (tools, memory, plugins, streaming, structured output, cost tracking) with pandas DataFrames, matplotlib visualizations, and a feature proof dashboard.

### Packages

| Package | Version | Note |
|---------|---------|------|
| `nucleusiq` | **0.7.3** | Tool message `name` field fix |
| `nucleusiq-openai` | 0.6.0 | No change |
| `nucleusiq-gemini` | **0.2.1** | All Gemini fixes above (requires `nucleusiq>=0.7.3`) |

---

## [0.7.2](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.2) — 2026-03-31

### Added

- **Unified exception hierarchy** — All framework errors now inherit from `NucleusIQError`. New error hierarchies for every subsystem:
  - `ToolError` (ToolExecutionError, ToolTimeoutError, ToolValidationError, ToolPermissionError, ToolNotFoundError)
  - `AgentError` (AgentConfigError, AgentExecutionError, AgentTimeoutError)
  - `AttachmentError` (AttachmentValidationError, AttachmentProcessingError, AttachmentUnsupportedError)
  - `NucleusMemoryError` (MemoryWriteError, MemoryReadError, MemoryImportError, MemoryCapacityError)
  - `PromptError` (PromptTemplateError, PromptConfigError, PromptGenerationError)
  - `StreamingError` (StreamInterruptedError, StreamOrchestrationError)
  - `ContextLengthError` added to LLMError hierarchy
  - `PluginExecutionError` added to PluginError hierarchy
- **`AgentResult` response contract** — `Agent.execute()` now returns a typed, immutable `AgentResult` (Pydantic `BaseModel`, `frozen=True`) instead of raw `Any`. Includes: `output`, `status` (SUCCESS/ERROR/HALTED), `error`, `error_type`, `duration_ms`, `agent_id`, `agent_name`, `task_id`, `mode`, `model`, `created_at`, `usage`, and extension fields for future observability (`tool_calls`, `llm_calls`, `plugin_events`, etc.).
- **Backward compatible** — `str(result)` returns the output text, `bool(result)` returns `True` on success. Existing `print(result)` and `if result:` patterns continue to work.

### Changed

- **Re-parented existing errors** — `PluginError`, `PluginHalt`, `StructuredOutputError`, `WorkspaceSecurityError` now extend `NucleusIQError` instead of bare `Exception`. Enables `except NucleusIQError` catch-all.
- `NucleusIQError` canonical location moved to `nucleusiq.errors` (re-exported from `nucleusiq.llms.errors` for backward compat).
- All error classes now carry structured context attributes (e.g. `tool_name`, `provider`, `status_code`, `mode`, `task_id`).
- `Agent.execute()` catches all exceptions and wraps them in `AgentResult(status="error")` — no more unhandled exceptions from `execute()`.

### Packages

| Package | Version | Note |
|---------|---------|------|
| `nucleusiq` | **0.7.2** | Exception hierarchy + AgentResult |
| `nucleusiq-openai` | 0.6.0 | No change |
| `nucleusiq-gemini` | 0.2.0 | No change |

---

## [0.7.1](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.1) — 2026-03-30

### Fixed

- **`.env` loading broken for pip-installed consumers** — `core/__init__.py` used a hard-coded `Path(__file__).parents[2]` to locate `.env`, which resolved into `site-packages/` for anyone who `pip install nucleusiq`. Replaced with `load_dotenv(override=False)` (no path argument), which uses `python-dotenv`'s built-in `find_dotenv()` to search from the caller's **working directory** upward. Any project with a `.env` in its root now works out of the box.

### Packages

| Package | Version | Note |
|---------|---------|------|
| `nucleusiq` | **0.7.1** | Patch fix for `.env` loading |
| `nucleusiq-openai` | 0.6.0 | No change (requires `nucleusiq>=0.7.0`, accepts 0.7.1) |
| `nucleusiq-gemini` | 0.2.0 | No change (requires `nucleusiq>=0.7.0`, accepts 0.7.1) |

---

## [0.7.0](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.0) — 2026-03-30

### Security

- **`requests`** — minimum raised to `>=2.33.0` (CVE-2026-25645: insecure temp-file reuse in `extract_zipped_paths()`). Lockfiles refreshed for all transitive and dev dependencies.
- **`pygments`** (transitive, dev/test) — constrained to `>=2.20.0` via `[tool.uv] constraint-dependencies` to avoid **CVE-2026-4539** (ReDoS in `AdlLexer`). Lockfiles updated to `2.20.0`.

### Breaking changes

- **Provider packages** — `nucleusiq-openai` is now **0.6.0** and `nucleusiq-gemini` is **0.2.0**, both requiring **`nucleusiq>=0.7.0`**. Pin or upgrade core and providers together.

### Fixed

- **PyPI wheel packaging** — `nucleusiq.tools.builtin` was omitted from `[tool.setuptools] packages`, so `pip install nucleusiq` produced a broken install (`ModuleNotFoundError: No module named 'nucleusiq.tools.builtin'`). The subpackage is now included in the wheel.

### Added

- **`scripts/verify_core_package_layout.py`** — fails CI if any `core/**/__init__.py` package is missing from `pyproject.toml` (prevents recurrence).
- **CI: wheel smoke test** — after building the core wheel, install only from `dist/*.whl` in a clean venv and import `nucleusiq.tools.builtin` (catches wheel-only failures; editable installs always masked this).

### CI

- **`actions/upload-artifact`** — v6 → v7 (workflow maintenance).

### Dependencies (lockfile refresh)

- `requests` 2.32.5 → **2.33.0** (security)
- `pygments` 2.19.2 → **2.20.0** (security)
- `python-dotenv` 1.2.1 → **1.2.2**
- `pytest-cov` 7.0.0 → **7.1.0**
- `ruff` 0.15.2 → **0.15.8**
- `openai` (SDK) → **2.30.0**

### Packages

| Package | Version | Requires |
|---------|---------|----------|
| `nucleusiq` | **0.7.0** | — |
| `nucleusiq-openai` | **0.6.0** | `nucleusiq>=0.7.0` |
| `nucleusiq-gemini` | **0.2.0** | `nucleusiq>=0.7.0` |

---

## [0.6.0](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.6.0) — 2026-03-28

### Added

- **Google Gemini Provider** (`nucleusiq-gemini` v0.1.0) — second LLM provider proving provider portability:
  - `BaseGemini` implementing the `BaseLLM` contract via `google-genai` SDK (GA)
  - `call()` and `call_stream()` with system/user/assistant messages
  - Multimodal attachment support (images, PDFs, files) via `process_attachments()`
  - Streaming adapters converting Gemini SDK chunks into framework `StreamEvent` objects
  - Thinking/reasoning support via `GeminiThinkingConfig` for Gemini 2.5+ models
  - 4 native tools: Google Search, Code Execution, URL Context, Google Maps
  - Structured output via JSON schema mode (`response_mime_type` + `response_json_schema`)
  - `GeminiLLMParams` with safety settings, thinking config, and candidate count
  - Retry with exponential backoff for rate limits (429), server errors (5xx), connection errors
  - 10 examples covering basic usage, streaming, tools, agent modes, native tools, cost estimation
  - Comprehensive `README.md` with full usage documentation
- `**@tool` Decorator** — create tools from plain functions without subclassing `BaseTool`:
  - `@tool`, `@tool("name")`, `@tool(name="...", description="...")` decorator forms
  - Auto-generates `get_spec()` from function signature + type hints
  - Docstring parsing (first-line, `:param:`, Google-style `Args:`)
  - Supports `str`, `int`, `float`, `bool`, `list`, `dict` parameter types and optional defaults
  - Both sync and async function support
  - Optional `args_schema` for Pydantic model validation
  - Handles `from __future__ import annotations` (string annotation resolution)
- **Cost Estimation** — dollar-cost tracking from token usage:
  - `CostTracker` with `ModelPricing` Pydantic model and `CostBreakdown`
  - Built-in pricing tables for 15 models (OpenAI: gpt-4o, gpt-4.1, o3, o4-mini, etc.; Gemini: 2.5-pro, 2.5-flash, 2.0-flash, etc.)
  - Cost breakdown by purpose (main, planning, tool_loop, critic, refiner) and origin (user vs framework)
  - User-configurable pricing via `tracker.register("my-model", ModelPricing(...))`
  - Prefix-match model lookup (e.g. `gpt-4o-2024-11-20-custom` matches `gpt-4o`)
- **Framework-Level Error Taxonomy** — provider-agnostic exception hierarchy:
  - `NucleusIQError` → `LLMError` base with 9 typed exceptions: `AuthenticationError`, `RateLimitError`, `InvalidRequestError`, `ModelNotFoundError`, `ContentFilterError`, `ProviderServerError`, `ProviderConnectionError`, `PermissionDeniedError`, `ProviderError`
  - Each error carries `provider`, `status_code`, `original_error` attributes
  - `from_provider_error()` factory classmethod for consistent error mapping
  - `BaseLLM.call()` documents the exception contract
  - Both OpenAI and Gemini retry modules map SDK errors to framework types
- **LLM Parameter Standardization** — universal `max_output_tokens` across all providers:
  - `LLMParams.max_output_tokens` replaces `max_tokens` as the canonical parameter
  - Each provider translates internally: OpenAI uses `max_tokens` (older) or `max_completion_tokens` (o-series); Gemini uses `max_output_tokens`
  - O-series model detection (`o1`, `o3`, `o4-mini`) for correct wire format
  - All core framework call sites updated (modes, components, plugins, memory)

### Changed

- Bumped `nucleusiq` to 0.6.0
- Bumped `nucleusiq-openai` to 0.5.0 (requires `nucleusiq>=0.6.0`)
- New package `nucleusiq-gemini` 0.1.0 (requires `nucleusiq>=0.6.0`, `google-genai>=1.0.0`)
- OpenAI `_shared/retry.py` now raises framework-level exceptions (`RateLimitError`, `AuthenticationError`, etc.) instead of raw SDK exceptions
- `BaseLLM.call()` / `call_stream()` signature uses `max_output_tokens` (was `max_tokens`)
- Removed `n` and `stream` from base `LLMParams` (provider-specific concerns)
- `ContextWindowPlugin.max_tokens` clarified as input context window budget (not output tokens)

### Testing

- **2,285 tests passing** (1,795 core + 224 OpenAI + 221 Gemini unit + 45 Gemini integration)
- 266 Gemini tests covering call, stream, tools, native tools, structured output, agent integration, provider portability, retry/error handling
- 38 `@tool` decorator tests (decorator forms, execution, spec generation, docstring parsing, Pydantic schema)
- 35 cost estimation tests (pricing validation, registration, lookup, estimation, display, integration with UsageTracker)
- 12 framework error taxonomy tests (hierarchy, attributes, factory, catchability, provider-agnostic behavior)

---

## [0.5.0](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.5.0) — 2026-03-11

### Added

- **Token Origin Split** — `TokenOrigin` enum (`USER` / `FRAMEWORK`) and `PURPOSE_ORIGIN_MAP` in `UsageTracker`. Every `UsageRecord` now carries an `origin` field. The summary includes a `by_origin` breakdown separating user tokens (the initial MAIN call) from framework overhead (planning, tool loops, critic, refiner). Designed for direct consumption by the future Observability plugin
- `**UsageSummary` Pydantic schema** — `agent.last_usage` now returns a typed `UsageSummary` model (not a raw dict) with `TokenCount`, `BucketStats` sub-models:
  - `usage.summary()` — returns a plain `dict` for JSON serialization / logging / dashboards
  - `usage.display()` — returns a formatted human-readable string (totals, by-purpose, by-origin with % split)
  - Individual attribute access: `usage.total.prompt_tokens`, `usage.by_origin["user"].total_tokens`, etc.
- `**FileWriteTool`** — new built-in tool for writing/appending text files within the workspace sandbox. Features: backup-on-overwrite (`.bak` copy, configurable), max write size limit (default 5 MB), automatic parent directory creation, write/append modes
- `**FileExtractTool` query filtering** — two new parameters:
  - `columns` — comma-separated column names for CSV/TSV filtering (case-insensitive matching)
  - `key_path` — dot-separated key path for JSON/YAML/TOML navigation with array index support (e.g. `"database.host"`, `"items.0.name"`)
- `**FileSearchTool` configurable binary extensions** — `DEFAULT_BINARY_EXTENSIONS` promoted to module-level constant; three new constructor params: `include_extensions` (whitelist mode), `exclude_extensions` (additions to skip set), `binary_extensions` (full override)
- `**DirectoryListTool` max entries** — `max_entries` constructor parameter (default 200) with truncation message to prevent LLM context waste on large directory trees
- `**FileReadTool` encoding auto-detection** — `_detect_encoding()` using `chardet` (optional dependency, first 4 KB sample). Default encoding changed from `"utf-8"` to `"auto"` (auto-detect with UTF-8 fallback)
- **New examples** — `v050_features_example.py` (all 6 features), `usage_tracking_example.py` (OpenAI usage tracking with `summary()` and `display()`)
- `**MockLLM` now returns simulated `usage` data** — enables realistic token tracking in tests and examples without a real LLM

### Changed

- Bumped `nucleusiq` to 0.5.0 (OpenAI provider remains at 0.4.0 — no provider changes)
- `agent.last_usage` return type changed from `dict` to `UsageSummary` (Pydantic model) — use `.summary()` for a plain dict, `.display()` for formatted string
- `StandardMode._tool_call_loop` now tags first LLM call as `MAIN` (user) and subsequent calls after tool results as `TOOL_LOOP` (framework) — matching the streaming path behavior
- `agents/__init__.py` now exports `TokenCount`, `BucketStats`, `UsageSummary`, `TokenOrigin`
- `FileExtractTool` handlers refactored: shared `_format_csv_table()` and `_format_json_value()` renderers (DRY), `ExtractOptions` parameter bag, `_resolve_key_path()` and `_filter_tabular_columns()` helpers
- `tools/builtin/__init__.py` now exports `FileWriteTool`

### Testing

- **1,721 tests passing** (core + all v0.5.0 additions, 4 skipped)
- 59 usage tracker tests (including Pydantic models, display, summary, origin split)
- 50 tool feature tests covering FileWriteTool, query filtering, search config, max entries, encoding detection

---

## [0.4.0](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.4.0) — 2026-03-10

### Added

- **Multimodal Attachments** — 7 `AttachmentType`s (`TEXT`, `PDF`, `IMAGE_URL`, `IMAGE_BASE64`, `FILE_BYTES`, `FILE_BASE64`, `FILE_URL`) with `Attachment` model, `AttachmentProcessor`, and `Task.attachments` support
- **Provider-native file processing** — `BaseLLM.process_attachments()` pluggable contract; OpenAI provider overrides for server-side PDF/XLSX/CSV processing via both Chat Completions and Responses API
- **Provider capability introspection** — `BaseLLM.NATIVE_ATTACHMENT_TYPES`, `SUPPORTED_FILE_EXTENSIONS`, `describe_attachment_support()` with import-time exhaustiveness guards
- **4 Built-in File Tools** — sandboxed to a `workspace_root` directory, inheriting `BaseTool`:
  - `FileReadTool` — read file content with optional `start_line`/`end_line`, large-file truncation, binary detection, and max file size enforcement
  - `FileSearchTool` — text/regex search across files with `max_results` cap
  - `DirectoryListTool` — list directory with glob filtering, recursive option, file sizes
  - `FileExtractTool` — structured extraction for CSV, TSV, JSON, JSONL/NDJSON, YAML, XML, TOML via pluggable `_FORMAT_HANDLERS` registry with `register_extract_format()` for extensibility
- **Workspace sandbox** (`workspace.py`) — `resolve_safe_path()` blocks `../` traversal, symlink escape, and absolute path injection
- `**AttachmentGuardPlugin`** — policy-based attachment validation (allowed/blocked types, max file size, max count, extension filter) via `before_agent` hook
- **File-aware memory** — all 5 memory strategies store attachment metadata alongside messages; user messages get a `[Attached: ...]` summary prefix for context continuity
- `**UsageTracker`** — `UsageRecord`, `CallPurpose` enum (MAIN, PLANNING, TOOL_LOOP, CRITIC, REFINER), wired into all 3 execution modes with `agent.last_usage` and streaming metadata
- **OpenAI API auto-routing** — transparent routing between Chat Completions and Responses API based on tool types, with format conversion and streaming adapters for both
- **Validation hardening** — `AttachmentProcessor.process()` enforces size limits (50 MB), MIME magic-bytes check (warn on mismatch), and large text warning (> 100 KB suggests FileReadTool)
- **File handling guide** — [https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/file-handling/](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/file-handling/) (Attachment vs Tool vs Both decision flowchart)
- **New examples** — `file_attachment_example.py`, `file_tools_example.py`, `attachment_guard_example.py`, OpenAI-native file input examples
- **v0.5.0 gap analysis** — `docs/v0.5.0-gaps.md` consolidating 10 prioritized items from the post-release audit

### Fixed (v0.4.0 audit)

- `**AutonomousMode.run_stream()` missing `store_task_in_memory`** — streaming autonomous mode now stores the user's task in memory before decomposition, matching the non-streaming path
- **Removed dead `_last_metadata` field** from `SummaryMemory` — was stored but never exposed or persisted
- **Removed duplicate `build_attachment_*` helpers** — consolidated module-level and static method versions in `base_mode.py`

### Changed

- Bumped `nucleusiq` to 0.4.0, `nucleusiq-openai` to 0.4.0
- `nucleusiq-openai` now requires `nucleusiq>=0.4.0`
- Memory strategies now accept `metadata` kwarg in `add_message()` for file-aware storage
- `_setup_execution()` delegates user message storage to mode-level `store_task_in_memory()` (avoids double-store)
- `FileExtractTool` now supports 7 formats via `_FORMAT_HANDLERS` registry (was 2)
- `FileReadTool` now detects binary files (null byte check) and enforces configurable max file size (default 10 MB)

### Testing

- **1,649 tests passing** (core + all v0.4.0 additions, 4 skipped)
- 42 new built-in file tools unit tests
- 10 new file tools integration tests (agent with tools in Standard mode tool loop)
- 22 new file-aware memory unit tests
- 15 new AttachmentGuardPlugin unit tests
- 80 attachment unit tests (including validation, exhaustiveness, capability metadata)
- 45 new edge-case tests (symlinks, binary detection, error propagation, all attachment types, multi-turn memory, autonomous streaming memory, format registry)

---

## [0.3.0](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.3.0) — 2026-02-27

### Added

- **End-to-end streaming** via `Agent.execute_stream()` — async generator yielding `StreamEvent` objects with real-time token-by-token output across all 3 execution modes
- `**StreamEvent` + `StreamEventType`** data model (`core/streaming/events.py`) — 8 event types: `TOKEN`, `TOOL_CALL_START`, `TOOL_CALL_END`, `LLM_CALL_START`, `LLM_CALL_END`, `THINKING`, `COMPLETE`, `ERROR`
- `**BaseLLM.call_stream()**` — abstract streaming contract with non-streaming fallback; `MockLLM.call_stream()` for testing
- `**BaseOpenAI.call_stream()**` — OpenAI provider streaming for both Chat Completions and Responses API backends
- `**stream_adapters.py**` — adapter layer converting raw OpenAI SDK chunks/SSE events into framework `StreamEvent` objects
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

## [0.2.0](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.2.0) — 2026-02-25

### Added

- **Configurable tool limits per execution mode**: Direct (5), Standard (30), Autonomous (100) — configurable via `AgentConfig.max_tool_calls`
- **Tool support in DirectMode** — up to 5 tool calls (previously no tools)
- **Critic/Refiner integration in AutonomousMode** — replaces simple LLM review (Layer 3) and generic retry with independent verification and targeted correction
- **Tool limit validation** — agent raises `ValueError` at execution time if more tools are configured than the mode allows
- `**AgentConfig.get_effective_max_tool_calls()`** — centralized method for mode-aware tool limits
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

## [0.1.0](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.1.0) — 2026-02-24

**Initial public release** of the NucleusIQ framework and OpenAI provider.

### Packages


| Package            | Version | PyPI                                                           |
| ------------------ | ------- | -------------------------------------------------------------- |
| `nucleusiq`        | 0.1.0   | [nucleusiq](https://pypi.org/project/nucleusiq/)               |
| `nucleusiq-openai` | 0.1.0   | [nucleusiq-openai](https://pypi.org/project/nucleusiq-openai/) |


### Agent System

- **3 Execution Modes** via Strategy Pattern:
  - `DIRECT` — single LLM call, no tools
  - `STANDARD` — LLM + tool-calling loop
  - `AUTONOMOUS` — orchestration with parallel execution, external validation, structured retry, and progress tracking
- **Autonomous Mode** with `ValidationPipeline` (3-layer validation: tool checks → plugin validators → optional LLM review), `ProgressTracker`, and `Decomposer` for complex task parallelization
- `**ResultValidatorPlugin`** — abstract base class for domain-specific external validation (the framework orchestrates, the LLM executes, external signals validate)
- **ReAct Agent** — Reasoning + Acting pattern implementation
- **Structured Output** — schema-based output parsing and validation
- `**AgentConfig`** — Pydantic configuration with execution mode, retry settings, and sub-agent limits

### Prompt Engineering

- **7 Prompt Techniques**: `ZeroShot`, `FewShot`, `ChainOfThought`, `AutoChainOfThought`, `RetrievalAugmentedGeneration`, `PromptComposer`, `MetaPrompt`
- `**PromptFactory`** — create prompts by technique name via `PromptTechnique` enum

### Tool System

- `**BaseTool**` — LLM-agnostic tool interface with JSON schema generation
- `**BaseTool.from_function()**` — create tools from plain Python functions
- **OpenAI native tools**: `function`, `code_interpreter`, `file_search`, `web_search`, `mcp`, `connector` (via `OpenAITool`)

### Memory System

- **5 Memory Strategies** via `MemoryFactory`:
  - `FullHistoryMemory` — keep all messages
  - `SlidingWindowMemory` — keep last N messages
  - `SummaryMemory` — summarize older messages via LLM
  - `SummaryWindowMemory` — sliding window + summary of dropped messages
  - `TokenBudgetMemory` — keep messages within token budget

### Plugin System

- `**BasePlugin`** ABC with typed request models (`ModelRequest`, `ToolRequest`, `AgentContext`)
- `**PluginManager**` — chain-of-responsibility hook pipeline
- **Decorator API** — `@before_agent`, `@after_agent`, `@before_model`, `@after_model`, `@wrap_model_call`, `@wrap_tool_call`
- **9 Built-in Plugins**:


| Plugin                  | Purpose                                             |
| ----------------------- | --------------------------------------------------- |
| `ModelCallLimitPlugin`  | Limits LLM call count per execution                 |
| `ToolCallLimitPlugin`   | Limits tool call count                              |
| `ToolRetryPlugin`       | Retries failed tools with exponential backoff       |
| `ModelFallbackPlugin`   | Tries fallback models on primary failure            |
| `PIIGuardPlugin`        | Detects/redacts/masks/blocks PII                    |
| `HumanApprovalPlugin`   | Human approval gate with `ApprovalHandler` pattern  |
| `ContextWindowPlugin`   | Trims messages to fit context window                |
| `ToolGuardPlugin`       | Tool whitelist/blacklist                            |
| `ResultValidatorPlugin` | Abstract base for domain-specific result validation |


### LLM Provider — OpenAI (`nucleusiq-openai`)

- **Chat Completions API** — full support with tool calling
- **Responses API** — automatic routing based on tool types
- `**OpenAILLMParams`** — type-safe parameters with typo detection and merge chain (LLM defaults < AgentConfig < per-execute overrides)
- **6 Native Tool Types** — function, code_interpreter, file_search, web_search_preview, mcp, connector
- **Structured Output** — JSON schema enforcement via `response_format`

### Testing

- **1358 tests passing** (1242 core + 116 OpenAI provider, 2 skipped)
- 98% plugin system branch coverage

### Documentation & Examples

- `notebooks/agents/pe_due_diligence.ipynb` — end-to-end autonomous agent demo with 8 PE due diligence scenarios
- 17 core examples + 28 OpenAI provider examples

---

## [Unreleased](https://github.com/nucleusbox/NucleusIQ/compare/v0.7.4...HEAD)

### Planned for v0.7.5+

- Native + Custom tool mixing (ToolStrategy Protocol — Gemini native/custom split)
- Full observability wiring (PluginEvent, MemorySnapshot, AutonomousDetail, prompt tracing)
- Agent Types: ReAct integration into mode system, Chain-of-Thought as config flag
- Context Window Management (budget tracker, tool result compression)
- New LLM Providers: Anthropic, Ollama
- Gemini advanced features: Batch API, Deep Research Agent, File Search
- CostTracker Agent integration (`agent.last_cost`)
- See `docs/BACKLOG.md` for full list

