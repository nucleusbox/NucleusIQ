# Changelog

All notable changes to NucleusIQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.7.9](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.9) — 2026-05-07

### Added

#### LLM rate limiting (core)

- **`nucleusiq.llms.retry_policy`** — shared HTTP **429** handling: parse **`Retry-After`** (integer delay or HTTP-date), merge with **capped exponential backoff**, and apply **`DEFAULT_RATE_LIMIT_MAX_SLEEP_SECONDS`** (120s) as a single-sleep ceiling. Helpers: **`extract_retry_after_header`**, **`parse_retry_after_seconds`**, **`compute_rate_limit_sleep`**, typed metadata **`RateLimitRetryMeta`** for structured logs.
- **Exports** — `retry_policy` symbols re-exported from **`nucleusiq.llms`** (`__all__` updated).

#### Groq provider

- **`nucleusiq_groq.capabilities`** — **`PARALLEL_TOOL_CALLS_DOCUMENTED_MODELS`**, **`check_parallel_tool_calls_capability`** (warn on unknown models; strict path raises **`InvalidRequestError`**).
- **`GroqLLMParams.strict_model_capabilities`** — opt-in validation before send; excluded from **`to_call_kwargs()`** so it never hits the wire.
- **Docs** — `docs/design/GROQ_PROVIDER.md`: §**7.1** 429 flow, §**7.2** cross-provider retries, §**8.1** hosted vs local tools, §**9.1** structured-output models, beta checklist.
- **`nucleusiq_groq._shared.stream_create`** — `open_streaming_completion` / `apply_stream_options`; streaming **open** uses the same **429** / **`Retry-After`** policy as non-stream chat.
- **Tests** — `test_capabilities.py`, `test_retry.py` (**Retry-After**), `test_base_groq.py` / `test_llm_params` for strict + kwargs; **`test_tools_hosted_ids.py`** (hosted tool ID constants); **`tests/integration/test_groq_live_smoke.py`** (`pytest -m integration`, **`GROQ_API_KEY`**); default pytest **`not integration`** in Groq package **`addopts`**.

#### Tests (framework)

- **`tests/llms/unit/test_retry_policy.py`** — parsing and sleep math.
- **`tests/llms/unit/test_retry_policy_provider_contract.py`** — stable **`policy`** dict keys for observability pipelines.

#### Tests (providers)

- **OpenAI** — `tests/test_openai_rate_limit_retry_after.py`; **`tests/test_openai_retry_status_mapping.py`** (**404** / **409** no spurious retry).
- **Gemini** — `tests/unit/test_retry.py` — **429** respects **`Retry-After`** on **`ClientError.response`**.

### Changed

- **`nucleusiq.core.__version__`** — aligned with **`pyproject.toml`** (**0.7.9**).
- **`nucleusiq_groq._shared.retry`** — **429** uses **`compute_rate_limit_sleep`** + structured **`WARNING`** logs (`sleep=`, `policy=`).
- **`nucleusiq_openai._shared.retry`** — same **429** policy; explicit **`openai.NotFoundError`** → **`ModelNotFoundError`** and **`openai.ConflictError`** → **`InvalidRequestError`** (409) **before** generic **`APIError`** so **404**/**409** are not retried as server errors.
- **`nucleusiq_gemini._shared.retry`** — **429** uses **`retry_policy`** when **`response`** carries **`Retry-After`**; dependency floor **`nucleusiq>=0.7.9`**.
- **`nucleusiq-groq`** — README and **`pyproject.toml`** require **`nucleusiq>=0.7.9`**; **`0.1.0b1`** public **beta** (`Development Status :: 4 - Beta`): stream **open** uses **`stream_create.open_streaming_completion`** (same **429** / **`Retry-After`** policy as non-stream); **`pytest -m integration`** live smoke (`GROQ_API_KEY`); **`GROQ_COMPOUND_HOSTED_TOOL_IDS`** / **`GROQ_GPT_OSS_HOSTED_TOOL_IDS`** in **`nucleusiq_groq.tools`** (Groq-docs mirror; not wired).
- **`nucleusiq-openai`** / **`nucleusiq-gemini`** — dependency floor **`nucleusiq>=0.7.9`**.

### Packages

| Package            | Version        | Note |
| ------------------ | -------------- | ---- |
| `nucleusiq`        | **0.7.9**      | `retry_policy`; aligned `__version__` |
| `nucleusiq-openai` | **0.6.4**      | 429 **`Retry-After`**; 404/409 mapping; `nucleusiq>=0.7.9` |
| `nucleusiq-gemini` | **0.2.6**      | 429 **`retry_policy`**; `nucleusiq>=0.7.9` |
| `nucleusiq-groq`   | **0.1.0b1** β  | Public beta; stream-open retry; integration marker hosted-tool ID constants; `nucleusiq>=0.7.9` |

### Validation

- Run **`pytest`** in **`src/nucleusiq`** (LLM unit tests), **`src/providers/inference/groq`**, **`src/providers/llms/openai`**, and **`src/providers/llms/gemini`** before publishing wheels.

---

## [0.7.8](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.8) — 2026-05-06

### Added

#### Groq inference provider (standalone package)

- **`nucleusiq-groq` (`0.1.0a1`, alpha)** — new installable provider under `src/providers/inference/groq/`: `BaseGroq`, `GroqLLMParams`, Chat Completions + streaming, local tools, structured output, retries. Install: `pip install nucleusiq-groq` (with `nucleusiq>=0.7.8`). Docs: `src/providers/inference/groq/README.md`, `docs/design/GROQ_PROVIDER.md`.

#### Run-local context state (L4 / analyst state)

- **`InMemoryWorkspace`** (`nucleusiq.agents.context.workspace`) — bounded per-run notebook: notes, artifacts, summaries with `WorkspaceEntry`, `WorkspaceStats`, and `WorkspaceLimitError` for cap violations.
- **Workspace tools** (`nucleusiq.agents.context.workspace_tools`) — factory **`build_workspace_tools`**: `write_workspace_note`, `write_workspace_artifact`, `list_workspace_entries`, `read_workspace_entry`, `summarize_workspace`. Helpers **`is_workspace_tool_name`** and **`is_context_management_tool_name`** (framework-injected context tools that should not consume the user’s external tool budget).
- **`InMemoryEvidenceDossier`** + **`EvidenceItem`** (`nucleusiq.agents.context.evidence`) — structured claims with status, confidence, tags, quotes, and provenance (`source` ref / locator / tool metadata).
- **Evidence tools** (`nucleusiq.agents.context.evidence_tools`) — **`build_evidence_tools`**: `add_evidence`, `add_evidence_gap`, `list_evidence`, `summarize_evidence`, `evidence_coverage` plus **`is_evidence_tool_name`**.

#### L5 document corpus (lexical, in-memory)

- **`InMemoryDocumentCorpus`** (`nucleusiq.agents.context.document_search`) — indexes caller-provided text into bounded chunks with **`DocumentRef`**, **`DocumentChunk`**, **`ChunkHit`**, and lexical search (no PDF parsing, embeddings, or Task E logic). Exposes **`DocumentSearchStats`** (`documents_indexed`, `chunks_indexed`, search/retrieval counts, chars returned, promotions to evidence).
- **Document corpus tools** (`nucleusiq.agents.context.document_corpus_tools`) — **`build_document_corpus_tools(corpus, evidence=…)`**: `search_document_corpus`, `get_document_chunk`, `list_indexed_documents`, **`promote_document_chunk_to_evidence`** (optional dossier link). **`is_document_corpus_tool_name`** for injection/budget rules.

#### L4.5 automatic activation

- **`ContextStateActivator`** (`nucleusiq.agents.context.state_activator`) — after each **business** tool result (skips framework context tools), applies:
  - **Strict heuristics** to promote evidence-shaped facts into the dossier and optional **gap** items.
  - **Light ingest** — durable workspace notes + L5 corpus indexing for substantive read/search/file-style outputs (tool-name hints, length gates, acronym/false-positive guards for short tokens like `"ai"`).
- **`ContextActivationMetrics`** — cumulative counters (tool results seen/activated, workspace/evidence promotions, light ingests, skips, synthesis-package / critic-package flags, etc.) surfaced via **`AgentResult.metadata["context_activation"]`**.

#### L6 phase telemetry and evidence gate (framework-visible)

- **`PhaseController`** + **`AgentPhase`** literals (`nucleusiq.agents.context.phase_control`) — records ordered phase transitions, durations, evidence-gate outcomes, and flags such as **`synthesis_used_package`**, **`critic_used_package`**, **`refiner_used_gaps`**. Snapshot **`PhaseStats.to_dict()`** exposed as **`AgentResult.metadata["phase_control"]`**.
- **`EvidenceGate`** + **`EvidenceGateDecision`** (`nucleusiq.agents.context.phase_control`) — optional tag-based completeness check against the dossier (`passed`, `blocked`, missing/gap tags).
- **`AgentConfig`** (`nucleusiq.agents.config.agent_config`) — new fields: **`evidence_gate_required_tags`**, **`evidence_gate_enforce`**, **`context_tool_result_corpus_max_chars`** (per-result cap for auto-indexing into L5; `0` disables), **`context_activation_ingest_min_chars`** (minimum text size for light ingest when not evidence-shaped).

#### Synthesis package

- **`SynthesisPackage`** + **`build_synthesis_package`** (`nucleusiq.agents.context.synthesis_package`) — deterministic, bounded final-answer input from workspace + evidence (supported claims, gaps, source index, recalled snippets) with omission metadata. **`Agent.build_synthesis_package`** and **`_last_synthesis_package`**; package metadata may appear as **`AgentResult.metadata["synthesis_package"]`**.

#### Agent wiring and observability

- **`Agent`** (`nucleusiq.agents.agent`) — lazy accessors: **`workspace`**, **`evidence_dossier`**, **`document_corpus`**, **`phase_controller`**, **`evidence_gate`**, **`build_synthesis_package`**. Autonomous initialization can provision workspace, dossier, corpus, phase controller, gate, and activator together.
- **`AgentResult.metadata`** extensions when state exists: **`workspace`**, **`evidence`**, **`document_search`** (corpus stats), **`phase_control`**, **`context_activation`**, **`synthesis_package`** (alongside existing **`context_telemetry`** on the result object).

#### Shared tool-result serialization

- **`tool_result_to_context_string`** (`nucleusiq.agents.modes.tool_payload`) — single implementation used by **`BaseExecutionMode`**, **`StandardMode`**, and **`DirectMode`** for appending tool outputs to context (strings pass through; other values JSON-serialized with safe fallback).

#### Tests

- Broad unit and integration coverage for document search/corpus tools, workspace/evidence tools, synthesis package, phase control, state activator, tool payload, `AgentResult`/metadata wiring, and context integration (see `tests/unit/context/` and `tests/agents/unit/`).

### Changed

- **`scripts/verify_core_package_layout.py`** — now checks **Hatch** wheels for OpenAI, Gemini, and **Groq** providers: `packages = ["…"]` matches the on-disk import root, and every package directory that contains `.py` modules includes **`__init__.py`** (classic layout / import–wheel alignment). Still validates **`nucleusiq`** setuptools `packages` vs `core/` on disk.
- **`nucleusiq_groq/_shared/__init__.py`** — added so `_shared` matches the same layout rules as other providers.
- **`CriticRunner`** (`nucleusiq.agents.modes.autonomous.critic_runner`): on **any** exception during critique (infra/LLM/parser), returns **`Verdict.UNCERTAIN`** with score **`0.0`** and explicit feedback instead of treating failures as a synthetic pass — safer default for autonomous orchestration.
- **Autonomous simple path** (`nucleusiq.agents.modes.autonomous.simple_runner`): after a successful **Refiner** pass, **`agent._last_messages`** is refreshed so a subsequent **Critic** sees the revised trace.
- **`ContextStateActivator`** (`nucleusiq.agents.context.state_activator`): stricter topic/heuristic gates (e.g. reduced false positives from bare **`"ai"`** substring matches) and tuned ingest/evidence promotion behavior.
- **Context compaction / masking** (`nucleusiq.agents.context.compactor`, `nucleusiq.agents.context.strategies.conversation`, `nucleusiq.agents.context.strategies.emergency`, `nucleusiq.agents.context.strategies.observation_masker`) — updates to stay consistent with the state stack and new regression coverage (no user-facing API breaks intended).

### Fixed

- Tool results that are **already `str`** are no longer **double-encoded** (e.g. JSON-wrapped twice) when merged into the visible context in **standard**, **direct**, and **base** modes (**`tool_result_to_context_string`** behavior).

### Provider updates

- **`nucleusiq-groq` (0.1.0a1)** — new **alpha** Groq provider: Chat Completions via the official **`groq`** SDK; local function calling, structured output (`json_schema` where the model supports it), streaming, retries, and wire compatibility fixes for Groq’s OpenAI-shaped API. Requires **`nucleusiq>=0.7.8`**. Developer README: `src/providers/inference/groq/README.md`; design tracker: `docs/design/GROQ_PROVIDER.md`.
- **`nucleusiq-openai` (0.6.3)** — dependency floor raised to **`nucleusiq>=0.7.8`** (package version unchanged).
- **`nucleusiq-gemini` (0.2.5)** — same **`nucleusiq>=0.7.8`** floor.

### Validation

- **2554 passed**, **2 skipped** with `pytest tests --ignore=tests/memory/integration` (local gate, ~2026-05-06).
- **`tests/memory/integration`** may still report failures when the configured OpenAI project cannot call **`gpt-4o-mini`** (`403 model_not_found`); treat as **provider access**, not a core regression.

### Packages

| Package            | Version       | Note                                                                 |
| ------------------ | ------------- | -------------------------------------------------------------------- |
| `nucleusiq`        | **0.7.8**     | Context state stack + L5 corpus tools + Critic/tool-payload fixes     |
| `nucleusiq-openai` | **0.6.3**     | `nucleusiq>=0.7.8`                                                    |
| `nucleusiq-gemini` | **0.2.5**     | `nucleusiq>=0.7.8`                                                    |
| `nucleusiq-groq`   | **0.1.0a1** α | Groq Chat Completions provider; `nucleusiq>=0.7.8` (pre-release)       |

---

### CI

- **Groq provider** — `test-groq` (pytest + coverage), **uv** sync step, **Pyrefly** for `src/providers/inference/groq`, import check + **build** matrix entry for `nucleusiq-groq`, **pip-audit** editable include.
- **`import-check`** — `verify_core_package_layout.py` validates **core setuptools** and **Hatch** provider roots / `__init__.py` layout before package installs.

---

## [0.7.7](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.7) — 2026-04-27

### Added

- Context Management v2 recall and masking tests covering policy classification, squeeze gating, marker parsing, synthesis rehydration, recall round-trips, and streaming/non-streaming masking symmetry.
- Autonomous-mode stability tests covering Critic/Refiner routing, abstention, compute-budget escalation, and tool-budget synthesis behavior.
- Tool deduplication metadata via opt-in `BaseTool.idempotent` / `@tool(idempotent=True)`. The default remains `False` so live-data tools are never deduplicated unless explicitly declared safe.

### Changed

- Stabilized Context Management v2 around one `Compactor`, bounded observation markers, recallable offload, and synthesis-time rehydration.
- Kept `ContextConfig.squeeze_threshold=0.70` as the stable default after validating that later masking can miss hard context limits on PDF-heavy runs.
- Bounded marker previews to a small orientation preview only for large tool results, preventing marker bloat from becoming a second evidence store.
- Restored Task E to the normal benchmark surface (`list_pdf_inventory` + `read_annual_report_excerpt`) and quarantined evidence-pack/search helpers as experiment-only artifacts, not framework behavior.

### Fixed

- Synthesis rehydration now uses the engine's resolved model context window when `ContextConfig.max_context_tokens` is auto-detected, instead of silently skipping rehydration when the config field is `None`.
- `AgentResult` now surfaces legacy mode error-string sentinels as `status=error` when the agent state is already `ERROR`, including a stable `ToolCallLimitError` classification for exhausted tool budgets.
- Standard mode forces a tools-free synthesis pass when the configured tool-call cap is reached and synthesis is enabled, so autonomous runs can still reach validation/refinement instead of stopping at a raw tool-limit string.

### Provider Updates

- **OpenAI provider (`nucleusiq-openai` 0.6.3)** — keeps provider behavior in sync with the stable V2 core changes, including Responses/Chat tool conversion and response-normalization fixes covered by provider tests.
- **Gemini provider (`nucleusiq-gemini` 0.2.5)** — keeps Gemini response normalization and base-call behavior aligned with the updated core message/tool contracts.

### Validation

- Focused stable gates: `1340 passed` across `src/nucleusiq/tests/unit/context`, `src/nucleusiq/tests/agents/unit`, `test_context_management_e2e.py`, and `test_recall_round_trip_e2e.py`.
- Full local gate excluding provider-backed memory integration: `2469 passed, 2 skipped`.
- Full local gate including provider-backed memory integration: `2469 passed, 2 skipped, 8 failed` because the configured OpenAI project returned `403 model_not_found` for `gpt-4o-mini`. These are external/provider access failures, not release-blocking framework regressions.

### Known Limitations

- This release does not claim that long evidence-heavy tasks are fully solved.
- Task E is not production-grade on `gpt-5.2`; it remains an internal research benchmark.
- Workspace state, document search, evidence dossier, and autonomous phase control are deferred to the next feature line.

### Packages

| Package            | Version   | Note                                                                 |
| ------------------ | --------- | -------------------------------------------------------------------- |
| `nucleusiq`        | **0.7.7** | Stable V2 context/autonomous framework release                       |
| `nucleusiq-openai` | **0.6.3** | Provider sync for stable V2 message/tool contracts (`nucleusiq>=0.7.7`) |
| `nucleusiq-gemini` | **0.2.5** | Provider sync for stable V2 message/tool contracts (`nucleusiq>=0.7.7`) |

## [0.7.6](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.6) — 2026-04-10

### Breaking Changes

- `**prompt` is now mandatory on `Agent**` — `BaseAgent.prompt` changed from `Optional[BasePrompt]` (default `None`) to a required `BasePrompt` field. All agents must now be created with an explicit prompt object (e.g. `ZeroShotPrompt().configure(system="...")`). This eliminates the ambiguity between the prompt system and the role/objective/narrative fallback system.
- `**narrative` field removed from `BaseAgent**` — the `narrative` parameter no longer exists. Any content previously placed in `narrative` should be moved to `prompt.system` (the system message for the LLM).
- `**role` and `objective` are now labels only** — they are used for logging, sub-agent naming, and documentation. They are **not** sent to the LLM. `role` defaults to `"Agent"` and `objective` defaults to `""`. Neither is required.
- `**MessageBuilder.build()` no longer accepts `role`, `objective`, or `narrative`** — it only accepts `prompt` for system message construction.
- `**ZeroShotPrompt.input_variables**` — `"user"` moved from mandatory `input_variables` to `optional_variables`. Only `"system"` is now required. The task objective provides the user query; `prompt.user` is an optional preamble.

### Migration Guide

```python
# Before (v0.7.5)
agent = Agent(
    name="MyBot",
    role="Analyst",
    objective="Analyze data",
    narrative="You are a data analyst. Provide detailed analysis.",
    llm=llm,
)

# After (v0.7.6)
from nucleusiq.prompts.zero_shot import ZeroShotPrompt

agent = Agent(
    name="MyBot",
    role="Analyst",              # label only — NOT sent to LLM
    objective="Analyze data",    # label only — NOT sent to LLM
    prompt=ZeroShotPrompt().configure(
        system="You are a data analyst. Provide detailed analysis.",
    ),
    llm=llm,
)
```

### Added — Context Window Management (13 new files)

- **Context Window Management (Phase 1 + Phase 2)** — OS-inspired context hygiene system for quality, cost, and latency. Phase 1 provides overflow prevention; Phase 2 provides proactive context rot prevention:
  - `ContextConfig` — immutable configuration with `optimal_budget` (default 50K tokens, quality-optimized), `enable_observation_masking` (default True), `cost_per_million_input` for dollar-savings telemetry, mode-aware defaults (`ContextConfig.for_mode("autonomous")`)
  - `ContextEngine` — facade with 4-method API: `prepare()` before LLM calls, `post_response()` after LLM response (Tier 0 masking), `ingest_tool_result()` after tool execution, `checkpoint()` at task boundaries
  - `ObservationMasker` (Tier 0) — automatically strips consumed tool results after each LLM response, replacing with slim markers. Full content preserved in `ContentStore`. Research shows this alone solves ~80% of context rot
  - `ContextLedger` + `Region` enum — O(1) per-message token tracking grouped by region (`SYSTEM`, `MEMORY`, `USER`, `ASSISTANT`, `TOOL_RESULT`, `RESERVED`)
  - `ContextBudget` — frozen value-object snapshots with `utilization`, `available`, `can_fit()` properties
  - `TokenCounter` protocol + `DefaultTokenCounter` — provider-agnostic token estimation; providers inject precise implementations
  - `ContentStore` + `ContentRef` — offloaded artifact storage with preview + rehydration markers
  - `CompactionPipeline` — 4-tier progressive strategy chain (quality-focused thresholds against `optimal_budget`):
    - `ObservationMasker` (Tier 0, always) — strip consumed tool results post-response
    - `ToolResultCompactor` (@ 60%) — truncate/offload oversized tool results (Minor GC)
    - `ConversationCompactor` (@ 75%) — remove old turns with optional structured working state summary (Major GC)
    - `EmergencyCompactor` (@ 90%) — last-resort, keeps system + recent only (Full GC)
  - `ContextTelemetry` — peak/final utilization, per-compaction events, region breakdown, offloaded artifacts, observations masked, tokens masked, optimal budget, estimated cost savings — all exposed in `AgentResult.context_telemetry`
  - `SummarySchema` — contract for structured summarization
  - `ConversationCompactor` enhanced with optional structured working state summary (goals, decisions, tool findings) when `enable_summarization=True`
  - `BaseLLM.get_context_window()` — base method (default 128K) for ContextEngine auto-detection, with provider overrides

### Added — Prompt System Refactor

- **Mandatory `prompt` on Agent** — all agents must now be created with an explicit `BasePrompt` instance. This is the single source of truth for what the LLM sees.
- `**role` and `objective` clarified as labels** — updated field descriptions, docstrings, and `ExecutionContext` protocol to make clear these are for logging/documentation only.
- `**ZeroShotPrompt` relaxed** — `user` moved to `optional_variables`; only `system` is required. The task objective serves as the user query.
- **Shared test helper `make_test_prompt()`** — added to `nucleusiq.tests.conftest` for consistent test agent creation.

### Added — Synthesis Pass

- **Synthesis pass in Standard + Streaming modes** — after multiple rounds of tool calls, the agent makes one final LLM call **without tools** and with an explicit "write the full deliverable" nudge. Breaks the "mode inertia" pattern where the model stays in tool-calling behaviour and returns a terse summary instead of the full output.
- `**AgentConfig.enable_synthesis`** — `bool = True`. Controls whether the synthesis pass fires after multi-round tool loops.
- `**CallPurpose.SYNTHESIS**` — new enum value for usage tracking and observability.

### Added — Observability & Tracing

- **Prompt strategy tracing** — `LLMCallRecord.prompt_technique` now populated automatically from the agent's prompt object in both streaming and non-streaming paths via `_extract_prompt_technique()`.
- `**ObservabilityConfig` consolidation** — unified config with `tracing`, `verbose`, `log_level`, `log_llm_calls`, `log_tool_results`. `AgentConfig.observability` takes precedence over legacy `verbose` + `enable_tracing` fields.
- `**AgentConfig.effective_tracing` / `effective_verbose`** — properties resolving the observability-vs-legacy precedence.
- **Sub-agent metric rollup** — `AutonomousMode._rollup_sub_agent_metrics()` merges sub-agent LLM calls, tool calls, and context telemetry into the parent agent's tracer and `AgentResult`.

### Added — Provider: OpenAI (`nucleusiq-openai` 0.6.2)

- **Context window registry** — new `_CONTEXT_WINDOWS` dict in `_shared/model_config.py` with 20 models (GPT-4.1, GPT-4.1-mini/nano, GPT-5, GPT-5-mini/nano, GPT-5.4, o1/o3/o4-mini, GPT-4o, GPT-3.5-turbo, etc.). Supports exact name and prefix-matching for versioned model names (e.g. `gpt-4o-2024-08-06` matches `gpt-4o`). Default fallback: 128K.
- `**get_context_window()`** — new function in `_shared/model_config.py` and new method override on `BaseOpenAI`, enabling `ContextEngine` to auto-detect the model's context budget.
- **Prompt API migration** — all 20 example files updated to use `prompt=ZeroShotPrompt().configure(system=...)` (removed `narrative=`, `instructions=`). 2 test files updated (`test_openai_file_input.py`, `test_openai_file_input_integration.py`).

### Added — Provider: Gemini (`nucleusiq-gemini` 0.2.4)

- `**get_context_window()` method on `BaseGemini`** — new method delegating to the existing `_MODEL_REGISTRY` (7 models: gemini-2.5-pro/flash/flash-lite, 2.0-flash/flash-lite, 1.5-pro/flash, each with `context_window` in `GeminiModelInfo`). The registry and `get_context_window()` function in `_shared/model_config.py` already existed; this release wires the method onto `BaseGemini` so `ContextEngine` can call it via the `BaseLLM.get_context_window()` protocol.
- **Prompt API migration** — all 5 example files updated to use `prompt=ZeroShotPrompt().configure(system=...)` (removed `narrative=`, `instructions=`, `model=` from config). 3 integration test files updated (`test_gemini_agent.py`, `test_mixed_tools.py`, `test_provider_portability.py`).

### Added — Tests

- **97 new context management unit tests** — `test_budget.py` (21), `test_counter.py` (7), `test_store.py` (9), `test_strategies.py` (11), `test_engine.py` (11), `test_config.py` (13), `test_observation_masker.py` (11), `test_engine_phase2.py` (14)
- **4 new agent-level context integration tests** — `test_context_coverage.py`, `test_context_engine_wiring.py`, `test_context_real_world_proof.py`, `test_synthesis_and_masking.py`
- **27 existing test files updated** for the prompt system refactor (removed `narrative=`, added `prompt=`, removed `role=`/`objective=` from `MessageBuilder.build()` calls).

### Added — Notebooks & Scripts

- `**context_management_tcs_deep_dive.ipynb`** — full TCS research analyst demo with context management across all 3 modes (baseline, standard-managed, autonomous-managed). Uses `timeout=300.0` for long synthesis generations.
- `**context_window_management_showcase.ipynb**` — context management capability showcase.
- `**scripts/demo_context_management.py**` — standalone context management demo script.
- `**gemini_mixed_tools_showcase.ipynb**` — updated to new prompt API.
- `**research_analyst_tcs.ipynb**` — updated to new prompt API.

### Changed

- `AgentConfig` gains `context: ContextConfig | None`, `observability: ObservabilityConfig | None`, and `enable_synthesis: bool` fields
- `Agent._setup_execution()` creates `ContextEngine` per execution (when configured)
- `Agent._build_result()` captures `ContextTelemetry` and merges sub-agent telemetries into `AgentResult.context_telemetry`
- `BaseExecutionMode.call_llm()` calls `engine.prepare()` before LLM, `engine.post_response()` after LLM response
- `StandardMode._process_tool_calls()`, `DirectMode._handle_tool_calls()`, and `_streaming_tool_call_loop()` call `engine.ingest_tool_result()` after every tool execution
- `MessageBuilder.build()` simplified — always uses `prompt.system`/`prompt.user`; removed `role`/`objective`/`narrative` fallback
- `BaseExecutionMode.build_messages()` no longer passes `role`/`objective`/`narrative` to `MessageBuilder`
- `Decomposer._create_sub_agent()` now inherits `prompt` from parent agent and collects sub-agent results for telemetry rollup
- `BaseAgent` docstring rewritten to clearly separate labels (`role`, `objective`) from LLM instructions (`prompt`)
- `ReActAgent` docstring updated to reflect new prompt API
- `ExecutionContext` protocol updated — `prompt` is now `BasePrompt` (non-optional)
- `AgentResult.display()` enhanced with context telemetry section (compactions, regions, offloaded artifacts)
- All 3 execution modes use `config.llm_max_output_tokens` instead of hardcoded values
- Default compaction thresholds tuned for quality focus: 60%/75%/90%
- `build_llm_call_record()` and `build_llm_call_record_from_stream()` accept `prompt_technique` parameter

### Files Changed (92 total)


| Category                           | Modified | New | Total |
| ---------------------------------- | -------- | --- | ----- |
| Core runtime                       | 17       | 14  | 31    |
| Core tests                         | 27       | 13  | 40    |
| Core examples                      | 6        | 0   | 6     |
| OpenAI provider (runtime)          | 2        | 0   | 2     |
| OpenAI provider (tests + examples) | 22       | 0   | 22    |
| Gemini provider (runtime)          | 1        | 0   | 1     |
| Gemini provider (tests + examples) | 8        | 0   | 8     |
| Notebooks + scripts                | 2        | 3   | 5     |
| Docs + changelog                   | 1        | 0   | 1     |


### Packages


| Package            | Version   | Requires           | Note                                                                                   |
| ------------------ | --------- | ------------------ | -------------------------------------------------------------------------------------- |
| `nucleusiq`        | **0.7.6** | —                  | Context window management, prompt system refactor, synthesis pass, ObservabilityConfig |
| `nucleusiq-openai` | **0.6.2** | `nucleusiq>=0.7.6` | Context window registry (30 models), prompt API migration                              |
| `nucleusiq-gemini` | **0.2.4** | `nucleusiq>=0.7.6` | `get_context_window()` override, prompt API migration                                  |


---

## [0.7.5](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.5) — 2026-04-03

### Added

- **Native + Custom tool mixing (Gemini proxy pattern)** — Gemini's `generateContent` API rejects requests that combine native tools (`google_search`, `code_execution`, `url_context`, `google_maps`) with custom function declarations. The proxy pattern transparently resolves this:
  - `tool_splitter.py` — `has_mixed_tools()`, `classify_tools()`, `build_proxy_spec()` with `PROXY_DESCRIPTIONS` for all 4 native tool types
  - `_GeminiNativeTool` enhanced with `_enable_proxy_mode()` / `_disable_proxy_mode()` lifecycle — in proxy mode, native tools appear as function declarations to the LLM and execute via a `generate_content` sub-call
  - `BaseGemini.convert_tool_specs()` override detects mixed tools and activates proxy mode automatically
  - Defense-in-depth warning in `build_tools_payload()` if mixed declarations are detected
  - **Zero core framework changes** — works across `DirectMode`, `StandardMode`, `AutonomousMode` via existing `BaseLLM.convert_tool_specs()` hook
- **Full observability wiring** — all `AgentResult` trace fields now populated when `enable_tracing=True`:
  - `**PluginEvent` audit trail** — `PluginManager` records timing and metadata for all 6 plugin hooks (`before_agent`, `after_agent`, `before_model`, `after_model`, `execute_model_call`, `execute_tool_call`) via tracer
  - `**AutonomousDetail`** — autonomous mode populates `attempts`, `max_attempts`, `complexity` ("simple" / "complex"), `sub_tasks` (for complex decomposition), `refined`, and `validations` (tuple of `ValidationRecord`)
  - `**MemorySnapshot**` — `Agent._build_result()` captures memory strategy name, message count, token count, and last 10 messages (truncated to 200 chars) when a tracer and memory are present
  - **Decomposer tracing** — `Decomposer.analyze()` now records its direct `agent.llm.call()` on the tracer manually, closing the bypass gap noted in v0.7.4
  - `**AgentResult.display()` enhanced** — rich human-readable output now includes plugin events, memory snapshot summary, and autonomous execution details
  - `**LLMCallRecord.prompt_technique`** — optional field added for future prompt strategy tracing
- **59 Gemini proxy pattern tests** — `test_tool_splitter.py`, `test_native_tool_proxy.py`, `test_convert_tool_specs_mixed.py` (unit) + `test_mixed_tools.py` (integration)
- **17 observability tests** — `test_plugin_event_tracing.py` (7), `test_autonomous_detail_tracing.py` (5), `test_memory_snapshot.py` (5)

### Changed

- `PluginManager` now accepts an `ExecutionTracerProtocol` via a `tracer` property — all 6 hook methods instrumented with `time.perf_counter()` timing
- `AutonomousMode._run_simple()` and `_run_complex()` now record `ValidationRecord` and `AutonomousDetail` on the tracer at each validation checkpoint
- `Decomposer.analyze()` records LLM call directly on tracer (previously bypassed the `call_llm` path)
- `Agent._build_result()` captures `MemorySnapshot` from agent memory (when tracer and memory are both present) before assembling `AgentResult`

### Packages


| Package            | Version   | Note                                                                        |
| ------------------ | --------- | --------------------------------------------------------------------------- |
| `nucleusiq`        | **0.7.5** | Full observability wiring, proxy pattern support in core result model       |
| `nucleusiq-openai` | 0.6.1     | No change                                                                   |
| `nucleusiq-gemini` | **0.2.3** | Native + Custom tool mixing via proxy pattern (requires `nucleusiq>=0.7.4`) |


---

## [0.7.4](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.4) — 2026-04-03

### Added

- `**observability` package** — dedicated `nucleusiq.agents.observability` package with SRP file layout:
  - `protocol.py` — `ExecutionTracerProtocol` (`@runtime_checkable`)
  - `default_tracer.py` — `DefaultExecutionTracer` (in-memory, `__slots__` optimised)
  - `noop_tracer.py` — `NoOpTracer` (Null Object, zero overhead)
  - `record_builders.py` — `build_tool_call_record`, `build_llm_call_record`, `build_llm_call_record_from_stream`
  - `_response_parser.py` — `extract_tool_calls` (OpenAI + Gemini), `safe_int`, `usage_dict_from_response`
- **Agent wiring** — each `execute()` / `execute_stream()` run resets a fresh `DefaultExecutionTracer` on the agent; `BaseExecutionMode.call_llm` / `call_tool` / `_streaming_tool_call_loop` record timings and outcomes; `AgentResult` now receives `tool_calls`, `llm_calls`, `plugin_events`, `memory_snapshot`, `autonomous`, and `warnings` from the tracer where populated.
- **All 3 execution modes traced** — Direct, Standard, and Autonomous (including Critic LLM calls with `purpose="critic"`). Known gap: `Decomposer.analyze()` bypasses the tracer (will be wired in v0.7.6).
- **Tests** — `tests/unit/test_execution_tracer.py`, `tests/agents/unit/test_agent_tracer_integration.py`, `tests/agents/unit/test_autonomous_tracer_integration.py`.
- `**integration_test/run_integration.py`** — version check for `0.7.4` plus tracer smoke assertions.
- `**AgentConfig.enable_tracing**` — `bool = False`. When off (default), `AgentResult` trace fields are empty tuples with zero overhead. When on, `DefaultExecutionTracer` captures all LLM/tool calls, durations, and warnings.
- **Pyrefly static type checking (Meta, MIT)** — integrated across all packages as Python's "compile-time" safety layer:
  - `[tool.pyrefly]` config in `pyproject.toml` for core, OpenAI, and Gemini.
  - All **121 type errors** fixed across core (71), OpenAI (33), Gemini (17) — null safety guards, undeclared dynamic attributes, type annotation corrections, override signature fixes.
  - Declared `_last_messages` and `_execution_progress` as `PrivateAttr` on `Agent`.
  - Fixed `_plugin_manager` type to `PluginManager | None`.
  - Fixed `build_call_kwargs` return type, `responses_api` tuple types, `ChatCompletionsPayload` construction.
  - Pyrefly added to `dependency-groups.lint` (dev-only, not shipped in wheel).
  - CI `type-check` job gates the build: installs with `editable_mode=strict` then runs `pyrefly check` for all 3 packages.
- `**core/errors/` package** — converted from single `errors.py` to a proper package: `base.py` defines `NucleusIQError` (cycle-free), `__init__.py` provides lazy `__getattr__` re-exports of all 40+ error types. All 9 subsystem error modules updated to import from `nucleusiq.errors.base`. Backward-compatible: `from nucleusiq.errors import NucleusIQError` still works.
- `**core/agents/usage/` package** — extracted `usage_tracker.py` and `pricing.py` from `components/` into a dedicated `usage/` package with public `__init__.py` re-exports. Old shim files deleted (no backward-compatibility wrappers — clean break).
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

- `**_setup_execution`** — usage tracker and execution tracer are reset immediately after plugin counter reset (before `BEFORE_AGENT`), so halted or failed setups do not leak prior-run tracer data. Tracer creation gated on `AgentConfig.enable_tracing`.
- All internal imports updated to canonical paths (`nucleusiq.agents.usage.*`, `nucleusiq.errors.base`).
- `components/usage_tracker.py` and `components/pricing.py` shim files **deleted** — all imports now use canonical `nucleusiq.agents.usage.*` paths.
- `standard_mode.run()` and `direct_mode.run()` now raise `AgentExecutionError` with mode context instead of returning error strings.
- `base_agent.py` retry loop raises `AgentExecutionError`/`AgentTimeoutError` instead of `RuntimeError`/`TimeoutError`.
- OpenAI provider: `AuthenticationError` instead of `ValueError` for missing API key; `AttachmentUnsupportedError` for unknown attachment types; `ToolValidationError` for MCP config; `StructuredOutputError`/`SchemaParseError` for structured output parsing; `ContentFilterError`/`ContextLengthError` in retry.
- Gemini provider: `AuthenticationError` instead of `ValueError` for missing API key; `StructuredOutputError`/`SchemaParseError` for structured output parsing; `ContentFilterError`/`ContextLengthError` in retry.

### Removed

- **Dead GPT-2 tokenizer code** — removed `get_tokenizer()`, `_get_token_ids_default_method()`, `get_token_ids()`, `get_num_tokens()`, and `custom_get_token_ids` from `BaseLanguageModel`. The GPT-2 tokenizer was never called by the framework and is inaccurate for modern models (GPT-4 uses `cl100k_base`/`o200k_base`; Gemini has its own tokenizer). The `tokenizers` library dependency is no longer needed.
- `**BaseLanguageModel` no longer extends `ABC`** — it is now a plain mixin providing the `metadata` field. `BaseLLM` remains the abstract base class with `@abstractmethod call()`.

### Token Estimation (new contract)

- `**BaseLLM.estimate_tokens(text)**` — new base method using `len(text) // 4` heuristic as a zero-dependency, cross-provider default. Providers override with precise implementations:
  - **OpenAI**: uses `tiktoken.encoding_for_model()` (already shipped)
  - **Gemini**: uses `len(text) // 4` (Gemini `count_tokens` API available for precise counts)
- `**ContextWindowPlugin`** and `**TokenBudgetMemory**` accept `token_counter` callbacks — users can wire `llm.estimate_tokens` for provider-accurate counting:
  ```python
  ContextWindowPlugin(max_tokens=8000, token_counter=llm.estimate_tokens)
  ```
- **UsageTracker** is unaffected — it reads `prompt_tokens`/`completion_tokens` directly from provider API responses (always accurate).

### Developer Support

- **Pyrefly type checking** — CI pipeline now includes a `type-check` job using [Pyrefly](https://pyrefly.org/) (Meta, MIT license, Rust-based, 1.85M lines/sec). Catches undefined names, null safety violations, type mismatches, and override signature inconsistencies at "compile time" — before tests or deployment.
  - CI flow: `ruff check` → `ruff format` → `**pyrefly check`** → `pytest` → `import-check` → `security` → `build`
  - Added to `dependency-groups.lint` alongside `ruff` (dev-only, not in wheel).
  - Requires `editable_mode=strict` install for `setuptools` `package-dir` mapping resolution.
  - Provider configs use `ignore-missing-imports` for cross-package deps (`nucleusiq.*`, `google.*`) — these resolve at CI time when all packages are installed; locally they gracefully fall back to `Any`.

### Packages


| Package            | Version   | Note                                                                                            |
| ------------------ | --------- | ----------------------------------------------------------------------------------------------- |
| `nucleusiq`        | **0.7.4** | ExecutionTracer, configurable tracing, error/usage package restructure, exhaustive error wiring |
| `nucleusiq-openai` | **0.6.1** | Custom error types wired, ContentFilter/ContextLength mapped (requires `nucleusiq>=0.7.4`)      |
| `nucleusiq-gemini` | **0.2.2** | Custom error types wired, ContentFilter/ContextLength mapped (requires `nucleusiq>=0.7.4`)      |


---

## [0.7.3](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.3) — 2026-04-02

### Fixed

- **Gemini `function_response.name` cannot be empty** — tool result messages (`role="tool"`) now carry `name=tc.name` in `ChatMessage` across all execution modes (standard, direct, base streaming). Previously, `name` was `None`, causing Gemini API 400 errors on the second turn of any tool-calling conversation.
- **Gemini `function_response.response` must be dict** — `response_normalizer.py` now wraps non-dict payloads (e.g. `json.dumps("string")`) in `{"result": ...}` to satisfy the `google-genai` SDK's Pydantic validation.
- **Defense-in-depth name inference** — if `name` is still missing from an incoming tool message dict, the Gemini response normalizer infers it from the prior assistant message's `tool_calls` by matching `tool_call_id`.
- **Gemini `$ref`/`$defs` inlining** — `_clean_schema` in structured output builder now inlines `$ref` references (matching OpenAI's cleaner quality). Previously, `$defs` were silently dropped, producing broken schemas for nested Pydantic models.

### Added

- **Tools + structured output guard** — `BaseGemini.call()` detects when both `tools` and `response_format` are set, logs a warning, and drops JSON schema mode. Gemini API rejects `response_mime_type: application/json` combined with function calling; this guard prevents the 400 error.
- `**OutputSchema` tuple handling** — `BaseGemini.call()` now unpacks the `(provider_format, schema)` tuple from core's `StructuredOutputHandler`, matching the OpenAI provider's behavior.
- **Gemini integration tests: `test_gemini_tool_round_trip.py`** — full multi-turn tool loop with tools resent on second call, JSON string content round-trip, multiple tool calls, structured output + tools guard test.
- **OpenAI integration test scaffold** — `tests/integration/` directory created with `conftest.py` + `test_openai_tool_round_trip.py` mirroring Gemini's integration test structure. Uses `@pytest.mark.integration` (previously defined but never applied in OpenAI tests).
- **Nested Pydantic model unit tests** — `test_nested_pydantic_model_refs_inlined` and `test_deeply_nested_refs_inlined` verify `$ref` inlining works for 2+ levels of model nesting.

### Changed

- **Notebook: `research_analyst_tcs.ipynb`** — rewritten as a complete framework showcase demonstrating all core features (tools, memory, plugins, streaming, structured output, cost tracking) with pandas DataFrames, matplotlib visualizations, and a feature proof dashboard.

### Packages


| Package            | Version   | Note                                                 |
| ------------------ | --------- | ---------------------------------------------------- |
| `nucleusiq`        | **0.7.3** | Tool message `name` field fix                        |
| `nucleusiq-openai` | 0.6.0     | No change                                            |
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
- `**AgentResult` response contract** — `Agent.execute()` now returns a typed, immutable `AgentResult` (Pydantic `BaseModel`, `frozen=True`) instead of raw `Any`. Includes: `output`, `status` (SUCCESS/ERROR/HALTED), `error`, `error_type`, `duration_ms`, `agent_id`, `agent_name`, `task_id`, `mode`, `model`, `created_at`, `usage`, and extension fields for future observability (`tool_calls`, `llm_calls`, `plugin_events`, etc.).
- **Backward compatible** — `str(result)` returns the output text, `bool(result)` returns `True` on success. Existing `print(result)` and `if result:` patterns continue to work.

### Changed

- **Re-parented existing errors** — `PluginError`, `PluginHalt`, `StructuredOutputError`, `WorkspaceSecurityError` now extend `NucleusIQError` instead of bare `Exception`. Enables `except NucleusIQError` catch-all.
- `NucleusIQError` canonical location moved to `nucleusiq.errors` (re-exported from `nucleusiq.llms.errors` for backward compat).
- All error classes now carry structured context attributes (e.g. `tool_name`, `provider`, `status_code`, `mode`, `task_id`).
- `Agent.execute()` catches all exceptions and wraps them in `AgentResult(status="error")` — no more unhandled exceptions from `execute()`.

### Packages


| Package            | Version   | Note                              |
| ------------------ | --------- | --------------------------------- |
| `nucleusiq`        | **0.7.2** | Exception hierarchy + AgentResult |
| `nucleusiq-openai` | 0.6.0     | No change                         |
| `nucleusiq-gemini` | 0.2.0     | No change                         |


---

## [0.7.1](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.1) — 2026-03-30

### Fixed

- `**.env` loading broken for pip-installed consumers** — `core/__init__.py` used a hard-coded `Path(__file__).parents[2]` to locate `.env`, which resolved into `site-packages/` for anyone who `pip install nucleusiq`. Replaced with `load_dotenv(override=False)` (no path argument), which uses `python-dotenv`'s built-in `find_dotenv()` to search from the caller's **working directory** upward. Any project with a `.env` in its root now works out of the box.

### Packages


| Package            | Version   | Note                                                   |
| ------------------ | --------- | ------------------------------------------------------ |
| `nucleusiq`        | **0.7.1** | Patch fix for `.env` loading                           |
| `nucleusiq-openai` | 0.6.0     | No change (requires `nucleusiq>=0.7.0`, accepts 0.7.1) |
| `nucleusiq-gemini` | 0.2.0     | No change (requires `nucleusiq>=0.7.0`, accepts 0.7.1) |


---

## [0.7.0](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.0) — 2026-03-30

### Security

- `**requests`** — minimum raised to `>=2.33.0` (CVE-2026-25645: insecure temp-file reuse in `extract_zipped_paths()`). Lockfiles refreshed for all transitive and dev dependencies.
- `**pygments**` (transitive, dev/test) — constrained to `>=2.20.0` via `[tool.uv] constraint-dependencies` to avoid **CVE-2026-4539** (ReDoS in `AdlLexer`). Lockfiles updated to `2.20.0`.

### Breaking changes

- **Provider packages** — `nucleusiq-openai` is now **0.6.0** and `nucleusiq-gemini` is **0.2.0**, both requiring `**nucleusiq>=0.7.0`**. Pin or upgrade core and providers together.

### Fixed

- **PyPI wheel packaging** — `nucleusiq.tools.builtin` was omitted from `[tool.setuptools] packages`, so `pip install nucleusiq` produced a broken install (`ModuleNotFoundError: No module named 'nucleusiq.tools.builtin'`). The subpackage is now included in the wheel.

### Added

- `**scripts/verify_core_package_layout.py`** — fails CI if any `core/**/__init__.py` package is missing from `pyproject.toml` (prevents recurrence).
- **CI: wheel smoke test** — after building the core wheel, install only from `dist/*.whl` in a clean venv and import `nucleusiq.tools.builtin` (catches wheel-only failures; editable installs always masked this).

### CI

- `**actions/upload-artifact`** — v6 → v7 (workflow maintenance).

### Dependencies (lockfile refresh)

- `requests` 2.32.5 → **2.33.0** (security)
- `pygments` 2.19.2 → **2.20.0** (security)
- `python-dotenv` 1.2.1 → **1.2.2**
- `pytest-cov` 7.0.0 → **7.1.0**
- `ruff` 0.15.2 → **0.15.8**
- `openai` (SDK) → **2.30.0**

### Packages


| Package            | Version   | Requires           |
| ------------------ | --------- | ------------------ |
| `nucleusiq`        | **0.7.0** | —                  |
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
- `**BaseLLM.call_stream()`** — abstract streaming contract with non-streaming fallback; `MockLLM.call_stream()` for testing
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
- Upgraded issue templates to YAML forms (bug report, feature request, question)mode
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

- `**BaseTool`** — LLM-agnostic tool interface with JSON schema generation
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
- `**PluginManager`** — chain-of-responsibility hook pipeline
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

## [Unreleased](https://github.com/nucleusbox/NucleusIQ/compare/v0.7.6...HEAD)

### Planned for v0.7.7+

- Hyperparameter auto-tuning + adaptive API constraint handling
- Agent Types: ReAct integration into mode system, Chain-of-Thought as config flag
- Agent DX: String argument support for `execute()`
- New LLM Providers: Anthropic, Ollama
- Gemini advanced features: Interactions API, Batch API, Deep Research Agent, File Search
- CostTracker Agent integration (`agent.last_cost`)
- See `docs/BACKLOG.md` for full list

