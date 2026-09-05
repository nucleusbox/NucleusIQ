# Changelog

All notable changes to NucleusIQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.7.13] — 2026-09-05

> Adds a generic **OpenAI-compatible** provider so any server speaking the OpenAI Chat Completions protocol — vLLM, SGLang, TGI, llama.cpp, LM Studio, NVIDIA NIM, OpenRouter, Together, Fireworks, Azure OpenAI v1 — works without a dedicated package. Replaces model-name guessing in core with a declared provider identity, fixes three response/streaming bugs found by testing against a real endpoint, and clears all 32 open Dependabot advisories.
>
> **Coordinated package versions:**
>
> - `nucleusiq` **0.7.13**
> - `nucleusiq-openai-compatible` **0.1.0** — **new**, Stable
> - `nucleusiq-openai` **0.7.1** — Responses API usage accounting fixes
> - `nucleusiq-mcp` **0.1.1** — security floor `mcp>=1.28.1`
> - `nucleusiq-gemini` **0.3.1**, `nucleusiq-anthropic` **0.2.1**, `nucleusiq-ollama` **0.2.1**, `nucleusiq-groq` **0.1.1** — declare `PROVIDER_NAME`
>
> Only `nucleusiq-openai-compatible` floors on `nucleusiq>=0.7.13`; it needs the new resolver, because the pre-0.7.13 class-name match reads `OpenAICompatibleLLM` as `"openai"` and would send OpenAI-cloud-shaped structured output to a self-hosted server. Every other provider keeps its `nucleusiq>=0.7.12` floor — `PROVIDER_NAME` is inert on older cores, where class-name matching still yields the same answer. **4,382 tests passing across the monorepo.**

### Monorepo test gate at the time of this entry (all non-live, no API keys required)

| Package | Tests | Coverage | Status |
| --- | --- | --- | --- |
| `nucleusiq` | **2607 passed** (2 skipped, 8 live deselected) | n/a (no gate) | green |
| `nucleusiq-openai` | **239 passed** (5 skipped) | n/a (existing gate) | green |
| `nucleusiq-gemini` | **292 unit passed** | n/a (existing gate) | green |
| `nucleusiq-anthropic` | **157 passed** | ≥ 95% | green (gate ≥ 95%) |
| `nucleusiq-groq` | **79 passed** | ≥ 90% | green (gate ≥ 90%) |
| `nucleusiq-ollama` | **98 passed** | ≥ 95% | green (gate ≥ 95%) |
| `nucleusiq-openai-compatible` | **675 passed** (16 live deselected) | **99.40%** | green (gate ≥ 95%) |
| `nucleusiq-mcp` | **235 passed** | ≥ 90% | green (gate ≥ 90%) |
| **Total** | **4,382 passing** | — | — |

`ruff check src/` and `ruff format --check src/` are clean on all 679 files under the repo-root `ruff.toml`; `pyrefly` reports 0 errors on every package; `scripts/verify_core_package_layout.py` reports OK on all 8 packages.

### Added — `nucleusiq-openai-compatible` 0.1.0 (new package, Stable)

One provider for every OpenAI-protocol server, built around **bring-your-own-model / bring-your-own-key**. No model-name heuristics anywhere: capabilities are declared, not guessed, because a generic provider cannot know what `my-finetune-v3` supports.

```python
from nucleusiq_openai_compatible import OpenAICompatibleLLM

llm = OpenAICompatibleLLM(
    base_url="http://gpu-node-1:8000/v1",
    model="gemma-4-27b-it",
    api_key="token-abc123",   # omit entirely for an unauthenticated server
    context_window=32_768,    # declared; probed only as a fallback
    engine="vllm",
)
```

- **Engine presets** (`ENGINE_PRESETS`, `EngineProfile`) for `vllm`, `sglang`, `tgi`, `llamacpp`, `lmstudio`, `ollama`, `generic` — recording per-engine support for tools, JSON schema, parallel tool calls, streamed usage and reasoning. `generic` is the conservative default.
- **BYOK auth strategies** — `NoAuth`, `BearerAuth`, `HeaderAuth` (for gateways wanting a custom header), plus `build_auth`. Credentials may be literals, env-var names or callables; a callable is resolved **once per request** and shared between headers and the SDK key, so token-minting hooks aren't billed twice.
- **Structured output with a real degradation path** — `DropPolicy`, `ErrorPolicy`, `PromptPolicy`. Engines without server-side schema enforcement fall back to `json_object` plus a prompt-injected schema instead of silently returning prose. Also works around the vLLM bug where `response_format` + `tools` + `tool_choice="auto"` suppresses tool calls entirely.
- **Reasoning/thinking models** — `reasoning_content` is separated from answer text on both the streaming and non-streaming paths, with `chat_template_kwargs` and `reasoning_effort` passthrough.
- **Token counting** — optional `[tokenizer]` extra uses a real HF tokenizer; otherwise a documented heuristic counter, since `tiktoken` is meaningless for open-weight vocabularies.
- **Preflight validation** — `ValidationReport` checks reachability, auth, model presence via `/v1/models`, and capability claims before an agent run, turning a misconfigured endpoint into one clear message instead of a mid-run failure.
- **Credential redaction** across errors, logs and `repr`.
- **Packaging** — `tests/` and `examples/` are excluded from **both** the wheel and the sdist. The sdist default ships everything not gitignored, so the exclusion is explicit and asserted by a test that builds real artifacts and inspects the members.

### Changed — core `nucleusiq` 0.7.13 — provider identity is declared, not guessed

- **`BaseLLM.PROVIDER_NAME: ClassVar[str | None]`** — new declaration mirroring the existing `NATIVE_TOOL_TYPES` / `NATIVE_ATTACHMENT_TYPES` pattern. `get_provider_from_llm()` reads it first and falls back to class-name matching only for adapters predating the contract. All six first-party adapters now declare it, so provider identity survives subclassing and renaming.
- **Removed `supports_native_output()`** from `nucleusiq.agents.structured_output.resolver`. It hardcoded OpenAI model prefixes (`gpt-3`, `gpt-4`, `o1`, …) that went stale as vendors shipped new models, and it was dead: both branches of `_auto_select_mode` returned NATIVE, so it never changed an outcome, and nothing imported it. It was never exported from `structured_output/__init__.py`, so this is not a public API break.
- **`OutputMode.AUTO` still resolves to NATIVE for every adapter.** Core deliberately does *not* ask adapters whether their backend can enforce a schema server-side: `OutputMode.implemented_modes()` is `{AUTO, NATIVE}`, so routing to PROMPT would raise, and degrading the transport is the adapter's job — it knows its engine, core doesn't. NATIVE means "hand the schema to the adapter", not "the server implements `json_schema`".

### Fixed — `nucleusiq-openai` 0.7.1 — Responses API token accounting

Both bugs affected only the Responses API path (native server tools, reasoning models) and were invisible to mocked tests:

- **Non-streaming `call()` reported zero tokens.** `normalize_responses_output` built its response without populating `usage`, and `UsageTracker.record_from_response` / `build_llm_call_record` both start at `response.usage` and return early when it's absent. Every run reported 0 tokens and $0.00 cost — wrong in the direction nobody notices.
- **Streaming reported a total with no prompt or completion split.** The Responses API renamed the counters to `input_tokens` / `output_tokens`; the COMPLETE event forwarded that vocabulary, but the framework reads the Chat Completions names. Both are now mapped to `prompt_tokens` / `completion_tokens`, with `reasoning_tokens` carried through from `output_tokens_details`.

### Fixed — `nucleusiq-openai-compatible` — three bugs caught by live-endpoint testing

Found by running the suite against a real endpoint (Ollama Cloud), and unreachable by mocks, which return a scripted reply no matter what they're sent:

- **The tool loop died on its second call.** Core serialises a tool call flat as `{"id", "name", "arguments"}` and leaves the wire dialect to the provider; Chat Completions requires it nested under `function`. The first request carries no tool history so it succeeded; the request echoing the call back alongside its result failed with `400 invalid tool call arguments`. `sanitize_messages` now translates assistant tool calls, copying a message only when a rewrite is needed.
- **Streaming agents silently ignored every tool.** The COMPLETE event omitted `tool_calls`, and `base_mode` streaming reads that key to decide whether to run the tool loop. Nothing errored — the loop just never fired. The event now also publishes `usage`, `model`, `finish_reason` and `reasoning_content`.
- **`response_format` passed as a call kwarg bypassed the policy layer**, reintroducing the vLLM tool-suppression bug. A new inbound normaliser routes it through the policy regardless of shape (Pydantic model, OpenAI wire format, NucleusIQ generic format, or `(provider, schema)` tuple).

### Changed — `nucleusiq-openai-compatible` — `ollama` preset no longer claims schema support

Measured, not assumed: the Ollama `/v1` shim accepts both `json_schema` and `json_object` and honours neither, returning markdown-fenced prose with whatever keys the model chose. Claiming support sent a schema the server discarded, so an agent believed it had a validated object and got prose. `supports_json_schema=False` routes through `json_object` plus a prompt-injected schema, which does return the requested shape. Ollama's **native** API does support schemas via `format`; `nucleusiq-ollama` uses it.

### Fixed — packages declared fewer dependencies than they import

`nucleusiq-openai` imported `httpx` at module scope in `_shared/retry.py` (to classify transport errors) while declaring only `nucleusiq`, `openai` and `tiktoken`. It worked because the `openai` SDK pulled `httpx` in transitively — until `openai` 3.x stopped shipping it, at which point `import nucleusiq_openai` raised `ModuleNotFoundError` on a fresh install. Since the floor was an unbounded `openai>=1.0`, anyone installing after openai 3.0's release got the broken combination.

Every provider had some version of this. All of them now declare what they import:

| Package | Added |
| --- | --- |
| `nucleusiq-openai` | `httpx>=0.27,<1`, `pydantic>=2.11.0,<3.0`, and an `openai<3.0` bound |
| `nucleusiq-anthropic`, `nucleusiq-groq`, `nucleusiq-ollama`, `nucleusiq-openai-compatible` | `httpx>=0.27,<1`, `pydantic>=2.11.0,<3.0` |
| `nucleusiq-gemini`, `nucleusiq-mcp` | `pydantic>=2.11.0,<3.0` |

The `openai<3.0` bound is deliberate: 3.x is a major SDK revision that has not been exercised against this adapter's Responses API paths, and every test in that package is mocked, so a green suite would say nothing about it. Raising it belongs in its own change, verified against the live API. Core's three optional imports (`chardet`, `pdfplumber`, `tomli`) are already guarded by `try/except ImportError` with graceful degradation and are correctly left undeclared.

No CI job could have caught this, because `test-*`, `import-check` and `type-check` all install siblings and dev tooling into one environment where any neighbour supplies the missing module. A new `dependency-completeness` job (`scripts/verify_dependency_completeness.py`) checks it the only way it can be checked: one virtualenv per package, holding that package and its declared dependencies alone, then importing the public API. The isolation is the test, so it is deliberately kept out of the other jobs.

### Fixed — lint was not reproducible

The `lint` job ran an unpinned `pip install ruff`, so its verdict depended on release timing rather than on the code. ruff 0.16 began formatting fenced code blocks inside `.md` files, which failed the build on four provider READMEs nobody had touched. ruff is now pinned to `0.16.6`, and `*.md` is excluded from the formatter — reflowing prose samples collapses the aligned trailing comments that make the option tables readable. `docstring-code-format` still applies to docstrings, which is where it was wanted.

### Security — all 32 open Dependabot advisories resolved

- **`nucleusiq-mcp` floor raised to `mcp>=1.28.1`** (from `>=1.27`), including the `oauth` extra. This is the only advisory that reached published dependency metadata. 1.27.x shipped three advisories against the server transports the adapter hands to the SDK: HTTP transports served session requests without verifying the authenticated principal, experimental task handlers let any client read and cancel another client's tasks (both fixed in 1.27.2), and the WebSocket transport lacked Host/Origin validation (fixed in 1.28.1). A `>=1.27` floor still *permits* resolving to a vulnerable release.
- **All 8 lockfiles regenerated** (`uv lock --upgrade`), clearing the remaining 31 alerts — `cryptography` → 50.0.1 (Bleichenbacher oracle in PKCS#7 decryption, wildcard-DNS `permittedSubtrees` escape, vulnerable bundled OpenSSL), `pyasn1` → 0.6.4 (three DoS vectors in BER/CER/DER decoding), `starlette` → 1.6.0 (`request.form()` limits ignored, authority poisoning), `python-multipart` → 0.0.32 (quadratic querystring parsing, parameter smuggling), `pydantic-settings` → 2.15.0 (symlink escape from `secrets_dir`), `mcp` → 1.29.1.
- Regenerating the locks in one resolution supersedes the ten open Dependabot PRs, which each touched the same files and would have conflicted pairwise.

### Changed — CI action and tool versions

- GitHub Actions majors bumped across all three workflows: `actions/checkout` v6 → **v7**, `actions/setup-python` v6 → **v7**, `actions/cache` v5 → **v6**, `actions/setup-node` v6 → **v7**. Applied directly rather than by merging the open Dependabot PR, which targeted a `ci.yml` that has since grown several jobs and would have conflicted.
- `nucleusiq-anthropic`'s `lint` group required `pyrefly>=0.33,<1`, a cap that excluded the `pyrefly` 1.x that `nucleusiq` and `nucleusiq-mcp` already require and that CI installs. Aligned to `pyrefly>=1.0,<2`.

### Changed — verify scripts fail closed when a provider is added

`scripts/verify_core_package_layout.py` and `scripts/verify_dependency_completeness.py` already listed `nucleusiq-openai-compatible` and already run in CI (`import-check` and `dependency-completeness`). They now also discover every Hatch package under `src/providers/{llms,inference,tools}` and fail if the hardcoded registry is missing one — so a ninth provider cannot ship without both jobs being updated. Layout additionally requires sdist `exclude` to list `tests` and `examples`. Covered by `src/nucleusiq/tests/unit/test_verify_scripts.py`.

### Fixed — Gemini README examples did not construct a valid `Agent`

Community PR [#38](https://github.com/nucleusbox/NucleusIQ/pull/38) (Kostas Tsoukalochoritis) reported that the `nucleusiq-gemini` README copy-paste examples used `model=` and `instructions=` on `Agent`. Those are not fields — Pydantic ignores them — so the snippet never set a system prompt. The examples now match the OpenAI README: required `prompt=ZeroShotPrompt().configure(...)`, `Task` for `execute()`, and `result.output`. Streaming samples use `event.token` (the `StreamEvent` field) rather than a non-existent `event.delta`.

### Added — CI/CD coverage for the new package

`nucleusiq-openai-compatible` was absent from every pipeline job except `lint`. Now wired into: its own `test-openai-compatible` matrix job (Python 3.10 + 3.12, `--cov-fail-under=95`), the `test-uv` install check, `type-check` (pyrefly), `import-check` (including an assertion that `PROVIDER_NAME == "openai_compatible"`), the `security` pip-audit sweep, the `build` matrix with `twine check`, and a `publish-openai-compatible` job in `publish.yml` gated on the same PyPI version check as its siblings. Its `test` dependency group gained `httpx`, `hatchling` and `tokenizers` so the loopback HTTP server, the real-artifact packaging test and the accurate token counter all execute rather than skip.

---

## [0.7.12] — 2026-05-26

> Single coordinated release promoting every alpha/beta provider to its first stable line, plus the provider-agnostic native-tool observability that powers it.
>
> **Coordinated package versions:**
>
> - `nucleusiq` **0.7.12**
> - `nucleusiq-anthropic` **0.2.0** — Stable (Phase B feature-complete: native server tools, prompt caching, extended thinking, server-tool observability, 3 example scripts, 6 live integration tests)
> - `nucleusiq-ollama` **0.2.0** — Stable (+ vision)
> - `nucleusiq-groq` **0.1.0** — Stable
> - `nucleusiq-mcp` **0.1.0** — Stable
> - `nucleusiq-openai` **0.7.0**
> - `nucleusiq-gemini` **0.3.0**
>
> All providers floor on `nucleusiq>=0.7.12`. **3,705+ tests passing across the monorepo** (incl. 6 live Anthropic Phase B tests against the real API). After this release the project returns to a bug-fix / single-provider cadence.

### Monorepo test gate at the time of this entry (all non-live, no API keys required)

| Package | Tests | Coverage | Status |
| --- | --- | --- | --- |
| `nucleusiq` | **2603 passed** (2 skipped) | n/a (no gate) | green |
| `nucleusiq-anthropic` | **151 unit passed + 6 live integration passed** | 95.91% | green (gate ≥ 95%) |
| `nucleusiq-openai` | **232 passed** (5 skipped) | n/a (existing gate) | green |
| `nucleusiq-gemini` | **292 unit passed** (1 live integration unrelated) | n/a (existing gate) | green |
| `nucleusiq-groq` | **79 passed** | 92.51% | green (gate ≥ 90%) |
| `nucleusiq-ollama` | **98 passed** | 99.85% | green (gate ≥ 95%) |
| `nucleusiq-mcp` | **248 passed** | n/a | green |
| **Total** | **3,705+ passing** (incl. 6 live Anthropic Phase B integration tests against `claude-sonnet-4-5-20250929`) | — | — |

`scripts/verify_core_package_layout.py` reports OK on every package; `ruff` is clean on every file touched in this sprint.

### Added — core (Batch A) — already in this branch

**Provider-agnostic native-tool observability scaffolding** in `nucleusiq` (additive, no breaking changes — version bump to **`0.7.12`** pending):

- **`ToolCallRecord.executed_by: Literal["local", "provider"] = "local"`** — new field on `nucleusiq.agents.AgentResult.tool_calls` that lets the tracer distinguish locally-executed tools from provider/server-executed ones (Anthropic `web_search`, OpenAI `web_search` / `code_interpreter` / `file_search`, Gemini `google_search` / `code_execution`, Groq `compound_custom`, …). `build_tool_call_record` defensively coerces unknown values to `"local"`.
- **`LLMCallRecord` enrichment** — new fields populated by record builders:
  - `provider: str | None` — provider identifier (`"anthropic"`, `"openai"`, `"gemini"`, …)
  - `request_id: str | None` — provider-side response id for cross-system correlation
  - `organization_id: str | None` — best-effort header lookup
  - `stop_reason: str | None` — provider-reported finish reason
  - `cache_read_input_tokens: int`, `cache_creation_input_tokens: int` — per-bucket prompt-cache usage
  - `metadata: dict[str, Any]` — generic bag for provider-specific extras
- **`build_llm_call_record` / `build_llm_call_record_from_stream`** auto-extract these from both **Anthropic-style** (`usage.cache_read_input_tokens`, `stop_reason`) and **OpenAI-style** (`usage.prompt_tokens_details.cached_tokens`, `finish_reason`) response shapes via new internal helpers `_extract_str`, `_extract_int`, `_usage_cache_tokens`.
- **Autonomous sub-agent telemetry** (`modes/autonomous/telemetry.py`) — propagates all new fields from sub-agent `LLMCallRecord`s into the parent `AgentResult` so fidelity is preserved across nested decompositions.
- **Tests** — 9 new dedicated unit tests in `tests/unit/test_execution_tracer.py` covering: `ToolCallRecord.executed_by` coercion, defaults on `LLMCallRecord`, Anthropic-style + OpenAI-style cache-token extraction, auto-extraction of `request_id` / `stop_reason`, and stream-record propagation.

### Added — `nucleusiq-anthropic` 0.2.0 (Stable) — Phase B feature-complete

Promoted from **Alpha (`0.1.0a1`) → Stable (`0.2.0`)**; `Development Status :: 5 - Production/Stable`; floor bumped to **`nucleusiq>=0.7.12`**. Description updated to "Anthropic Claude provider (native server tools, prompt caching, extended thinking, server-tool observability)".

**New `AnthropicTool` factory** (`nucleusiq_anthropic.tools.anthropic_tool`) — first-class native server tool API:

```python
from nucleusiq_anthropic import AnthropicTool, AnthropicLLMParams, BaseAnthropic

agent = Agent(
    llm=BaseAnthropic(
        model_name="claude-opus-4-20250514",
        llm_params=AnthropicLLMParams(
            thinking="medium",          # extended thinking, 8000-token budget
            cache_tools=True,           # prompt-cache tool definitions
            cache_system=True,          # prompt-cache system prompt
            strict_tools=True,          # strict tool-use schema
        ),
    ),
    tools=[
        AnthropicTool.web_search(max_uses=3),    # server-side; NO beta header needed
        AnthropicTool.web_fetch(),               # server-side; auto-injects anthropic-beta: web-fetch-2025-09-10
        AnthropicTool.code_execution(),          # server-side; auto-injects anthropic-beta: code-execution-2025-05-22
        lookup_order,                            # your own @tool — still works
    ],
)
```

- **`NATIVE_TOOL_TYPES = {"web_search", "web_fetch", "code_execution"}`** — registry of supported native tools.
- **`NATIVE_TOOL_WIRE_TYPES`** — dated wire identifiers (`web_search_20250305`, `web_fetch_20250910`, `code_execution_20250522`) injected automatically by `marker_to_wire()`.
- **`NATIVE_TOOL_BETA_HEADERS`** — required `anthropic-beta` tokens (`web-fetch-2025-09-10`, `code-execution-2025-05-22`) collected automatically by `required_beta_headers(tools)` and merged with any user-supplied `anthropic-beta` headers in `build_create_kwargs`.
- **`tools/converter.py`** — `to_anthropic_tool_definition` now routes `AnthropicTool` markers through `marker_to_wire` before falling back to OpenAI-style / `input_schema` conversion. `spec_looks_native` recognises both internal markers and raw dated wire identifiers. Unknown native names are returned verbatim so Anthropic surfaces a precise error instead of NucleusIQ silently mutating the spec.

**Prompt caching** — `AnthropicLLMParams.cache_tools` and `.cache_system` thread through `build_create_kwargs`:

- `_shared/wire.py`: `flatten_tools(..., cache_tools=True)` adds a `cache_control` block to the last tool definition; `system_with_cache(text, cache_system=True)` upgrades the plain system string into a block list with `cache_control`.
- Private marker keys (`_cache_tools`, `_cache_system`, `_strict_tools`, `_disable_parallel_tool_use`) are interpreted by `build_create_kwargs` and then stripped by `drop_unsupported_sampling` before reaching `messages.create`.

**Extended thinking** — `AnthropicLLMParams.thinking` accepts:

- `bool` — `True` → minimal thinking, `False` → disabled
- `"low"|"medium"|"high"|"max"` (new `ThinkingEffort` literal) → resolved via `_THINKING_EFFORT_BUDGETS` to `{"type": "enabled", "budget_tokens": N}`
- `dict` — passthrough for full control

**Strict tools / parallel-tool-use guard** — `AnthropicLLMParams.strict_tools` adds `strict: True` to every custom (non-native) tool definition; `AnthropicLLMParams.disable_parallel_tool_use` augments `tool_choice` with `disable_parallel_tool_use: true`.

**Server-tool observability** — Anthropic's server-executed `tool_use` blocks no longer mix with client-side tool calls:

- New `ServerToolCall` Pydantic model (`_shared/response_models.py`) capturing `id`, `name`, `input`, `result`.
- `UsageInfo` extended with `cache_read_input_tokens` and `cache_creation_input_tokens`.
- `AnthropicLLMResponse` extended with `stop_reason`, `organization_id`, `server_tool_calls: list[ServerToolCall]`, plus a `request_id` property alias for `response_id`.
- **Non-stream path** (`nb_anthropic/messages.py`) — `normalize_message_response` splits incoming `tool_use` blocks via `_is_server_tool_block`. Client blocks become `ToolCall`s; server blocks land in `AnthropicLLMResponse.server_tool_calls`. `_extract_organization_id` performs best-effort header extraction.
- **Stream path** (`nb_anthropic/stream_adapter.py`) — `_process_raw_events` separates server vs client tool calls in `content_block_start` events, surfaces `stop_reason` / `request_id` / per-cache-bucket usage / `server_tool_calls` on the terminal `COMPLETE` `StreamEvent.metadata`. `thinking_delta` events handled correctly.

**Public API** — `from nucleusiq_anthropic import AnthropicTool, ThinkingEffort, AnthropicLLMParams, BaseAnthropic, ...` (`__all__` updated).

**Server-tool observability hardening (post-Phase B normalizer fix):**

Anthropic emits `server_tool_use` blocks (not `tool_use`) for native tools, plus per-tool result blocks like `code_execution_tool_result` and `web_search_tool_result` (not generic `tool_result`). The non-stream `normalize_message_response` was updated to:

- Treat `tool_use` and `server_tool_use` as the same id/name/input shape, with `server_tool_use` always routed to `ServerToolCall`.
- Match any block whose type ends in `_tool_result` (web_search, code_execution, web_fetch, future variants) as a server tool-result attachment.
- Reduce the result payload via new `_coerce_tool_result_content` helper — accepts Pydantic-like objects (`model_dump()`), dicts, lists, scalars, or arbitrary objects (with JSON-safe fallback) — so `ServerToolCall.result` is always JSON-serialisable.

**Phase B example scripts** (`src/providers/llms/anthropic/examples/agents/`):

- `10_anthropic_native_tools.py` — calls `BaseAnthropic.call(...)` directly with `AnthropicTool.web_search(max_uses=2)` then `AnthropicTool.code_execution()`; prints the populated `server_tool_calls` list and stop reason.
- `11_anthropic_prompt_caching.py` — runs an `Agent` twice with a 9,780-character system prompt and `AnthropicLLMParams(cache_system=True)`; prints per-`LLMCallRecord` `prompt_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `stop_reason` so the cache hit is visible.
- `12_anthropic_extended_thinking.py` — runs an `Agent` at `thinking="low"` and `thinking="medium"` (resolved to 2 000 / 8 000 budget tokens) with `max_output_tokens=16_384` (must exceed the budget) and `temperature=1.0` (required by Anthropic when thinking is on).
- Default model: `claude-sonnet-4-5-20250929` (override via `ANTHROPIC_PHASE_B_MODEL`). All three scripts have been verified end-to-end against the live API.

**Live integration tests** (`src/providers/llms/anthropic/tests/integration/test_anthropic_phase_b_live.py`):

Gated by `pytest -m integration` and `ANTHROPIC_API_KEY`; excluded from the default test run and from CI. Each test skips cleanly if the selected model isn't available on the active API key. Covers:

- `test_live_web_search_emits_server_tool_call` — `AnthropicTool.web_search()` surfaces `ServerToolCall(name="web_search")` in `server_tool_calls`.
- `test_live_code_execution_emits_server_tool_call` — `AnthropicTool.code_execution()` surfaces `ServerToolCall(name="code_execution")` with a populated `.result`.
- `test_live_prompt_caching_reads_cache_on_second_call` — two calls with the same long system + `cache_system=True` produce non-zero `cache_creation_input_tokens + cache_read_input_tokens`.
- `test_live_extended_thinking_completes[low]` / `[medium]` — `thinking="low"` and `"medium"` end-to-end with non-empty completion text and a populated `stop_reason`.
- `test_live_disable_parallel_tool_use_round_trip` — `disable_parallel_tool_use=True` reaches the wire without a 400 error.

**Tests + coverage:**

- **151 unit tests** + **6 live integration tests** passing; **95.91% line coverage** (gate `--cov-fail-under=95`).
- New `tests/test_anthropic_tool.py` — `NATIVE_TOOL_TYPES` membership, wire/beta-header consistency, factory methods with parameter validation, `marker_to_wire` conversion incl. unknown-marker fallback, `required_beta_headers` collection, `build_create_kwargs` integration of native tools / caching / strict tools / `disable_parallel_tool_use`, end-to-end `BaseAnthropic.call` flows.
- Updated `tests/test_stream_events.py` — `_process_raw_events` separation of `server_tool_calls`, `stop_reason` + cache tokens on `COMPLETE`, `message_start` usage handling, `thinking_delta`.
- Updated `tests/test_messages_normalize.py` — 4 new tests for `server_tool_use`, per-tool `*_tool_result` (incl. list content), missing-id handling, and `_coerce_tool_result_content` Pydantic/JSON/string fallbacks.
- Updated `tests/test_converter.py` — `test_spec_looks_native_registry_membership` reflects the populated `NATIVE_TOOL_TYPES`.
- Ruff clean.

### Added — provider-agnostic native-tool observability — wired across the agent loop

- **Generic `build_server_tool_call_records(server_tool_calls, *, round)`** factory in `nucleusiq.agents.observability.record_builders` — accepts mappings, Pydantic models, or any object exposing `id` / `name` / `input` / `result` attributes (Anthropic `ServerToolCall`, OpenAI hosted-tool blocks, Gemini `code_execution` blocks). Empty / `None` inputs return `[]` so the call-sites can call it unconditionally.
- **`base_mode.py` agent loop** now centrally detects the provider for every LLM call via `get_provider_from_llm(agent.llm)` and threads `provider="anthropic" | "openai" | "google" | "ollama" | "groq"` into both `build_llm_call_record` and `build_llm_call_record_from_stream`. The non-stream path, the streaming `complete_event` path, and the autonomous synthesis stream path all also surface any `server_tool_calls` as `ToolCallRecord(executed_by="provider")` automatically. **One change, six providers benefit.**
- **3 new unit tests** in `tests/unit/test_execution_tracer.py` covering the new factory: dict + Pydantic-style item shapes and the empty-input fast path.

### Added — `nucleusiq` 0.7.12

- `src/nucleusiq/pyproject.toml` `version` → `0.7.12`; `src/nucleusiq/core/__init__.py` `__version__` → `"0.7.12"`.

### Added — `nucleusiq-ollama` 0.2.0 Stable

- Promoted from **Alpha (`0.1.0a1`) → Stable (`0.2.0`)**; classifier flipped to `Development Status :: 5 - Production/Stable`; floor bumped to `nucleusiq>=0.7.12`.
- **Vision (image messages) in `_shared/wire.py`** — `sanitize_messages` now splits OpenAI-style multimodal content lists into Ollama's chat-message shape: text parts become the `content` string; `image_url` parts whose URL is a `data:image/*;base64,…` URL are decoded into the `images` field; raw `{"type": "image", "data": "..."}` blocks pass through; HTTP(S) image URLs are skipped with a warning (no implicit download).
- New helpers `_split_data_url` and `_extract_text_and_images` cover the data-URL split logic.
- `LLMCallRecord.provider="ollama"` is now populated automatically via the central `base_mode.py` hook.
- **+14 new unit tests** in `tests/test_wire.py` — covers data URL decoding, OpenAI-part splitting, HTTP-URL warning, raw `image` blocks, system + user vision messages, merging with pre-existing `images`, garbage parts. **98 tests passing, coverage 99.85%** (gate ≥ 95%).

### Added — `nucleusiq-groq` 0.1.0 Stable

- Promoted from **Beta (`0.1.0b1`) → Stable (`0.1.0`)**; classifier flipped to `Development Status :: 5 - Production/Stable`; floor bumped to `nucleusiq>=0.7.12`.
- New **`ServerToolCall`** model in `_shared/response_models.py`; `GroqLLMResponse.server_tool_calls: list[ServerToolCall]` field added.
- `nb_groq/chat.py` `_extract_server_tool_calls` reads Groq's hosted/compound-tool surface (`message.executed_tools`) and emits server-tool records ready for the central agent-loop hook; handles dict, Pydantic-`model_dump`, and SimpleNamespace-style SDK items; auto-parses JSON `arguments` strings.
- `LLMCallRecord.provider="groq"` is now populated automatically.
- **+2 new unit tests** in `tests/test_chat.py`. **79 tests passing, coverage 92.51%** (gate ≥ 90%).

### Added — `nucleusiq-openai` 0.7.0

- Version bumped `0.6.4` → `0.7.0`; floor bumped to `nucleusiq>=0.7.12`.
- New **`ServerToolCall`** model in `_shared/response_models.py`; `_LLMResponse.server_tool_calls: list[ServerToolCall]` field added.
- **`normalize_responses_output`** in `nb_openai/response_normalizer.py` now also detects Responses-API output items of type `web_search_call`, `code_interpreter_call`, `file_search_call`, `computer_use_call`, `image_generation_call` and surfaces them as `ServerToolCall` records (with `name` normalised by stripping the `_call` suffix — e.g. `web_search`). Existing `native_outputs` back-compat preserved.
- `LLMCallRecord.provider="openai"` is now populated automatically.
- **+2 new unit tests** in `tests/test_tool_conversion.py`. **232 tests passing**.

### Added — `nucleusiq-gemini` 0.3.0

- Version bumped `0.2.6` → `0.3.0`; floor bumped to `nucleusiq>=0.7.12`.
- New **`ServerToolCall`** model in `_shared/response_models.py`; `GeminiLLMResponse.server_tool_calls: list[ServerToolCall]` field added.
- **`normalize_response`** / `_normalize_candidate` in `nb_gemini/response_normalizer.py` now pair `executable_code` + `code_execution_result` parts into a single `code_execution` server-tool record (orphan `executable_code` without a paired result is still recorded as an unfinished server invocation); `grounding_metadata` on a candidate becomes a `google_search` server-tool record with the metadata dump as the result payload.
- `LLMCallRecord.provider="google"` is now populated automatically (consistent with the existing `get_provider_from_llm` mapping).
- **292 unit tests passing** (one live integration test unrelated to these changes returned empty content from the API).

### Added — `nucleusiq-mcp` 0.1.0 Stable

- Promoted from **Beta (`0.1.0b1`) → Stable (`0.1.0`)**; classifier flipped to `Development Status :: 5 - Production/Stable`; floor bumped to `nucleusiq>=0.7.12`; core `nucleusiq[mcp]` extras alias bumped to `>=0.1.0`. **No API changes from `0.1.0b1`.** 248 tests passing.

### Deferred — explicitly out of v0.7.12 scope

- **OpenAI Web Search GA parity** — observability emission ships in v0.7.12; full GA parity (Responses + Chat path equivalence, capability surface) becomes a `v0.7.13+` milestone.
- **Devil's advocate Anthropic notebook** — deferred to a follow-up showcase release.
- **Groq Phase B** (Responses API + hosted tools + remote MCP) — targets `nucleusiq-groq 0.2.x`.
- **Anthropic Phase C** (Memory / `computer_use` / `bash`) — targets `nucleusiq-anthropic 0.3.x`.
- **Ollama embeddings** — beyond chat + vision.

---

## [0.7.11](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.11) — 2026-05-25

### Added

#### MCP tool adapter — `nucleusiq-mcp` **0.1.0b1** (beta)

First public release of the **Model Context Protocol** tool adapter — connect any MCP-compliant server (stdio, Streamable HTTP, or legacy SSE) to a NucleusIQ Agent in **one line**:

```python
agent = Agent(
    tools=[
        MCPTool("npx -y @modelcontextprotocol/server-github"),     # stdio, auto-detected
        MCPTool("https://mcp.slack.com/api", auth="xoxb-..."),    # Streamable HTTP + bearer
        MCPTool(github_oauth_url, auth=OAuthAuth(...)),            # OAuth 2.1 + PKCE
        my_custom_calculator_tool,                                 # existing BaseTool still works
    ],
)
```

Layout: `src/providers/tools/mcp/`, import **`nucleusiq_mcp`**, PyPI **`nucleusiq-mcp`**. Requires **`nucleusiq>=0.7.11`** and **`mcp>=1.27,<2`** (official Python SDK). Install convenience: **`pip install "nucleusiq[mcp]"`**.

**Maturity — why Beta, not Alpha:**

- Feature-complete for the published scope (Phases 0-3 in [MCP_INTEGRATION_DESIGN.md](docs/design/MCP_INTEGRATION_DESIGN.md)).
- **98.68% line coverage** across 11 modules; **235 unit tests** (mocked SDK) + **13 live integration tests** (10 transport tests against the official `@modelcontextprotocol/server-everything` reference server + 3 end-to-end Anthropic Claude tests proving the full Agent → MCPTool → real MCP server → tracer loop).
- Ruff 0.15.14 + Pyrefly 1.0 (GA) clean.
- All bugs found during 3 review passes (rc3/rc5/rc6) have been fixed with regression tests.
- Public API (`MCPTool`, `MCPSession`, `MCPBoundTool`, `MCPServerConfig`, `MCPAuth` family, `MCPError` hierarchy) is considered stable; only Phase 4 additions and feedback-driven tweaks are expected before 1.0.

**Public API (`from nucleusiq_mcp import ...`):**

- **`MCPTool(target, *, name=…, auth=…, transport=…, include_tools=…, exclude_tools=…, tool_filter=…, rename=…, prefix=…, on_collision=…, on_connect_failure=…, health_check=…)`** — user-facing factory. Implements the new `nucleusiq.tools.ExpandableTool` protocol so the Agent expands it into one or more `MCPBoundTool` instances during `initialize()`.
- **`MCPSession`** — `AsyncExitStack`-managed lifecycle wrapper around the SDK's `ClientSession`. `_state_lock` makes `connect()`/`disconnect()` idempotent; `_call_lock` serialises RPCs against a single SDK session (the SDK is not safe for concurrent calls). Usable directly (`async with MCPSession(cfg) as s:`) for read-only / discovery flows.
- **`MCPBoundTool`** — `BaseTool` subclass that adapts one remote MCP tool. `execute` translates `BaseTool` ↔ `MCPSession.call_tool`, raises `MCPToolError` on `isError=True`, sets `source = "mcp://server=<name> (path=A)"` so the tracer can attribute calls.
- **`MCPServerConfig` + `MCPTransport`** — typed config with auto-detect: HTTP/HTTPS → `STREAMABLE_HTTP`; anything else → `STDIO`. Override with `transport=MCPTransport.SSE` for the legacy SSE endpoint shape.
- **Auth** — `BearerAuth`, `EnvAuth("MY_TOKEN")`, `OAuthAuth(...)` (wraps the SDK's `OAuthClientProvider` with PKCE + dynamic client registration), `CustomHeadersAuth({...})`. String auth shorthand auto-coerces to `BearerAuth`; dict auth shorthand auto-coerces to `CustomHeadersAuth`. All auth classes use `__slots__` + redact secrets from `__repr__`.
- **Decorator API** — `@mcp_tool_filter(name="...", description="...")` for declarative tool filters (metadata surfaced as `filter_name` / `filter_description`); `MCPToolset.all_of(...) / .any_of(...)` composes filters.
- **Errors** — `MCPError` → `MCPConnectionError` / `MCPAuthError` / `MCPTimeoutError` / `MCPProtocolError` / `MCPToolError(ToolExecutionError)`. `MCPToolError` is a `ToolExecutionError` subclass so existing agent error-handling Just Works™.

**Phase 2 production hardening (added in this release):**

- **`on_connect_failure="raise" | "skip"`** — `"skip"` lets a multi-server agent survive an unreachable server (logs a warning, returns `[]` from `expand()`, tears down the half-open session). Default stays `"raise"` (fail-fast).
- **`health_check=True` (default)** — `connect()` issues a single `list_tools` round-trip after the protocol handshake so transport-opens that won't actually speak MCP (wrong URL, wrong content-type, proxy stripping headers, expired token) fail fast. Set `False` for fastest possible startup when you trust the transport.
- **`MCPTool.ping()`** — public health-check returning `bool`; never raises. Useful for liveness probes and dashboards.

**Observability:**

- Every MCP-backed tool call lands in `AgentResult.tool_calls` with `source='mcp://server=<name> (path=A)'`. Downstream consumers (telemetry, billing, audit, dashboards) can filter on the `mcp://` prefix to count MCP usage. Path label (`A`) distinguishes the client-side adapter path from the future provider-hosted path (`B`).

**Tests + coverage:**

| Suite | Count | Notes |
|-------|-------|-------|
| Unit (`tests/unit/`) | **235** | Mocked SDK via `FakeMCPSession`; covers auth, config, session lifecycle, schema adapter, bound-tool execution, MCPTool facade (construction, filtering, rename, collision policy, graceful degradation, health checks, ping), retry policy, decorators, error hierarchy. |
| Live integration (`tests/integration/`, `-m integration`) | **13** | 10 transport tests against `@modelcontextprotocol/server-everything` (stdio, Streamable HTTP, SSE × {session lifecycle, call_tool, facade}); 3 end-to-end tests with `nucleusiq-anthropic` Claude Haiku 4.5 proving the full agent loop, source-label propagation, and tool selection. |
| Coverage | **98.68%** | Per-module: auth/bound_tool/decorators/exceptions/models/retry/schema_adapter 100%; config 96%; mcp_tool 96%; session 99%. |

**Examples (`src/providers/tools/mcp/examples/`):**

- `01_basic_stdio.py` — stdio one-liner against the reference server.
- `02_http_with_auth.py` — Streamable HTTP + `BearerAuth` / `EnvAuth` / `CustomHeadersAuth`.
- `03_oauth_flow.py` — full `OAuthAuth` (PKCE, dynamic client registration).
- `04_multi_server.py` — multiple servers + name-collision policies.
- `05_filter_and_rename.py` — `include_tools` / `exclude_tools` / `rename` / `prefix`.
- `06_error_handling.py` — `MCPConnectionError` / `MCPToolError` / `on_connect_failure="skip"`.
- `07_decorator_filters.py` — `@mcp_tool_filter`, `MCPToolset.all_of` / `.any_of`.
- `08_full_agent_with_llm.py` — end-to-end Agent + LLM + MCP server.

**Documentation:**

- `notebooks/agents/mcp_tools_showcase.ipynb` — executable end-to-end walkthrough: discovery → real Claude tool call → graceful degradation → transport auto-detection. Notebook runs the reference server in Streamable HTTP mode so it's portable across macOS, Linux, and Windows + IPython (anyio + IPython's wrapped stdin can't open stdio subprocesses; CLI/script use is unaffected).
- `src/providers/tools/mcp/README.md` + `examples/README.md` — install, quick-start, transport matrix, auth matrix.
- `docs/design/MCP_INTEGRATION_DESIGN.md` — v0.8.0-rc6: full architecture, SOLID rationale, comparison with LangChain `langchain-mcp-adapters` and CrewAI's `MCPServerAdapter`, security threat model, Phase 4 roadmap, live implementation-status checklist.

**Bugs found & fixed during 3 review passes (regression tests for each):**

1. **Orphan-task leak** in `Agent.initialize()` — `asyncio.gather` (default) propagated the first connect failure while leaving sibling tasks unattended. Fixed with `return_exceptions=True` + explicit re-raise. (`test_initialize_parallel_failure_no_orphans`)
2. **Cleanup skipped on `BaseException`** — outer `except Exception` silently let `KeyboardInterrupt` / `CancelledError` bypass `_cleanup_expandable_tools()`. Widened to `except BaseException` with an inner guard.
3. **`ToolCallRecord.source` defined but never populated** — added the wiring (`build_tool_call_record(..., source=...)` + `base_mode.call_tool` look-up + `MCPBoundTool.source` attribute) so MCP tool calls actually show up labelled in telemetry.
4. **`on_connect_failure="skip"` silently bypassed for HTTP / SSE** — the MCP SDK uses `anyio.create_task_group()` internally; when a sub-task fails (e.g. `httpx.ConnectError`), the group re-raises it as `asyncio.CancelledError` — a `BaseException`. Previous `except Exception` clause didn't catch it, so the skip policy worked for stdio but not HTTP. Widened to `except BaseException` (carving out `KeyboardInterrupt`/`SystemExit`). (`test_skip_catches_cancelled_error`, `test_skip_does_not_swallow_keyboard_interrupt`)
5. **`OAuthAuth` docstring referenced a non-existent `with_defaults` helper** — replaced with a real wiring example using `OAuthClientProvider`.
6. **`session._dump` return-type unsoundness** — surfaced by Pyrefly 1.0 GA. `model_dump()` is typed `Any`, so returning it from a function annotated `-> dict[str, Any] | None` was unsound. Now narrows with `isinstance(..., dict)`.
7. **`anyio` listed as a direct dep but never imported** — removed. It's a transitive dep of the `mcp` SDK; we use stdlib `asyncio` everywhere (`AsyncExitStack`, `Lock`, `gather`, `wait_for`). Smaller dep graph, clearer intent.

**Stability commitments (Beta → 1.0):**

- **Public API will not break** without a deprecation window: `MCPTool`, `MCPSession`, `MCPBoundTool`, `MCPServerConfig`, `MCPTransport`, the four `*Auth` classes, the `MCPError` hierarchy.
- **`mcp` SDK pin (`>=1.27,<2`)** will be widened on a 2.x major bump only after we verify the new surface.
- **`ToolCallRecord.source` format** (`mcp://server=<name> (path=<A|B>)`) is stable; the `path` segment will grow values (`B` for provider-hosted MCP) but `A` will not change semantics.
- Phase 4 additions (Tool approvals, Progress, MCP Prompts, MCP Resources, Sampling, Elicitation, Resumability, stdio auto-respawn) are purely additive — no breaking changes expected.

**Limitations / known issues (deferred to Phase 4):**

- No MCP **Prompts** support yet — the SDK supports it but we surface only Tools today.
- No MCP **Resources** support yet.
- No **tool-approval policy** (`ask | always_allow | always_deny | callback`).
- No **progress notification** forwarding (long-running tools still complete, but mid-execution progress is dropped).
- No **streaming** tool results (each `call_tool` is request/response).
- No **stdio auto-respawn** on crash — caller must reconnect manually.
- **Windows + IPython kernel** — anyio's `open_process` cannot reach into the kernel's wrapped stdin, so MCP stdio servers don't open from IPython notebooks. Use Streamable HTTP mode in notebooks on Windows; stdio works fine in CLI scripts on every platform.

#### Core — `nucleusiq` **0.7.11** (patch)

A small, additive patch release that unlocks the MCP adapter. **No breaking changes**: existing agents and tools work exactly as before — `ExpandableTool` is opt-in, `ToolCallRecord.source` is `None` by default, and the `Agent.initialize` parallel-connect path is only exercised when at least one `ExpandableTool` is on the agent.

**New public surface:**

- **`nucleusiq.tools.protocols.ExpandableTool`** — `@runtime_checkable` Protocol defining `connect()` / `expand(existing_names: set[str]) -> list[BaseTool]` / `disconnect()`. Re-exported as `from nucleusiq.tools import ExpandableTool`. Tool factories (MCP adapters, future LangChain/CrewAI bridges, agent-as-tool wrappers) implement this contract; the core knows nothing MCP-specific.
- **`ToolCallRecord.source: str | None`** — optional opaque origin label on every tool call recorded in `AgentResult.tool_calls`. The framework reads `getattr(tool, "source", None)` automatically — no tool needs to import anything; just expose an attribute. (`source` field is `None` for hand-written `BaseTool`s — no behaviour change for existing code.)
- **`nucleusiq[mcp]`** extras — installs `nucleusiq-mcp>=0.1.0b1`. Lets users write `pip install "nucleusiq[mcp]"` instead of remembering the separate package name.

**Internal robustness:**

- **`Agent.initialize()`** — Phase A (split direct tools vs `ExpandableTool` adapters) / Phase B (parallel `connect()` via `asyncio.gather(return_exceptions=True)`) / Phase C (sequential `expand()` with collision detection) / Phase D (per-tool `initialize()`). The `return_exceptions=True` change prevents orphaned in-flight `connect()` tasks when a sibling adapter fails fast — verified by `test_initialize_parallel_failure_no_orphans`.
- **Rollback path catches `BaseException`** — so `_cleanup_expandable_tools()` still runs on `KeyboardInterrupt` / `asyncio.CancelledError`. Avoids leaking stdio subprocesses or OAuth-locked HTTP sessions on abnormal termination.
- **`_cleanup_expandable_tools()`** — parallel `disconnect()` via `asyncio.gather(return_exceptions=True)`. Runs from both the init-failure rollback and `Agent.__aexit__`.
- **`base_mode.call_tool`** — reads `getattr(tool, "source", None)` and forwards it to `build_tool_call_record(..., source=...)`. Generic — no MCP-specific code in core.
- **`build_tool_call_record`** — new `source: str | None = None` parameter.

**Tooling:**

- Bumped dev lint pins: **ruff ≥ 0.15.14** (was `≥ 0.4`), **pyrefly ≥ 1.0** (was `≥ 0.59`; Meta's type-checker reached GA on 2026-05-12, switching from "Beta" PyPI classifier to "Production/Stable").
- Core test suite: **1323 passed, 2 skipped** (no regressions from the patch).

**Tests added:**

- `tests/tools/unit/test_expandable_tool_protocol.py` — 20 tests covering the Protocol, parallel-connect ordering, partial-failure rollback, collision detection.
- `tests/agents/unit/test_agent_tracer_integration.py` — Agent + tracer + source-field integration test (3 tests).
- `tests/unit/test_execution_tracer.py` — `build_tool_call_record` accepts `source` (existing file extended).

#### Anthropic provider — `nucleusiq-anthropic` **0.1.0a1** (alpha)

Installable Claude provider (**Messages API**) at `src/providers/llms/anthropic`, import **`nucleusiq_anthropic`**, PyPI **`nucleusiq-anthropic`**. Requires **`nucleusiq>=0.7.10`**, **`anthropic>=0.40,<1`**.

- **`BaseAnthropic`** — **`call()`**, **`call_stream()`**; **`AnthropicLLMParams`** (beta headers, **`top_k`**); tool specs via **`to_anthropic_tool_definition`**; merges **`LLMParams`** from agents. **`response_format`** maps to Messages **`output_config.format`** (**JSON Schema** structured outputs) and returns validated Pydantic/dataclass instances when a schema type is supplied (drops structured output when **tools** are present; **`call_stream`** ignores **`response_format`** with a warning). Omits **`top_p`** when **`temperature`** is set so newer Claude models avoid **400** mutual-exclusion errors; **`build_create_kwargs`** drops **`top_p`** if both still appear after extras.
- **`nucleusiq_anthropic.structured_output`** — **`build_anthropic_output_config`**, **`parse_anthropic_response`** (parity with **`nucleusiq_openai` / `nucleusiq_gemini`** layouts); re-exported from **`nucleusiq_anthropic`**.
- **Errors / retries** — SDK exceptions mapped to **`nucleusiq.llms.errors`** (auth, rate limit, model not found, invalid request, content filter, context length, server/connection); **`call_with_retry`** with **`Retry-After`** + exponential backoff (shared **`retry_policy`**).
- **Tests** — **100+** unit tests; **`--cov-fail-under=95`** in CI; line coverage **~97%** on `nucleusiq_anthropic` locally.
- **Examples** — `examples/agents/` (**01**–**09**): DIRECT / STANDARD / AUTONOMOUS, tools, streaming, execution modes, raw LLM stream, offline tool-schema dump, **Models API** list helper for **`ANTHROPIC_MODEL`**. **`examples/output_parsers/anthropic_native_structured_example.py`** — Agent + native structured JSON (OpenAI **`output_parsers`** layout parity). See **`docs/design/ANTHROPIC_PROVIDER.md`**, **`examples/README.md`**.

#### CI / monorepo

- **`test-anthropic`**, **`import-check`** (**`build_anthropic_output_config`**, **`parse_anthropic_response`** + core exports), **`type-check`** (Pyrefly), **`build`** (**`nucleusiq-anthropic`** wheel), **`test-uv`** (Anthropic path), **`publish.yml`** — **`nucleusiq-anthropic`** included alongside other Hatch providers.

---

## [0.7.10](https://github.com/nucleusbox/NucleusIQ/releases/tag/v0.7.10) — 2026-05-09

### Security

- **`urllib3`** (transitive **pip** / **`uv.lock`** graphs) — monorepo lockfiles refreshed so resolved **`urllib3`** is **≥2.7.0**, mitigating **[GHSA-mf9v-mfxr-j63j]** (streaming decompression) and **[GHSA-qccp-gfcp-xxvc]** (sensitive headers on proxied low-level redirects) for versions **before 2.7.0**.
- **`nucleusiq` default install** — **`requests`** and a direct **`urllib3`** pin removed from **`[project.dependencies]`** (the framework does not import **`requests`**). Optional **`nucleusiq[http]`** installs **`requests>=2.33.0,<3.0`** and **`urllib3>=2.7.0`** for notebooks and apps that still need those stacks. **`uv.lock`** for **`nucleusiq`** may still list **`requests` / `urllib3`** for the **dev** graph (e.g. **`nucleusiq-openai`** in **test** deps).

### Added

#### Core optional extras

- **`nucleusiq[http]`** — optional **`requests`** and **`urllib3>=2.7.0`** (see **Security** above).

#### Ollama provider — `nucleusiq-ollama` **0.1.0a1** (alpha)

New installable inference provider for **[Ollama](https://ollama.com/)** (local daemon or hosted API) using the official **`ollama`** Python SDK (**no LangChain**). Layout: `src/providers/inference/ollama/`, import **`nucleusiq_ollama`**, PyPI **`nucleusiq-ollama`**. Requires **`nucleusiq>=0.7.10`**, **`ollama>=0.5.0,<1.0`**.

- **`BaseOllama`** — **`call()`**, **`call_stream()`** → **`StreamEvent`**; wire, retries, **`OllamaLLMParams`** (**`think`**, **`keep_alive`**), structured **`format`**, function tools, env **`OLLAMA_HOST`** / **`OLLAMA_API_KEY`**.
- **Tests** — **≥95%** gate; **100%** line coverage on package in CI; **`integration`** marker for optional live daemon.
- **Examples** — `examples/agents/` (**`00`–`03`** capability matrix: chat, stream, structured, thinking × **DIRECT / STANDARD / AUTONOMOUS**). **`docs/design/OLLAMA_PROVIDER.md`**, provider **`README.md`**.
- **`scripts/verify_core_package_layout.py`** — **`nucleusiq-ollama`** in Hatch **`HATCH_PROVIDERS`**.

### Changed

#### Core

- **`nucleusiq.core.__version__`** — **0.7.10** (aligned with **`pyproject.toml`**).
- **`nucleusiq.agents.structured_output.resolver`** — **`get_provider_from_llm`** returns **`"ollama"`** / **`"groq"`** for correct **`OutputSchema.for_provider()`** payloads with **Agent** structured output; removed stale **`NATIVE_SUPPORT`** table; **`supports_native_output()`** now uses **provider-aware** rules (trust **Groq** / **Ollama** adapters when provider is known; coarse GPT / Claude / Gemini name shapes when unknown); **`_auto_select_mode`** documents that **`OutputMode.AUTO`** maps **model_name set → NATIVE** (no prompt of the prefix table).

#### Monorepo / tooling

- **`uv.lock`** files refreshed across **`nucleusiq`** and provider packages; redundant **`[tool.uv] constraint-dependencies`** for **`urllib3`** dropped where resolution already yields **`urllib3` ≥2.7.0** (install-time dependencies remain **`pyproject.toml`** **`Requires-Dist`**, not lockfiles).

### Packages

| Package | Version | Note |
| --- | --- | --- |
| `nucleusiq` | **0.7.10** | **Security:** trimmed default deps + patched **`urllib3`** in locks; optional **`[http]`**; structured-output resolver; pairs with Ollama alpha |
| `nucleusiq-ollama` | **0.1.0a1** α | Alpha; **`nucleusiq>=0.7.10`** |

Existing **`nucleusiq-openai`**, **`nucleusiq-gemini`**, **`nucleusiq-groq`** wheels remain on **`nucleusiq>=0.7.9`** unless republished with a raised floor; **`nucleusiq-ollama`** requires **0.7.10** for the resolver fixes above.

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

