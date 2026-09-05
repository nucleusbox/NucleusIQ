# nucleusiq-anthropic

[![PyPI version](https://img.shields.io/pypi/v/nucleusiq-anthropic?color=brightgreen)](https://pypi.org/project/nucleusiq-anthropic/)
[![PyPI downloads](https://img.shields.io/pypi/dm/nucleusiq-anthropic?label=downloads%2Fmonth)](https://pypistats.org/packages/nucleusiq-anthropic)
[![Python versions](https://img.shields.io/pypi/pyversions/nucleusiq-anthropic)](https://pypi.org/project/nucleusiq-anthropic/)

**Anthropic Claude** provider for [NucleusIQ](https://github.com/nucleusbox/NucleusIQ): native **Messages API** via the official **`anthropic`** Python SDK (`AsyncAnthropic` / `Anthropic`), with message wiring, custom tools, retries, and streaming mapped to **`StreamEvent`**.

## Release status

| | |
|--|--|
| **PyPI package** | **`nucleusiq-anthropic`** |
| **This line** | **`0.2.1`** — **Development Status :: 5 - Production/Stable** (first stable line is `0.2.0`; `0.2.1` is a maintenance bump: declared `PROVIDER_NAME` + dependency completeness). |
| **What’s in 0.2.0 (Phase B feature-complete)** | `BaseAnthropic` (`call`, `call_stream`), chat/tool translation (`wire`), **`structured_output`**, error mapping + retries, stream adapter, **`AnthropicTool` factory** for native server tools (`web_search`, `web_fetch`, `code_execution`) with auto-injected `anthropic-beta` headers, **prompt caching** (`cache_tools`, `cache_system`), **extended thinking** (`thinking="low"|"medium"|"high"|"max"` or dict), **`strict_tools`** + **`disable_parallel_tool_use`**, **server-tool observability** with full handling of `server_tool_use` + per-tool `*_tool_result` blocks (`code_execution_tool_result`, `web_search_tool_result`, …) → `AnthropicLLMResponse.server_tool_calls` + `ToolCallRecord(executed_by="provider")` automatically emitted by the core agent loop, full `LLMCallRecord` enrichment (`stop_reason`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `request_id`, `organization_id`, `provider="anthropic"`). **151 unit tests + 6 live integration tests, 95.91% coverage** (gate ≥ 95%). |
| **Phase B examples** | `examples/agents/10_anthropic_native_tools.py` (web_search + code_execution), `11_anthropic_prompt_caching.py` (`cache_system=True`, prints per-`LLMCallRecord` cache tokens), `12_anthropic_extended_thinking.py` (`thinking="low"` / `"medium"`). All three verified end-to-end against `claude-sonnet-4-5-20250929`; override the model with `ANTHROPIC_PHASE_B_MODEL=<id>`. |
| **Live integration tests** | `tests/integration/test_anthropic_phase_b_live.py` — 6 tests (`-m integration`, requires `ANTHROPIC_API_KEY`): web_search, code_execution, prompt caching, extended thinking ×2 (low/medium), `disable_parallel_tool_use`. Skips cleanly if the chosen model is not available on the active API key. Excluded from default + CI runs. |
| **Deferred** | Phase C (Memory / `computer_use` / `bash`) — targets `nucleusiq-anthropic 0.3.x`. |

User guide and roadmap: [Anthropic provider guide](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/anthropic-provider/).

---

## Install

**PyPI (when the wheel is published):**

```bash
pip install nucleusiq nucleusiq-anthropic
```

**Monorepo (editable, recommended for development):**

```bash
cd src/providers/llms/anthropic
# unit tests + lint + python-dotenv for example scripts
uv sync --group full
```

Use `uv sync --group dev` if you only need pytest/ruff. Example scripts auto-load a repo-root `.env` when **`python-dotenv`** is installed (`full`).

For `pip` only, install **`nucleusiq`** from `src/nucleusiq` first, then this package (`pip install -e .`). Optionally `pip install python-dotenv` for `.env` loading in examples.

---

## Configuration

| Variable / argument | Purpose |
|-------------------|--------|
| **`ANTHROPIC_API_KEY`** | Required for live calls unless you pass `api_key="..."` to `BaseAnthropic`. |
| **`ANTHROPIC_MODEL`** | Optional. Default model ID for examples and `BaseAnthropic` (default: `claude-3-5-sonnet-20241022`). Override if your workspace uses Claude 4 / Haiku SKU strings (see Anthropic docs). |

Provider-specific sampling (temperature, `max_output_tokens`, etc.) should be passed with NucleusIQ **`LLMParams`** on **`AgentConfig.llm_params`** (merged into every `llm.call`). Use **`AnthropicLLMParams`** on the **`BaseAnthropic(..., llm_params=...)`** constructor for **beta headers**, **top_k**, and extra headers.

---

## Quick start (SDK only)

```python
import asyncio

from nucleusiq_anthropic import BaseAnthropic


async def main() -> None:
    llm = BaseAnthropic(model_name="claude-3-5-sonnet-20241022", async_mode=True)
    resp = await llm.call(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Say hello in exactly five words."}],
        max_output_tokens=128,
        temperature=0.3,
    )
    print(resp.choices[0].message.content)


asyncio.run(main())
```

---

## Agent examples

**DIRECT / STANDARD / AUTONOMOUS**, **streaming** (`Agent.execute_stream` and raw `BaseAnthropic.call_stream`), **DIRECT + one tool**, a **single-file all-modes** driver, and an **offline** tool-schema helper — similar coverage to `openai/examples/agents` / `groq/examples/agents`, minus OpenAI-hosted native tools until Phase B/C.

Commands and parity table: **[`examples/README.md`](examples/README.md)**.

---

## Import surface

```python
from nucleusiq_anthropic import (
    BaseAnthropic,
    AnthropicLLMParams,
    NATIVE_TOOL_TYPES,
    to_anthropic_tool_definition,
)
```

---

## License

MIT (same as NucleusIQ monorepo unless overridden in package metadata).
