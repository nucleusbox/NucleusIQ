# nucleusiq-anthropic

**Anthropic Claude** provider for [NucleusIQ](https://github.com/nucleusbox/NucleusIQ): native **Messages API** via the official **`anthropic`** Python SDK (`AsyncAnthropic` / `Anthropic`), with message wiring, custom tools, retries, and streaming mapped to **`StreamEvent`**.

## Release status

| | |
|--|--|
| **PyPI package** | **`nucleusiq-anthropic`** |
| **This line** | **`0.1.0a1`** — **Development Status :: 3 - Alpha** (pre-stable API; pin versions in production). |
| **What’s in this alpha** | `BaseAnthropic` (`call`, `call_stream`), chat/tool translation (`wire`), **`structured_output`** (JSON Schema -> `output_config.format`, parses assistant JSON into Pydantic/dataclasses), error mapping + retries, stream adapter, `AnthropicLLMParams` (`top_k`, `anthropic_beta`, `extra_headers`), **≥95%** unit-test coverage on `nucleusiq_anthropic`. |
| **Not in this alpha yet** | Rich observability (`LLMCallRecord` fields); full plugin/memory audit in the design doc; **server-side / native Claude tools** registry is empty until Phase B/C. |

Full roadmap: [`docs/design/ANTHROPIC_PROVIDER.md`](../../../../docs/design/ANTHROPIC_PROVIDER.md).

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
