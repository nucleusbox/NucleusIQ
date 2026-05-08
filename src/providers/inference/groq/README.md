# nucleusiq-groq

**Groq inference provider** for [NucleusIQ](https://github.com/nucleusbox/NucleusIQ): Chat Completions via Groq’s **OpenAI-compatible** API, using Groq’s official **`groq`** Python SDK (`AsyncGroq` / `Groq`).

**Status:** **0.1.0a1** (alpha / pre-release). Requires **`nucleusiq>=0.7.8`**.

Design, phased roadmap (Phase A vs B), and API caveats: [`docs/design/GROQ_PROVIDER.md`](../../../../docs/design/GROQ_PROVIDER.md) (repo root).

---

## Install

**PyPI (when published):**

```bash
pip install nucleusiq nucleusiq-groq
```

**Monorepo (editable):**

```bash
cd src/providers/inference/groq
uv sync --group dev
```

Core is pulled via `[tool.uv.sources]` as an editable path dependency; for `pip`-only local installs, install `nucleusiq` from `src/nucleusiq` first, then this package.

---

## Configuration

| Variable | Purpose |
|----------|---------|
| **`GROQ_API_KEY`** | Required. Groq API key. |
| **`GROQ_MODEL`** | Optional. Default for chat/tool examples: `llama-3.3-70b-versatile`. |
| **`GROQ_MODEL_STRUCTURED`** | Optional. Model for **`json_schema`** structured output (example `05` defaults to `openai/gpt-oss-20b` if unset). |

Unsupported Chat Completions fields (e.g. `logit_bias`, `messages[].name`) are stripped or rejected at the wire layer; see the design doc.

---

## Usage

```python
import asyncio
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq_groq import BaseGroq, GroqLLMParams

async def main() -> None:
    llm = BaseGroq(model_name="llama-3.3-70b-versatile", async_mode=True)
    agent = Agent(
        name="demo",
        prompt=ZeroShotPrompt(),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            llm_params=GroqLLMParams(temperature=0.2),
        ),
    )
    await agent.initialize()
    result = await agent.execute(Task(id="1", objective="Capital of France in one short phrase."))
    print(result.output)

asyncio.run(main())
```

---

## Phase A (this release)

- **`BaseGroq`** — `call` / `call_stream`, tool calling, structured output (`response_format` / Pydantic).
- **`GroqLLMParams`** — typed, `extra="forbid"`; merges into provider calls.
- **Local function tools** — OpenAI-style tool JSON; assistant `tool_calls` normalized for Groq before each request.
- **Retries** — rate-limit and transient errors with exponential backoff; errors mapped to NucleusIQ `LLMError` types.

**Not in Phase A:** Groq **Responses API**, **built-in/hosted tools**, **remote MCP** (see design doc Phase B).

---

## Examples

Runnable agents (real API): [`examples/README.md`](examples/README.md).

```bash
cd src/providers/inference/groq
uv run python examples/agents/01_groq_direct.py
```

---

## Development

From this directory:

```bash
uv sync --group dev
uvx ruff check nucleusiq_groq tests examples
uvx pyrefly check
uv run pytest
```

---

## License

MIT — same as the NucleusIQ monorepo.
