# Anthropic provider examples

Runnable scripts against the **Anthropic API** (`ANTHROPIC_API_KEY`). Layout mirrors **`groq/examples/agents`** and covers the same *shapes* of demos as **`openai/examples/agents`** (modes, tools, streaming) where the Claude SDK supports them.

## Prerequisites

1. Install (from `src/providers/llms/anthropic`):

   ```bash
   uv sync --group full
   ```

   (`full` adds `python-dotenv` for repo-root `.env`; use `uv sync --group dev` for tests-only.)

2. **`ANTHROPIC_API_KEY`** in the environment or **repo-root** `.env`.

3. Optional **`ANTHROPIC_MODEL`** — defaults to **`claude-3-5-sonnet-20241022`** (broad compatibility). Use another model ID if your org exposes Claude 4 / Haiku only.

Use **`LLMParams`** on **`AgentConfig.llm_params`** for sampling. Use **`AnthropicLLMParams`** on **`BaseAnthropic(..., llm_params=...)`** for beta headers / `top_k`.

## Scripts (parity map)

| # | Anthropic script | Rough OpenAI analogue | Purpose |
|---|------------------|-----------------------|---------|
| 01 | `agents/01_anthropic_direct.py` | DIRECT quick chat | **DIRECT**, no tools. |
| 02 | `agents/02_anthropic_direct_with_tool.py` | `openai_quick_start`-style shortcut | **DIRECT** + one local tool hop. |
| 03 | `agents/03_anthropic_standard_tools.py` | tool loop examples | **STANDARD** multi-tool loop. |
| 04 | `agents/04_anthropic_autonomous.py` | `autonomous_complex_tasks` (light) | **AUTONOMOUS** + tools + quality check. |
| 05 | `agents/05_anthropic_stream.py` | streamed agent runs | **`Agent.execute_stream`** (TOKEN → COMPLETE). |
| 06 | `agents/06_anthropic_execution_modes.py` | `execution_modes_example.py` | All **three gears** in one process. |
| 07 | `agents/07_anthropic_llm_stream_raw.py` | low-level provider stream | **`BaseAnthropic.call_stream`** only (no Agent). |
| 08 | `agents/08_anthropic_tool_schema.py` | `openai_tool_example` schema half | Prints **`to_anthropic_tool_definition`** output (no API). |
| 09 | `agents/09_anthropic_list_models.py` | (helper) | Lists **Models API** IDs for your key; set **`ANTHROPIC_MODEL`** to one. Optional **`ANTHROPIC_LIST_MODELS_VERBOSE=1`** prints capability blobs. |
| — | `output_parsers/anthropic_native_structured_example.py` | `openai/.../agent_output_parser_example.py` | **Agent + native JSON Schema** via Messages **`output_config.format`** — needs a Claude SKU with structured-output support. |

If you see **`404 model`** / **`model not_found_error`**, your key/org may not have the default Messages model. Run **09**, copy a printed **`id`**, and export **`ANTHROPIC_MODEL=<id>`**.

**Note:** Anthropic alpha has **empty** **`NATIVE_TOOL_TYPES`** — there is **no** `OpenAITool`-style factory for Claude **server-side** tools yet (`web_search`, etc.). That lands in Phase B/C per `docs/design/ANTHROPIC_PROVIDER.md`.

## Run

From `src/providers/llms/anthropic`:

```bash
uv run python examples/agents/01_anthropic_direct.py
uv run python examples/agents/02_anthropic_direct_with_tool.py
uv run python examples/agents/03_anthropic_standard_tools.py
uv run python examples/agents/04_anthropic_autonomous.py
uv run python examples/agents/05_anthropic_stream.py
uv run python examples/agents/06_anthropic_execution_modes.py
uv run python examples/agents/07_anthropic_llm_stream_raw.py
uv run python examples/agents/08_anthropic_tool_schema.py   # offline
uv run python examples/agents/09_anthropic_list_models.py    # discovers model IDs for your key
uv run python examples/output_parsers/anthropic_native_structured_example.py
```
