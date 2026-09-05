# nucleusiq-openai

[![PyPI version](https://img.shields.io/pypi/v/nucleusiq-openai?color=brightgreen)](https://pypi.org/project/nucleusiq-openai/)
[![PyPI downloads](https://img.shields.io/pypi/dm/nucleusiq-openai?label=downloads%2Fmonth)](https://pypistats.org/packages/nucleusiq-openai)
[![Python versions](https://img.shields.io/pypi/pyversions/nucleusiq-openai)](https://pypi.org/project/nucleusiq-openai/)

**OpenAI provider** for [NucleusIQ](https://github.com/nucleusbox/NucleusIQ): Chat Completions + Responses API through the official `openai` Python SDK.

## Status

**0.7.1** — stable provider line aligned with **`nucleusiq>=0.7.12`**.

Highlights:

- `BaseOpenAI` for agent execution across `DIRECT`, `STANDARD`, and `AUTONOMOUS` modes.
- Chat Completions + Responses API routing.
- Native OpenAI tools: `web_search`, `code_interpreter`, `file_search`, `computer_use`, `image_generation`.
- Structured output support through the NucleusIQ resolver.
- Native-tool observability: Responses API tool items surface as `server_tool_calls` and are emitted by core tracing as `ToolCallRecord(executed_by="provider")`.
- Responses API usage (`input_tokens` / `output_tokens`) mapped to `prompt_tokens` / `completion_tokens` on both streaming and non-streaming paths.

Self-hosted or third-party OpenAI-compatible servers (vLLM, SGLang, TGI, llama.cpp, OpenRouter, …) belong in [`nucleusiq-openai-compatible`](https://github.com/nucleusbox/NucleusIQ/tree/main/src/providers/inference/openai_compatible) — this package is the OpenAI **cloud** adapter (Responses API, hosted tools, tiktoken).

## Install

```bash
pip install nucleusiq nucleusiq-openai
```

## Quick start

```python
import asyncio

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq_openai import BaseOpenAI


async def main() -> None:
    agent = Agent(
        name="openai-agent",
        prompt=ZeroShotPrompt().configure(
            system="You are a concise assistant. Answer in one short sentence.",
        ),
        llm=BaseOpenAI(model_name="gpt-4o-mini"),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    await agent.initialize()
    result = await agent.execute(
        Task(id="openai-hello", objective="What is the capital of France?"),
    )
    print(result.output)


asyncio.run(main())
```

## See also

- [Root README](https://github.com/nucleusbox/NucleusIQ)
- [Published docs](https://nucleusbox.github.io/nucleusiq-docs/)
- [Changelog](https://github.com/nucleusbox/NucleusIQ/blob/main/CHANGELOG.md)
