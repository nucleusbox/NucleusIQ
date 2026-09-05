# NucleusIQ

[![PyPI version](https://img.shields.io/pypi/v/nucleusiq?color=brightgreen)](https://pypi.org/project/nucleusiq/)
[![PyPI downloads](https://img.shields.io/pypi/dm/nucleusiq?label=downloads%2Fmonth)](https://pypistats.org/packages/nucleusiq)
[![Python versions](https://img.shields.io/pypi/pyversions/nucleusiq)](https://pypi.org/project/nucleusiq/)

**Core package** for the NucleusIQ AI agent framework.

Includes agents, prompts, tools, memory, plugins, execution modes, and tracing.

Install a provider separately: `nucleusiq-openai`, `nucleusiq-gemini`, `nucleusiq-anthropic`, `nucleusiq-groq`, `nucleusiq-ollama`, `nucleusiq-openai-compatible` (vLLM / SGLang / TGI / llama.cpp / any Chat Completions server), or `nucleusiq-mcp`.

See the main [README](https://github.com/nucleusbox/NucleusIQ) for full documentation.

## Install

```bash
pip install nucleusiq
pip install nucleusiq-openai                    # OpenAI cloud
pip install nucleusiq-openai-compatible         # self-hosted / OpenAI-compatible
```

## Quick Start

```python
from nucleusiq.agents import Agent
from nucleusiq.llms import MockLLM

agent = Agent(name="test", role="assistant", objective="help", llm=MockLLM())
result = await agent.execute({"id": "1", "objective": "Hello!"})
```
