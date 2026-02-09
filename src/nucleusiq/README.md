# NucleusIQ

**Core package** for the NucleusIQ AI agent framework.

Includes agents, prompts, tools, and utilities.

See the main [README](https://github.com/nucleusbox/NucleusIQ) for full documentation.

## Install

```bash
pip install nucleusiq
```

## Quick Start

```python
from nucleusiq.agents import Agent
from nucleusiq.llms import MockLLM

agent = Agent(name="test", role="assistant", objective="help", llm=MockLLM())
result = await agent.execute({"id": "1", "objective": "Hello!"})
```
