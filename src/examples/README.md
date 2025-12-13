# NucleusIQ Examples

This directory contains comprehensive examples demonstrating how to use NucleusIQ.

**Important:** NucleusIQ is an **Agent-first framework**. You create Agents, not direct LLM calls. 
LLM providers (OpenAI, MockLLM, etc.) are internal implementation details used by Agents.

## Structure

```
src/examples/
├── README.md                    # This file
├── agents/                      # Agent examples (PRIMARY INTERFACE)
│   ├── basic_agent.py          # Basic agent with OpenAI
│   ├── math_agent.py           # Math agent with tools
│   └── openai_agent.py         # Full OpenAI agent example
├── prompts/                     # Prompt technique examples
│   ├── zero_shot_examples.py
│   ├── few_shot_examples.py
│   ├── chain_of_thought_examples.py
│   ├── auto_chain_of_thought_examples.py
│   ├── retrieval_augmented_generation_examples.py
│   ├── example_meta_prompt_usage.py
│   ├── prompt_composer_examples.py
│   └── usage_example_composer_prompts.py
└── tools/                       # Tool examples
    ├── basic_tools.py
    └── custom_tools.py
```

**Note:** Examples are located in `src/examples/` so they can import `nucleusiq` directly without path manipulation.

## Quick Start

### Running Examples

All examples focus on **Agent creation and usage**:

```bash
# Basic agent example
python src/examples/agents/basic_agent.py

# Math agent with tools
python src/examples/agents/math_agent.py

# Execution modes (DIRECT, STANDARD, AUTONOMOUS) - ⭐ NEW
python src/examples/agents/execution_modes_example.py

# Gearbox strategy guide - ⭐ NEW
python src/examples/agents/gearbox_strategy_example.py

# Full OpenAI agent
python src/examples/agents/openai_agent.py

# Prompt examples
python src/examples/prompts/zero_shot_examples.py
```

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables (for OpenAI examples):
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Examples by Category

### Agents (Primary Interface) ⭐
- **basic_agent.py** - Simple agent example with OpenAI
- **math_agent.py** - Math calculation agent with tools
- **simple_agent_example.py** - Simple agent with MockLLM and tools
- **execution_modes_example.py** - ⭐ NEW: Comprehensive examples of all execution modes (DIRECT, STANDARD, AUTONOMOUS)
- **gearbox_strategy_example.py** - ⭐ NEW: Guide to choosing the right execution mode for your task
- **task_usage_example.py** - ⭐ NEW: Different ways to create and use Tasks with Agents
- **openai_agent.py** - Full OpenAI agent example
- **openai_tool_example.py** - OpenAI tools integration
- **openai_all_tools_example.py** - All OpenAI native tools
- **openai_mcp_example.py** - Model Context Protocol example
- **openai_connector_example.py** - OpenAI connector example
- **react_agent_example.py** - ReAct agent pattern
- **task_prompt_plan_example.py** - Task, prompt, and plan relationships

**Key Concept:** You create Agents, not LLM clients. The LLM is passed to the Agent during creation:

```python
from nucleusiq.agents.agent import Agent
from nucleusiq.providers.llms.openai.nb_openai import BaseOpenAI

# Create LLM (internal detail)
llm = BaseOpenAI(model_name="gpt-3.5-turbo")

# Create Agent (primary interface)
agent = Agent(
    name="MyAgent",
    llm=llm,  # LLM is just a parameter
    ...
)
```

### Prompts
- **zero_shot_examples.py** - Zero-shot prompting
- **few_shot_examples.py** - Few-shot prompting with examples
- **chain_of_thought_examples.py** - Chain-of-thought reasoning
- **auto_chain_of_thought_examples.py** - Auto CoT prompting
- **retrieval_augmented_generation_examples.py** - RAG prompting
- **example_meta_prompt_usage.py** - Meta-prompting techniques
- **prompt_composer_examples.py** - Composing multiple prompts
- **usage_example_composer_prompts.py** - Additional composer examples

### Tools
- **basic_tools.py** - Creating and using basic tools
- **custom_tools.py** - Creating custom tools

## Philosophy

**NucleusIQ is Agent-centric:**
- ✅ Create Agents to solve problems
- ✅ Pass LLM providers as configuration
- ✅ LLM providers are interchangeable (OpenAI, MockLLM, etc.)
- ❌ Don't call LLM APIs directly
- ❌ Don't expose LLM internals to users

## Contributing

When adding new examples:
1. Follow the existing structure
2. Include docstrings explaining what the example demonstrates
3. Add error handling
4. Include comments for clarity
5. Test the example before committing

