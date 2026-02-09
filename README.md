# NucleusIQ: A Philosophy of Empowered AI Agents

**NucleusIQ** is an open-source framework grounded in the conviction that true innovation in AI stems from empowering developers with flexible, open-ended tools. By weaving together robust prompting techniques, multi-embedding model integration, and deep observability features, NucleusIQ sets out to redefine how we build and manage autonomous AI agents. Whether crafting a single-agent chatbot or orchestrating a constellation of specialized agents, NucleusIQ offers both the architectural freedom and practical reliability needed to bring advanced AI applications to life.

## Philosophy
At the heart of NucleusIQ lies a commitment to simplicity, flexibility, and control. We believe that building intelligent systems should be accessible and manageable, allowing developers to focus on innovation rather than getting bogged down by complexity. NucleusIQ is built from the ground up with these principles in mind, ensuring that every component is designed to work seamlessly together while providing the freedom to customize and extend as needed.

## Core Beliefs
1. Empower Developers: Provide tools that are powerful yet easy to use, enabling developers to build advanced AI systems without unnecessary hurdles.
2. Flexibility and Modularity: Allow developers to integrate various models, databases, and techniques, creating tailored solutions that fit specific needs.
3. Transparency and Control: Ensure that every aspect of the AI agent’s behavior is observable and controllable, fostering trust and reliability in AI applications.
4. Community-Driven Innovation: Encourage collaboration and contributions from the open-source community to continuously enhance and evolve the framework.

## Unique Attributes
**NucleusIQ** stands out in the crowded landscape of AI frameworks due to its **all-in-one** approach and foundational design. Here’s what makes it unique:

- Comprehensive Integration: Combines multiple Large Language Models (LLMs), embedding models, and vector databases into a single, cohesive framework.
- Built from the Ground Up: Designed with foundational principles that prioritize simplicity, flexibility, and control, avoiding the pitfalls of overly complex or rigid systems.
- Stateful and Iterative Workflows: Supports stateful interactions and iterative processes, allowing agents to remember past interactions and refine their actions over time.
- Extensible Architecture: Modular design that makes it easy to add new functionalities, models, or databases without disrupting existing workflows.
- Robust Observability: Detailed logging and monitoring capabilities ensure that developers can track and understand every step of the agent’s decision-making process.

## Key Pillars of NucleusIQ

- **Multi-LLM & Multi-Embedding Integration:**
    - Use Different LLMs: You can easily plug in various language models—such as OpenAI’s GPT series or other open-source models—and switch between them depending on your needs (for example, cost vs. performance).
    - Support for Multiple Embedding Models: You can choose from many embedding models (e.g., those from Hugging Face) to handle tasks like finding similar documents or clustering topics.
    - Seamless Switching: It’s simple to switch or combine models in your workflow, allowing you to experiment without being locked into one provider.

- **Advanced Retrieval-Augmented Generation (RAG):**
    - Vector Database Compatibility: Connect with a range of vector databases (e.g., Pinecone, Chroma) to store and retrieve embeddings at scale.
    - Contextual Knowledge Injection: Seamlessly ground agent outputs in relevant documents, improving factual correctness and trustworthiness.
    - Scalable Workflows: Design small, on-demand retrieval mechanisms or expand into complex multi-agent RAG systems—tailored to your application’s size and scope.

- **Comprehensive Prompting Techniques:**
    - Zero-Shot & Few-Shot: Rapidly prototype solutions without large datasets.
    - Chain of Thought & AUTO_CHAIN_OF_THOUGHT: Encourage agents to reason step-by-step, improving interpretability and correctness.
    - Retrieval-Augmented Generation: Combine external knowledge from vector databases with LLM-based reasoning.
    - Prompt Composer & Meta-Prompting: Layer or compose prompts for higher-level reasoning and planning.

- **Autonomous & Multi-Agent Workflows:**
    - **From Simple to Complex:** You can start with one chatbot handling basic questions or grow into a full network of specialized agents working together on bigger tasks.
    - **Orchestration:** An “orchestrator” agent can coordinate other agents, splitting tasks among them and gathering their results.
    - **Stateful Cycles:** Agents can remember what has happened before, letting them review or refine their own decisions in loops.
- **Observability & Persistence:**
    - **Verbose Logging:** Gain insight into the decision-making process at every stage, making it easy to troubleshoot, optimize, or audit agent behavior.
    - **Memory & Caching:** Persist intermediate outputs, conversations, and states to ensure continuity—especially critical for long-running tasks or complex RAG workflows.
    - **Human-in-the-Loop:** Incorporate user oversight at key junctures, enabling interactive correction or fine-tuning of decisions.

- **Controllability & Reliability:**
    - Fine-Grained Control: Maintain oversight of each agent’s flow, from prompt construction to final output, ensuring consistency and reliability.
    - Exception Handling & Recovery: Detect failures early and recover gracefully, utilizing retries or fallback strategies to maintain robustness.

- **Extensibility & Modularity:**
    - Plugin Architecture: Simplify the integration of new LLM providers, specialized prompt libraries, or domain-specific agents.
    - Object-Oriented Design: Utilize clear class hierarchies and composition, allowing each part of the framework (agents, orchestrators, connectors) to evolve independently.
    - Open-Source Community: Encourage contributions, enabling an ecosystem of shared best practices, adapters, and specialized modules.

## Installation

### Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or **pip**

### Quick Install

```bash
git clone https://github.com/nucleusbox/NucleusIQ.git
cd NucleusIQ/src/nucleusiq

# Using uv (recommended)
uv venv && uv sync --all-groups

# Or using pip
python -m venv .venv
# Windows: .venv\Scripts\activate | Unix: source .venv/bin/activate
pip install -e .
```

### Verify

```bash
cd src/nucleusiq
uv run pytest tests/ -q
```

## Current Status

**Version:** 0.1.0

### ✅ Implemented
- **Core agent framework**: Agent, ReActAgent with execution modes (DIRECT, STANDARD, AUTONOMOUS)
- **Planning**: LLM-based plan generation, basic fallback, `$step_N` context resolution
- **7+ prompt techniques**: Zero-shot, Few-shot, CoT, Auto-CoT, RAG, Meta-prompting, Prompt Composer
- **Tool system**: BaseTool, Executor, OpenAI function-calling, MCP, code interpreter, file search
- **Structured output**: NATIVE mode with OpenAI `response_format` (Pydantic, dataclass, TypedDict)
- **OpenAI provider**: `nucleusiq-openai` (chat, embeddings, structured output)
- **MockLLM** for testing without API calls
- **State management** and observability
- **355 tests**, 35 examples — all passing

### 📋 Planned
- Memory system (interface defined, implementation planned)
- Additional LLM providers (Ollama, Groq, Gemini)
- Vector database integrations (Pinecone, Chroma)
- Multi-agent orchestration
- Structured Output TOOL and PROMPT modes

## Contributing

NucleusIQ is open-source and we welcome contributions! If you're looking to contribute, please:

Fork the repository.
Create a new branch for your feature.
Add your feature or improvement.
Send a pull request.
We appreciate your input!

## License

This project is licensed under the [MIT License](LICENSE).

---

**Join us in building the future of autonomous AI orchestration with NucleusIQ!**
