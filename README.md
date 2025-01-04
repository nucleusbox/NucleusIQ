# NucleusIQ: A Philosophy of Empowered AI Agents

**NucleusIQ** is an open-source framework grounded in the conviction that true innovation in AI stems from empowering developers with flexible, open-ended tools. By weaving together robust prompting techniques, multi-embedding model integration, and deep observability features, NucleusIQ sets out to redefine how we build and manage autonomous AI agents. Whether crafting a single-agent chatbot or orchestrating a constellation of specialized agents, NucleusIQ offers both the architectural freedom and practical reliability needed to bring advanced AI applications to life.

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

- **Python 3.6+**
- **Git**

### Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/nucleusbox/NucleusIQ.git
    cd NucleusIQ
    ```
2. **Set Up Virtual Environment**

    It's recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv nb
    source nb/bin/activate  # On Windows: nb\Scripts\activate
    ```

3. **Install Dependencies**

    Install the required Python packages using `pip`.

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables**

    Create a `.env` file in the root directory and add necessary configurations:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    DATABASE_URL=your_database_url
    ```

5. **Run the Test**

    ```bash
    python pytest
    ```

## Contributing

CrewAI is open-source and we welcome contributions. If you're looking to contribute, please:

Fork the repository.
Create a new branch for your feature.
Add your feature or improvement.
Send a pull request.
We appreciate your input!

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

**Join us in building the future of autonomous AI orchestration with NucleusIQ!**
