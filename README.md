# NucleusIQ

**NucleusIQ** is an open-source framework for building and managing autonomous AI agents. It offers diverse strategies and architectures to create intelligent chatbots, financial tools, and multi-agent systems. With NucleusIQ, developers have the core components and flexibility needed to develop advanced AI applications effortlessly.

## Features

- **Autonomous AI Agents:** Easily create and manage AI-driven agents.
- **Versatile Architectures:** Supports various agentic frameworks and strategies.
- **Flexible Orchestration:** Seamlessly orchestrate multiple agents for complex tasks.
- **Intelligent Chatbots:** Develop smart conversational agents with ease.
- **Financial Analysis Tools:** Build applications for comprehensive financial insights.

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
