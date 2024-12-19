# NucleusIQ Prompt Framework

The **NucleusIQ Prompt Framework** is a versatile and extensible system for creating, managing, and utilizing prompt templates tailored to various language models (LLMs). By abstracting prompt engineering complexities and integrating multiple LLMs, this framework simplifies crafting sophisticated prompts for diverse use cases.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
    - [Zero-Shot Prompting](#zero-shot-prompting)
    - [Few-Shot Prompting](#few-shot-prompting)
    - [Chain-of-Thought (CoT) Prompting](#chain-of-thought-cot-prompting)
    - [Auto Chain-of-Thought (Auto-CoT) Prompting](#auto-chain-of-thought-auto-cot-prompting)
    - [Retrieval-Augmented Generation (RAG) Prompting](#retrieval-augmented-generation-rag-prompting)
4. [Running Unit Tests](#running-unit-tests)
5. [Extending the Framework](#extending-the-framework)
    - [Adding New Prompt Types](#adding-new-prompt-types)
    - [Integrating New LLMs](#integrating-new-llms)
6. [Troubleshooting](#troubleshooting)
7. [Contributing](#contributing)
8. [License](#license)

---

## Features

- **Multiple Prompt Techniques**:
  - **Zero-Shot Prompting**: Directly ask tasks without examples.
  - **Few-Shot Prompting**: Guide tasks using provided examples.
  - **Chain-of-Thought (CoT)**: Encourage step-by-step reasoning.
  - **Auto-CoT**: Automatically cluster questions and generate reasoning chains.
  - **Retrieval-Augmented Generation (RAG)**: Enhance prompts using retrieved context.

- **Extensible Design**:
  - Add custom prompt types or integrate new LLMs seamlessly.

- **Language Model Integration**:
  - Supports OpenAI API, with `MockLLM` for testing and debugging.

- **Adapter Pattern**:
  - Abstracts LLM interaction, enabling easy switching between models.

- **Pydantic Validation**:
  - Ensures correctness of prompt templates with robust data validation.

- **Factory Pattern**:
  - Simplifies prompt creation with centralized configuration management.

---

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package manager)

### Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/NucleusIQ.git
    cd NucleusIQ
    ```

2. **Set Up Virtual Environment**

    It's recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    Install the required Python packages using `pip`.

    ```bash
    pip install -r requirements.txt
    ```

    **`requirements.txt` Contents:**

    ```txt
    pydantic
    scikit-learn
    openai
    yaml
    ```

4. **Set OpenAI API Key (Optional)**

    If you plan to use OpenAI's models, set the `OPENAI_API_KEY` environment variable.

    - **Unix/Linux:**

        ```bash
        export OPENAI_API_KEY='your-openai-api-key'
        ```

    - **Windows:**

        ```cmd
        set OPENAI_API_KEY='your-openai-api-key'
        ```

    **Note:** If you don't have OpenAI API access, you can continue using the `MockLLM` for testing purposes.

---

## Usage

The NucleusIQ Prompt Framework allows you to create and manage various prompt types tailored to different language models. Below are detailed usage examples for each prompt type.

### Zero-Shot Prompting

**Description:**  
Zero-Shot Prompting involves directly asking the LLM to perform a task without providing any examples. It's useful for straightforward tasks where the model's inherent capabilities suffice.

**Code Example:**

```python
from nucleusiq.prompts.factory import PromptFactory
from nucleusiq.llms.mock_llm import MockLLM  # Using MockLLM for testing

def zero_shot_example():
    # Initialize MockLLM
    mock_llm = MockLLM()

    # Create ZeroShotPrompt
    zero_shot_prompt = PromptFactory.create_prompt(
        technique="zero_shot",
        llm=mock_llm
    ).partial(
        system="You are a helpful assistant.",
        user="Translate 'Good morning' to Spanish."
    )

    # Generate Prompt
    print("Zero-Shot Prompt:")
    print(zero_shot_prompt.format_prompt())

# Create FewShotPrompt with examples
    few_shot_prompt = PromptFactory.create_prompt(
        technique="few_shot",
        llm=mock_llm,
        examples=[
            {"input": "Translate 'Hello' to French.", "output": "Bonjour"},
            {"input": "Translate 'Thank you' to German.", "output": "Danke"}
        ]
    ).partial(
        user="Translate 'Good night' to Italian."
    )
```