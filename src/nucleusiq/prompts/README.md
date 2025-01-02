# NucleusIQ Prompt Framework

The **NucleusIQ Prompt Framework** is a versatile and extensible system for creating, managing, and utilizing prompt templates tailored to various language models (LLMs). By abstracting prompt engineering complexities and integrating multiple LLMs, this framework simplifies crafting sophisticated prompts for diverse use cases.

---

## Table of Contents

- [NucleusIQ Prompt Framework](#nucleusiq-prompt-framework)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)
    - [Zero-Shot Prompting](#zero-shot-prompting)
    - [Few-Shot Prompting](#few-shot-prompting)
    - [Chain-of-Thought (CoT) Prompting](#chain-of-thought-cot-prompting)
    - [Auto Chain-of-Thought (Auto-CoT) Prompting](#auto-chain-of-thought-auto-cot-prompting)
    - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
    - [Meta Prompt](#meta-prompt)
    - [Prompt Composer Technique](#prompt-composer-technique)
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

### Few-Shot Prompting

**Description:**  
Few-shot prompting provides the model with a few examples within the prompt to guide its behavior. This technique helps the model understand the desired task by learning from the provided examples.

**Use Case:**

- Mimicking specific writing styles.
- Learning a pattern quickly.
- Performing tasks with limited examples.

**Code Example:**

```python
    from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
    # Create a FewShotPrompt instance using the factory
        few_shot = PromptFactory.create_prompt(PromptTechnique.FEW_SHOT)

        # Configure the prompt and add initial examples
        few_shot.configure(
            system="You are a multilingual translation assistant.",
            user="Translate 'Good morning' to Japanese.",
            use_cot=False,
            examples=[
                {"input": "Translate 'Hello' to Spanish.", "output": "Hola"},
                {"input": "Translate 'Goodbye' to French.", "output": "Au revoir"}
            ]
        )

        # Add more examples incrementally
        few_shot.add_example(
            input_text="Translate 'Please' to German.",
            output_text="Bitte"
        )
        few_shot.add_example(
            input_text="Translate 'Thank you' to Italian.",
            output_text="Grazie"
        )

        # Format the prompt
        final_prompt = few_shot.format_prompt()
        print("Few-Shot Prompt with Combined Methods:\n")
        print(final_prompt)
```

### Chain-of-Thought (CoT) Prompting

**Description:**
Chain-of-Thought (CoT) prompting encourages the model to generate intermediate reasoning steps before providing a final answer. This approach enhances the model's ability to handle complex tasks by decomposing them into simpler, sequential steps.

**Use Case:**
- Solving complex problems.
- Encouraging logical reasoning.
- Improving problem-solving accuracy.

```python
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
def chain_of_thought_example():
    # Create a ChainOfThoughtPrompt instance using the factory
    cot_prompt = PromptFactory.create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)

    # Configure the prompt
    cot_prompt.configure(
        system="You are a logical reasoning assistant.",
        user="Solve the following problem: If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?",
        use_cot=True,
        cot_instruction="Let's reason through this step by step."
    )

    # Format the prompt
    final_prompt = cot_prompt.format_prompt()
    print("Chain-of-Thought Prompt:\n")
    print(final_prompt)
```

### Auto Chain-of-Thought (Auto-CoT) Prompting

**Description:**
Auto-CoT automates the generation of reasoning steps by prompting the model to produce intermediate steps without manual examples. This technique leverages the model's ability to self-generate logical sequences, reducing the need for handcrafted prompts.

**Use Case:**

- Automating reasoning processes.
- Reducing manual prompt design efforts.
- Enhancing model autonomy in problem-solving.

```python
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
def mock_llm(self):
	"""Simple fixture returning a mock LLM that returns a constant chain text."""
	return MockLLM()

def test_auto_cot_creation_success(self, mock_llm):
	"""
	Provide valid task, questions, llm, etc. => success
	"""
	task = "Provide detailed reasoning for these math problems."
	questions = [
		"Calculate the product of prime numbers < 10.",
		"Determine if 29 is prime."
	]
	system_prompt = "You are an assistant specialized in math."
	user_prompt = "Here are your problems:"

	auto_cot = (
		PromptFactory
		.create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
		.configure(
			llm=mock_llm,
			num_clusters=2,
			max_questions_per_cluster=1,
			instruction="Let's think step by step.",
			system=system_prompt,
			user=user_prompt
		)
	)
	# Then we call format_prompt or _construct_prompt
	prompt_text = auto_cot.format_prompt(task=task, questions=questions)
	print(prompt_text)
```

### Retrieval-Augmented Generation (RAG)

**Description:**
RAG combines large language models with external retrieval mechanisms to enhance the generation process. By accessing external knowledge sources, the model can produce more accurate and contextually relevant responses.

**Use Case:**

- Providing up-to-date information.
- Answering domain-specific queries.
- Enhancing response accuracy with external data.

```python
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
def test_rag_creation_success(self):
	"""
	Provide system, non-empty context, user => success
	"""
	rag_prompt = (
		PromptFactory
		.create_prompt(PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION)
		.configure(
			system="You are an assistant with knowledge base access.",
			context="France is a country in Western Europe known for its rich history and culture.",
			user="What is the capital of France?"
		)
	)
	prompt_text = rag_prompt.format_prompt()
	expected_prompt = (
		"You are an assistant with knowledge base access.\n\n"
		"France is a country in Western Europe known for its rich history and culture.\n\n"
		"What is the capital of France?"
	)
	assert prompt_text.strip() == expected_prompt.strip()
```

### Meta Prompt

**Description:**
Meta prompting focuses on creating prompts that guide the model in generating or refining other prompts. This higher-level approach emphasizes the structure and pattern of information, enabling dynamic prompt generation and iterative refinement.

**Use Case:**

- Developing dynamic prompt templates.
- Iteratively refining prompts for improved outputs.
- Enhancing prompt design efficiency.

Please refer this [Meta Prompt example](../../examples/example_meta_prompt_usage.py)


### Prompt Composer Technique

**Description:**
The Prompt Composer is an advanced prompt engineering technique that enables the dynamic generation of prompts by composing them from multiple variables, examples, and reasoning steps. This approach leverages a predefined template enriched with placeholders and function mappings to dynamically format input data into meaningful and structured outputs. By combining logical variable mappings, runtime transformations, and extended placeholders like examples and chains of thought, Prompt Composer ensures flexibility and adaptability in a variety of use cases.

**Use Case:**

1. Dynamic Template Composition:
Combines multiple data points, such as user queries, examples, and reasoning, into a cohesive prompt.

2. Runtime Transformations:
Uses function mappings to transform data dynamically during the prompt generation process.

3. Flexible Placeholders:
Includes predefined placeholders for elements like examples, chains of thought, and user queries, allowing for modular and reusable templates.

4. Error Handling and Validation:
Ensures that required fields are populated and all placeholders in the template are accounted for.

Please refer this [Prompt Composer Example](../../examples/prompt_composer_examples.py)
