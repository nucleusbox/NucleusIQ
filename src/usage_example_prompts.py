# examples/usage_example_prompts.py

from nucleusiq.prompts.factory import PromptFactory
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.llms.mock_llm import MockLLM

def main():
    # 1. Zero-Shot Prompt without CoT
    zero_shot = PromptFactory.create_prompt(
        technique="zero_shot"
    ).partial(
        system="You are a helpful assistant.",
        user="Translate the following English text to French: 'Hello, how are you?'"
        # 'context' is optional and omitted
    )
    
    print("Zero-Shot Prompt:")
    print(zero_shot.format_prompt())
    print("\n" + "="*50 + "\n")

    # 2. Zero-Shot Prompt with CoT
    zero_shot_cot = PromptFactory.create_prompt(
        technique="zero_shot",
        use_cot=True,  # Enable CoT
        cot_instruction="Let's think step by step."
    ).partial(
        system="You are a helpful assistant.",
        user="Translate the following English text to French: 'Hello, how are you?'"
        # 'context' is optional and omitted
    )
    
    print("Zero-Shot Prompt with CoT:")
    print(zero_shot_cot.format_prompt())
    print("\n" + "="*50 + "\n")

    # 3. Few-Shot Prompt without CoT
    few_shot = PromptFactory.create_prompt(
        technique="few_shot",
        examples=[
            {"input": "Translate 'Good morning' to Spanish.", "output": "Buenos días."},
            {"input": "Translate 'Thank you' to German.", "output": "Danke."},
        ],
    ).partial(
        system="",  # No additional system instructions
        user="Translate 'Good night' to Italian.",
        examples=None  # 'examples' are already added via 'add_example'
    )
    
    print("Few-Shot Prompt:")
    print(few_shot.format_prompt())
    print("\n" + "="*50 + "\n")

    # 4. Few-Shot Prompt with CoT
    few_shot_cot = PromptFactory.create_prompt(
        technique="few_shot",
        examples=[
            {
                "input": "Translate 'Good morning' to Spanish.",
                "output": "Buenos días."
            },
            {
                "input": "The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.",
                "output": "Adding all the odd numbers (9, 15, 1) gives 25. The answer is False."
            },
        ],
        use_cot=True,  # Enable CoT
        cot_instruction="Let's think step by step."
    ).add_example(
        "The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12, 24.",
        "Adding all the odd numbers (17, 19) gives 36. The answer is True."
    ).partial(
        system="",  # No additional system instructions
        user="The odd numbers in this group add up to an even number: 16,  11, 14, 4, 8, 13, 24.\nA:",
        examples=None  # 'examples' are already set
    )
    
    print("Few-Shot Prompt with CoT in Examples:")
    print(few_shot_cot.format_prompt())
    print("\n" + "="*50 + "\n")

    # 5. Chain-of-Thought Prompt
    cot_prompt = PromptFactory.create_prompt(
        technique="chain_of_thought"
    ).partial(
        system="You are an analytical assistant.",
        user="Is the following statement true or false? The sum of all even numbers between 1 and 10 is greater than 30."
        # 'context' is optional and omitted
    )
    
    print("Chain-of-Thought Prompt:")
    print(cot_prompt.format_prompt())
    print("\n" + "="*50 + "\n")

    # 6. Auto Chain-of-Thought Prompt

    # 1. Basic Auto Chain-of-Thought Prompt
    task = "Provide detailed reasoning for the following mathematical problems."
    questions = [
        "Calculate the product of all prime numbers less than 10.",
        "Determine if the number 29 is prime.",
        "Find the factorial of 5.",
        "What is the tallest mountain in the world?",
        "Who wrote the play 'Romeo and Juliet'?",
        "What is the chemical symbol for gold?"
    ]
    system_prompt = "You are an assistant specialized in mathematical problem-solving."
    user_prompt = "Here are your problems:"

    # Initialize NucleusIQLLM
    mock_llm = MockLLM()

     # Create AutoChainOfThoughtPrompt with NucleusIQLLM
    auto_cot_prompt_mock = PromptFactory.create_prompt(
        technique="auto_chain_of_thought",
        llm=mock_llm,  # Inject the mock LLM
        num_clusters=3,
        max_questions_per_cluster=1,
        instruction="Let's think step by step."
    ).partial(
        task=task,
        questions=questions,
        system=system_prompt,
        user=user_prompt
    )

    print("Auto Chain-of-Thought Prompt with MockLLM:")
    print(auto_cot_prompt_mock.format_prompt())
    print("\n" + "="*50 + "\n")

    # Create AutoChainOfThoughtPrompt with NucleusIQLLM without system and context
    auto_cot_prompt_mock_no_context = PromptFactory.create_prompt(
        technique="auto_chain_of_thought",
        llm=mock_llm,  # Inject the mock LLM
        num_clusters=2,
        max_questions_per_cluster=1,
        instruction="Let's think step by step."
    ).partial(
        task=task,
        questions=questions,
        user=user_prompt
    )

    print("Auto Chain-of-Thought Prompt without Context or System:")
    print(auto_cot_prompt_mock_no_context.format_prompt())
    print("\n" + "="*50 + "\n")

    # 7. Retrieval Augmented Generation Prompt
    rag_prompt = PromptFactory.create_prompt(
        technique="retrieval_augmented_generation"
    ).partial(
        system="You are an assistant with access to a comprehensive knowledge base.",
        context="France is a country in Western Europe known for its rich history and culture.",
        user="What is the capital of France?"
    )
    print("Retrieval Augmented Generation Prompt:")
    print(rag_prompt.format_prompt())
    print("\n" + "="*50 + "\n")

    # 8. Serialization and Deserialization
    # Save the RAG prompt
    rag_prompt.save("rag_prompt.json")
    
    # Load the RAG prompt
    loaded_rag_prompt = BasePrompt.load("rag_prompt.json")
    print("Loaded Retrieval Augmented Generation Prompt:")
    print(loaded_rag_prompt.format_prompt())
    print("\n" + "="*50 + "\n")
    
    # Clean up the saved file
    import os
    os.remove("rag_prompt.json")

if __name__ == "__main__":
    main()
