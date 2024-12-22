# examples/chain_of_thought_example.py

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

def chain_of_thought_invalid_use_cot():
    # Create a ChainOfThoughtPrompt instance using the factory
    cot_prompt = PromptFactory.create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)

    try:
        # Attempt to configure with use_cot=False
        cot_prompt.configure(
            system="You are a logical reasoning assistant.",
            user="Solve the following problem: What is 2 + 2?",
            use_cot=False  # This should raise an error
        )
    except ValueError as e:
        print(f"Error: {e}")

def chain_of_thought_custom_cot_example():
    # Create a ChainOfThoughtPrompt instance using the factory
    cot_prompt = PromptFactory.create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)

    # Configure the prompt with a custom cot_instruction
    cot_prompt.configure(
        system="You are a logical reasoning assistant.",
        user="Solve the following problem: What is the capital of France?",
        cot_instruction="Let's analyze the question step by step."
    )

    # Format the prompt
    final_prompt = cot_prompt.format_prompt()
    print("Chain-of-Thought Prompt with Custom CoT Instruction:\n")
    print(final_prompt)

if __name__ == "__main__":
    chain_of_thought_example()
    chain_of_thought_invalid_use_cot()
    chain_of_thought_custom_cot_example()
