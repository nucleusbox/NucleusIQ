# examples/auto_chain_of_thought_example.py

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.auto_chain_of_thought import AutoChainOfThoughtPrompt
from nucleusiq.llms.mock_llm import MockLLM

def auto_cot_example():
    auto_cot: AutoChainOfThoughtPrompt= PromptFactory.create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
    # Set multiple parameters in one go
    auto_cot.set_parameters(
        system="You are an intelligent assistant.",
        context="",
        user="Describe the process of photosynthesis.",
        instruction="Let's reason step by step.",
        num_clusters=2,
        max_questions_per_cluster=1,
        cot_instruction="(Additional CoT text at the end.)"
    )
    # Provide an LLM
    auto_cot.llm = MockLLM()
    # Format the prompt
    prompt_text = auto_cot.format_prompt(
        task="Explain how photosynthesis works in detail.",
        questions=[
            "What is the main chemical equation for photosynthesis?",
            "How does sunlight factor in photosynthesis?",
             "What is the main sdjhdjs equation for photosynthesis?",
            "How does skdjk factor in photosynthesis?"
        ]
    )
    print(prompt_text)

if __name__ == "__main__":
    auto_cot_example()
