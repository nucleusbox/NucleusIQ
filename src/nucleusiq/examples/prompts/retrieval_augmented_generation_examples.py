# src/examples/prompts/retrieval_augmented_generation_examples.py

import os
import sys

# Add src directory to path so we can import nucleusiq
_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique


def rag_example():
    # Create a RetrievalAugmentedGenerationPrompt instance using the factory
    rag_prompt = PromptFactory.create_prompt(
        PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION
    )

    # Configure the prompt
    rag_prompt.configure(
        system="You are an expert in environmental science.",
        context="According to the latest reports, global temperatures have risen by 1.1 degrees Celsius since pre-industrial times.",
        user="Summarize the impact of global warming on polar bears.",
    )

    # Format the prompt
    final_prompt = rag_prompt.format_prompt()
    print("Retrieval-Augmented Generation Prompt:\n")
    print(final_prompt)


if __name__ == "__main__":
    rag_example()
