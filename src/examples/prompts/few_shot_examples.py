# src/examples/prompts/few_shot_examples.py

import os
import sys

# Add src directory to path so we can import nucleusiq
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique


def few_shot_combined_example():
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

def few_shot_configure_example():
    # Create a FewShotPrompt instance using the factory
    few_shot = PromptFactory.create_prompt(PromptTechnique.FEW_SHOT)

    # Configure the prompt and add examples simultaneously
    few_shot.configure(
        system="You are a multilingual translation assistant.",
        user="Translate 'Good night' to Italian.",
        use_cot=False,
        examples=[
            {"input": "Translate 'Hello' to Spanish.", "output": "Hola"},
            {"input": "Translate 'Goodbye' to French.", "output": "Au revoir"}
        ]
    )

    # Format the prompt
    final_prompt = few_shot.format_prompt()
    print("Few-Shot Prompt with Configured Examples:\n")
    print(final_prompt)

def few_shot_combined__with_cot_example():
    # Create a FewShotPrompt instance using the factory
    few_shot = PromptFactory.create_prompt(PromptTechnique.FEW_SHOT)

    # Configure the prompt and add initial examples
    few_shot.configure(
        system="You are a multilingual translation assistant.",
        user="Translate 'Good morning' to Japanese.",
        use_cot=True,  # Enabling CoT
        # cot_instruction=None,  # Not providing cot_instruction; should default
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

if __name__ == "__main__":
    few_shot_combined_example()
    few_shot_configure_example()
    few_shot_combined__with_cot_example()
