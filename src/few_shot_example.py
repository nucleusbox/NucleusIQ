# src/few_shot_example.py

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.few_shot import FewShotPrompt

# Create a Few-Shot Prompt instance
few_shot_prompt: FewShotPrompt = PromptFactory.create_prompt(
    technique=PromptTechnique.FEW_SHOT,
)

# Set parameters
few_shot_prompt.set_parameters(
    system="You are a helpful assistant.",
    user="Translate the following English text to French: 'Good morning.'"
)

# Add a single example
few_shot_prompt.add_example(
    input_text="Translate 'Good evening.' to French.",
    output_text="'Bonsoir.'"
)

# Add multiple examples at once
additional_examples = [
    {"input": "Translate 'Thank you.' to French.", "output": "'Merci.'"},
    {"input": "Translate 'How are you?' to French.", "output": "'Comment ça va?'"}
]
few_shot_prompt.add_examples(additional_examples)

# Optionally, enable CoT
few_shot_prompt.set_parameters(
    use_cot=True,
    cot_instruction="Let's think step by step."
)

# Set metadata and tags
few_shot_prompt.set_metadata(
    metadata={"author": "Jane Smith", "date_created": "2024-04-01"}
).add_tags(
    tags=["translation", "French", "greeting", "CoT"]
)

# Optionally, set an output parser
def extract_translation(raw_output: str) -> str:
    """
    Extracts the translated text from the LLM's raw output.
    """
    # Example: Remove surrounding quotes and whitespace
    return raw_output.strip().strip("'\"")

few_shot_prompt.set_output_parser(
    parser=extract_translation
)

# Format the prompt
formatted_few_shot_prompt = few_shot_prompt.format_prompt()
print("Formatted Few-Shot Prompt:")
print(formatted_few_shot_prompt)
print("\nMetadata:")
print(few_shot_prompt.metadata)
print("\nTags:")
print(few_shot_prompt.tags)

# Simulate LLM response and parse it
raw_llm_response = "'Bonjour, comment ça va?'"
parsed_translation = few_shot_prompt.output_parser(raw_llm_response)
print("\nParsed Translation:")
print(parsed_translation)
