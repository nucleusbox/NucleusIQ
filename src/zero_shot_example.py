# src/zero_shot_example.py

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique

# Create a Zero-Shot Prompt instance
zero_shot_prompt = PromptFactory.create_prompt(
    technique=PromptTechnique.ZERO_SHOT
)

# Set parameters
zero_shot_prompt.set_parameters(
    system="You are a knowledgeable assistant.",
    user="Explain the theory of relativity."
)

# Optionally, enable CoT
zero_shot_prompt.set_parameters(
    use_cot=True,
    cot_instruction="Let's think step by step."
)

# Set metadata and tags
zero_shot_prompt.set_metadata(
    metadata={"author": "John Doe", "date_created": "2024-05-15"}
).add_tags(
    tags=["physics", "relativity", "education"]
)

# Optionally, set an output parser
def extract_explanation(raw_output: str) -> str:
    """
    Extracts the explanation from the LLM's raw output.
    """
    # Example: Remove surrounding quotes and whitespace
    return raw_output.strip().strip("'\"")

zero_shot_prompt.set_output_parser(
    parser=extract_explanation
)

# Format the prompt
formatted_zero_shot_prompt = zero_shot_prompt.format_prompt()
print("Formatted Zero-Shot Prompt:")
print(formatted_zero_shot_prompt)
print("\nMetadata:")
print(zero_shot_prompt.metadata)
print("\nTags:")
print(zero_shot_prompt.tags)

# Simulate LLM response and parse it
raw_llm_response = "'The theory of relativity encompasses two interrelated theories by Albert Einstein: special relativity and general relativity.'"
parsed_explanation = zero_shot_prompt.output_parser(raw_llm_response)
print("\nParsed Explanation:")
print(parsed_explanation)
