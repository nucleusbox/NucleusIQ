# src/chain_of_thought_example.py

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique

# Create a Chain-of-Thought Prompt instance
cot_prompt = PromptFactory.create_prompt(
    technique=PromptTechnique.CHAIN_OF_THOUGHT
)

# Set parameters
cot_prompt.set_parameters(
    system="You are a logical and analytical assistant.",
    user="Solve the following math problem: What is the integral of x^2?"
)

# Set metadata and tags
cot_prompt.set_metadata(
    metadata={"author": "Alice Johnson", "date_created": "2024-06-20"}
).add_tags(
    tags=["math", "calculus", "integration", "CoT"]
)

# Optionally, set an output parser
def extract_solution(raw_output: str) -> str:
    """
    Extracts the solution from the LLM's raw output.
    """
    # Example: Return the last line as the solution
    return raw_output.strip().split('\n')[-1].strip("'\"")

cot_prompt.set_output_parser(
    parser=extract_solution
)

# Format the prompt
formatted_cot_prompt = cot_prompt.format_prompt()
print("Formatted Chain-of-Thought Prompt:")
print(formatted_cot_prompt)
print("\nMetadata:")
print(cot_prompt.metadata)
print("\nTags:")
print(cot_prompt.tags)

# Simulate LLM response and parse it
raw_llm_response = """
To find the integral of x^2, we use the power rule for integration.
The power rule states that the integral of x^n is (x^(n+1))/(n+1) + C, where C is the constant of integration.
Applying this rule, the integral of x^2 is (x^3)/3 + C.
"""
parsed_solution = cot_prompt.output_parser(raw_llm_response)
print("\nParsed Solution:")
print(parsed_solution)
