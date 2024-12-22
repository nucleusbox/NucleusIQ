# src/prompt_composer_test.py

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique

def prompt_composer_example():
    composer_prompt = PromptFactory.create_prompt(PromptTechnique.PROMPT_COMPOSER)

    # Provide a custom template referencing placeholders
    composer_prompt.template = """\
System Prompt:
{system}

Few-Shot Examples:
{examples}

Chain-of-Thought:
{chain_of_thought}

User Query:
{user_query}
"""

    # Assign fields recognized by PromptComposer
    composer_prompt.system = "You are a multi-purpose AI assistant."
    composer_prompt.examples = "Example 1: ...\nExample 2: ..."
    composer_prompt.chain_of_thought = "Let's think carefully."
    composer_prompt.user_query = "Generate a report on climate change."

    # Now let's format the prompt
    prompt_text = composer_prompt.format_prompt()
    print("Formatted Prompt Composer:\n")
    print(prompt_text)


def prompt_composer_example1():
    composer = PromptFactory.create_prompt(PromptTechnique.PROMPT_COMPOSER)
    composer.configure(
        system="You are a multi-purpose AI assistant.",
        examples="Example 1: ...\nExample 2: ...",
        chain_of_thought="Let's reason carefully.",
        user_query="Generate a summary on climate change.",
        template="""\
System Prompt:
{system}

Few-Shot Examples:
{examples}

Chain-of-Thought:
{chain_of_thought}

User Query:
{user_query}
"""
    )

    # Optionally define variable/function mappings
    # e.g., rename user_query-> query, or bullet-format examples
    # For now we skip it

    final_prompt = composer.format_prompt()
    print("Prompt Composer Output:\n")
    print(final_prompt)

if __name__ == "__main__":
    prompt_composer_example()
    prompt_composer_example1()