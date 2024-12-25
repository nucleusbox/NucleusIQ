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

def basic_variable_mappings_example():
    # Define a custom template with placeholders
    custom_template = """
    System: {sys}
    User: {usr}
    Query: {qry}
    """

    # Define variable mappings: logical_var -> template_placeholder
    var_mappings = {
        "system": "sys",
        "user_query": "usr",
        "questions": "qry",
    }

    # Create a PromptComposer instance with variable mappings
    composer = (
        PromptFactory
        .create_prompt(PromptTechnique.PROMPT_COMPOSER)
        .configure(
            template=custom_template,
            variable_mappings=var_mappings
        )
    )

    # Format the prompt with logical variables
    formatted_prompt = composer.format_prompt(
        system="You are a helpful assistant.",
        user_query="How's the weather today?",
        questions="What's the forecast for tomorrow?"
    )

    print("=== Basic Variable Mappings Example ===")
    print(formatted_prompt)

def function_mappings_example():
    # Define a custom template with placeholders
    custom_template = """
    Tasks:
    {tasks}

    User: {user_query}
    """

    # Define a function to format tasks as a bulleted list
    def format_tasks(**kwargs):
        tasks = kwargs.get("tasks", "")
        if not tasks:
            raise ValueError("No tasks provided.")
        task_list = tasks.split("\n")
        return "\n".join([f"- {task.strip()}" for task in task_list if task.strip()])

    # Create a PromptComposer instance with function mappings
    composer = (
        PromptFactory
        .create_prompt(PromptTechnique.PROMPT_COMPOSER)
        .configure(
            template=custom_template,
            function_mappings={"tasks": format_tasks}
        )
    )

    # Format the prompt with raw tasks
    formatted_prompt = composer.format_prompt(
        tasks="Wash dishes\nClean the house\nTake out the trash",
        user_query="Please complete these tasks by evening."
    )

    print("=== Function Mappings Example ===")
    print(formatted_prompt)

def combined_mappings_example():
    # Define a custom template with placeholders
    custom_template = """
    System: {sys}
    Examples:
    {ex}
    Query: {q}
    """

    # Define variable mappings: logical_var -> template_placeholder
    var_mappings = {
        "system": "sys",
        "examples": "ex",
        "user_query": "q"
    }

    # Define a function to add exclamation marks to examples
    def add_exclamation(**kwargs):
        examples = kwargs.get("orig", "")
        return f"{examples}!!!"

    # Create a PromptComposer instance with both variable and function mappings
    composer = (
        PromptFactory
        .create_prompt(PromptTechnique.PROMPT_COMPOSER)
        .configure(
            template=custom_template,
            variable_mappings=var_mappings,
            function_mappings={"examples": add_exclamation}
        )
    )

    # Format the prompt with logical variables and original examples
    formatted_prompt = composer.format_prompt(
        system="You are a knowledgeable assistant.",
        user_query="Provide examples of AI applications.",
        orig="Example 1: Chatbots\nExample 2: Recommendation Systems"
    )

    print("=== Combined Variable and Function Mappings Example ===")
    print(formatted_prompt)


if __name__ == "__main__":
    prompt_composer_example()
    prompt_composer_example1()
    basic_variable_mappings_example()
    function_mappings_example()
    combined_mappings_example()