# src/usage_example_composer_prompts.py

from nucleusiq.prompts.factory import PromptFactory


def prompt_composer_example():
    print("\n--- Example 3: Variable Mappings ---")
    # Define a custom template with placeholders mapped from logical variables
    custom_template = """
    {system}

    Task: {my_task}

    Questions:
    {my_questions}

    Instructions: {instruction}

    User Prompt:
    {my_user_prompt}

    Desired Output Format: {output_format}
    """

    # Define input variables corresponding to logical variables
    input_vars = ["task", "questions", "user_prompt", "system", "instruction", "output_format"]

    # Define variable mappings: logical_var -> template_placeholder
    variable_mappings = {
        "task": "my_task",
        "questions": "my_questions",
        "user_prompt": "my_user_prompt"
    }

    # Initialize PromptComposer without attaching to any LLM
    try:
        custom_prompt_with_mapping = PromptFactory.create_prompt(
            technique="prompt_composer",
            template=custom_template,
            input_variables=input_vars,
            variable_mappings=variable_mappings,
        )
        print("PromptComposer instance created successfully with variable mappings.")
    except ValueError as e:
        print("Error during prompt creation:", e)
        return

    # Format the prompt with mapped variables
    try:
        formatted_prompt_mapped = custom_prompt_with_mapping.format_prompt(
            system="You are an assistant.",
            task="Translate the following text.",
            questions="1. Translate 'Hello' to French.\n2. Translate 'Goodbye' to German.",
            instruction="Provide accurate translations.",
            user_prompt="Please provide the translations below.",
            output_format="Text",
        )
        print("Prompt formatted successfully with variable mappings.")
    except ValueError as e:
        print("Error during prompt formatting:", e)
        return

    print("\nFormatted Prompt (Variable Mappings):")
    print(formatted_prompt_mapped)

    print("\n--- Example 4: Function Mappings ---")
    # Define a function to format questions as a bulleted list
    def format_questions_fn(**kwargs):
        question_list = kwargs["questions"].split("\n")
        formatted = "\n".join([f"- {q}" for q in question_list])
        return formatted

    # Create another PromptComposer instance with function mappings
    try:
        prompt_composer_func = PromptFactory.create_prompt(
            technique="prompt_composer",
            template="""
            {system}

            Task: {task}

            Questions:
            {questions}

            Instructions: {instruction}

            User Prompt:
            {user_prompt}

            Desired Output Format: {output_format}
            """,
            input_variables=["system", "task", "questions", "instruction", "user_prompt", "output_format"],
            function_mappings={"questions": format_questions_fn},
        )
        print("PromptComposer instance created successfully with function mappings.")
    except ValueError as e:
        print("Error during prompt creation with function mappings:", e)
        return

    # Format the prompt with function mappings
    try:
        formatted_prompt_func = prompt_composer_func.format_prompt(
            system="You are an assistant.",
            task="Provide detailed answers.",
            questions="1. What is AI?\n2. Explain machine learning.",
            instruction="Be thorough and precise.",
            user_prompt="Please provide the answers below.",
            output_format="Markdown",
        )
        print("Prompt formatted successfully with function mappings.")
    except ValueError as e:
        print("Error during prompt formatting with function mappings:", e)
        return

    print("\nFormatted Prompt (Function Mappings):")
    print(formatted_prompt_func)


if __name__ == "__main__":
    prompt_composer_example()
