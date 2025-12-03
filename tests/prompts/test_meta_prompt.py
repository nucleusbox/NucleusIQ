# tests/test_meta_prompt.py

import pytest
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.meta_prompt import MetaPrompt
from pydantic import ValidationError
from typing import Any, Callable

# Mock functions for testing
def mock_output_parser(output: str) -> str:
    return output.upper()

def faulty_function(**kwargs) -> Any:
    raise ValueError("Intentional Error in Function Mapping")

class TestMetaPrompt:
    """
    Test suite for the MetaPrompt class.
    """

    # 1. Initialization and Configuration Tests

    def test_initialization_with_valid_inputs(self):
        """
        Test that MetaPrompt initializes correctly with valid primary and feedback instructions.
        """
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                primary_instruction="Create a prompt for summarizing articles.",
                feedback_instruction="Ensure inclusion of key points and conclusions."
            )
        )
        assert meta_prompt.primary_instruction == "Create a prompt for summarizing articles."
        assert meta_prompt.feedback_instruction == "Ensure inclusion of key points and conclusions."

    def test_initialization_missing_required_fields(self):
        """
        Test that initializing MetaPrompt without required fields raises a ValueError.
        """
        meta_prompt = PromptFactory.create_prompt(PromptTechnique.META_PROMPTING)
        with pytest.raises(ValueError) as exc_info:
            meta_prompt.format_prompt()
        assert "Missing required field 'primary_instruction' or it's empty." in str(exc_info.value)

    def test_initialization_invalid_field_types(self):
        """
        Test that providing invalid types for fields raises appropriate errors.
        """
        with pytest.raises(ValidationError):
            (
                PromptFactory
                .create_prompt(PromptTechnique.META_PROMPTING)
                .configure(
                    primary_instruction=123,  # Should be str
                    feedback_instruction=["Ensure inclusion of key points and conclusions."]  # Should be str
                )
            )

    # 2. Template Handling Tests

    def test_template_non_empty(self):
        """
        Test that assigning an empty template raises a ValueError.
        """
        with pytest.raises(ValueError) as exc_info:
            (
                PromptFactory
                .create_prompt(PromptTechnique.META_PROMPTING)
                .configure(
                    template="   ",  # Empty after stripping
                    primary_instruction="Create a prompt.",
                    feedback_instruction="Provide feedback."
                )
            )
        assert "Template cannot be empty." in str(exc_info.value)

    def test_template_correct_placeholder_detection(self):
        """
        Test that all placeholders in the template are correctly detected and managed.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nExtra: {extra_variable}"
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        # 'extra_variable' should be added to optional_variables
        assert "extra_variable" in meta_prompt.optional_variables

    def test_conflicting_variable_mappings(self):
        """
        Test that conflicting variable mappings raise a ValueError.
        """
        custom_template = "Instruction: {instr}\nPrompt: {prompt}"
        var_mappings = {
            "primary_instruction": "instr",
            "feedback_instruction": "instr",  # Conflict: both map to 'instr'
            "generated_prompt": "prompt"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings
            )
        )
        with pytest.raises(ValueError) as exc_info:
            meta_prompt.format_prompt(
                primary_instruction="Initial instruction.",
                feedback_instruction="Refinement feedback.",
                generated_prompt="Generated prompt."
            )
        assert "Conflicting variable mappings, multiple fields map to ['instr']" in str(exc_info.value)

    # 3. Variable and Function Mappings Tests

    def test_variable_mappings_application(self):
        """
        Test that variable mappings correctly map logical variables to template placeholders.
        """
        custom_template = "Hello {user}, welcome to {platform}!"
        var_mappings = {
            "user_name": "user",
            "platform_name": "platform"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Greet the user.",
                feedback_instruction="Ensure platform name is included."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            user_name="Alice",
            platform_name="ChatGPT"
        )
        assert formatted_prompt == "Hello Alice, welcome to ChatGPT!"

    def test_function_mappings_execution(self):
        """
        Test that function mappings are correctly applied during prompt construction.
        """
        def uppercase_platform(**kwargs):
            platform = kwargs.get("platform_name", "")
            return platform.upper()

        custom_template = "Hello {user}, welcome to {platform}!"
        var_mappings = {
            "user_name": "user",
            "platform_name": "platform"
        }
        func_mappings = {
            "platform_name": uppercase_platform
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                function_mappings=func_mappings,
                primary_instruction="Greet the user.",
                feedback_instruction="Ensure platform name is in uppercase."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            user_name="Bob",
            platform_name="ChatGPT"
        )
        assert formatted_prompt == "Hello Bob, welcome to CHATGPT!"

    def test_non_callable_function_mapping(self):
        """
        Test that providing a non-callable in function mappings raises a ValueError.
        """
        custom_template = "Hello {user}, welcome to {platform}!"
        var_mappings = {
            "user_name": "user",
            "platform_name": "platform"
        }
        func_mappings = {
            "platform_name": "not_a_function"  # Should be callable
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                function_mappings=func_mappings,
                primary_instruction="Greet the user.",
                feedback_instruction="Ensure platform name is included."
            )
        )
        with pytest.raises(TypeError):
            meta_prompt.format_prompt(
                user_name="Charlie",
                platform_name="ChatGPT"
            )

    def test_faulty_function_mapping(self):
        """
        Test that a function mapping raising an exception is handled properly.
        """
        custom_template = "Hello {user}, welcome to {platform}!"
        var_mappings = {
            "user_name": "user",
            "platform_name": "platform"
        }
        func_mappings = {
            "platform_name": faulty_function  # This function raises an error
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                function_mappings=func_mappings,
                primary_instruction="Greet the user.",
                feedback_instruction="Handle platform name appropriately."
            )
        )
        with pytest.raises(ValueError) as exc_info:
            meta_prompt.format_prompt(
                user_name="Dave",
                platform_name="ChatGPT"
            )
        assert "Error in function mapping for 'platform_name': Intentional Error in Function Mapping" in str(exc_info.value)

    # 4. Prompt Formatting Tests

    def test_format_prompt_successful(self):
        """
        Test that format_prompt successfully generates the expected prompt.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a summary.",
        )
        expected = (
            "Create a prompt.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback."
        )
        assert formatted_prompt == expected

    def test_format_prompt_missing_required_fields(self):
        """
        Test that format_prompt raises an error when required fields are missing.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                # Missing primary_instruction and feedback_instruction
            )
        )
        with pytest.raises(ValueError) as exc_info:
            meta_prompt.format_prompt(
                generated_prompt="Generate a summary.",
            )
        assert "Missing required field 'primary_instruction' or it's empty." in str(exc_info.value)

    def test_format_prompt_with_optional_variables(self):
        """
        Test that optional variables are handled correctly during prompt formatting.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nOptional: {optional_var}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "optional_variable": "optional_var"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback.",
                optional_variables=["optional_var"]
            )
        )
        # Providing the optional variable
        formatted_prompt_with_optional = meta_prompt.format_prompt(
            generated_prompt="Generate a summary.",
            optional_variable="This is an optional field."
        )
        expected_with_optional = (
            "Create a prompt.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback.\n"
            "Optional: This is an optional field."
        )
        assert formatted_prompt_with_optional == expected_with_optional

        # Without providing the optional variable
        formatted_prompt_without_optional = meta_prompt.format_prompt(
            generated_prompt="Generate a summary."
        )
        expected_without_optional = (
            "Create a prompt.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback.\n"
            "Optional: "
        )
        assert formatted_prompt_without_optional == expected_without_optional

    # 5. Refinement Mechanism Tests

    def test_refine_prompt_updates_feedback_instruction(self):
        """
        Test that refine_prompt correctly updates the feedback_instruction.
        """
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                primary_instruction="Create a prompt.",
                feedback_instruction="Initial feedback."
            )
        )
        refined_prompt = meta_prompt.refine_prompt(
            feedback="Updated feedback.",
            current_prompt="Generate a report."
        )
        expected = (
            "Create a prompt.\n\n"
            "Generated Prompt:\n"
            "Generate a report.\n\n"
            "Feedback:\n"
            "Updated feedback."
        )
        assert refined_prompt == expected

    def test_refine_prompt_without_current_prompt(self):
        """
        Test that refine_prompt works correctly when current_prompt is not provided.
        """
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                primary_instruction="Create a prompt.",
                feedback_instruction="Initial feedback."
            )
        )
        refined_prompt = meta_prompt.refine_prompt(
            feedback="New feedback without changing the generated prompt."
        )
        expected = (
            "Create a prompt.\n\n"
            "Generated Prompt:\n"
            "\n\n"
            "Feedback:\n"
            "New feedback without changing the generated prompt."
        )
        assert refined_prompt == expected

    # 6. Output Parsing Tests

    def test_output_parser_application(self):
        """
        Test that the output_parser correctly processes the generated prompt.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                output_parser=mock_output_parser,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a summary."
        )
        parsed_output = meta_prompt.output_parser(formatted_prompt)
        expected_output = (
            "CREATE A PROMPT.\n\n"
            "PROMPT: GENERATE A SUMMARY.\n"
            "FEEDBACK: PROVIDE FEEDBACK."
        )
        assert parsed_output == expected_output

    def test_output_parser_not_provided(self):
        """
        Test that the absence of an output_parser does not affect the prompt generation.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a summary."
        )
        # No output parser applied
        assert formatted_prompt == (
            "Create a prompt.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback."
        )

    # 7. Serialization and Deserialization Tests

    def test_serialization_deserialization(self, tmp_path):
        """
        Test that MetaPrompt can be serialized to a file and deserialized back accurately.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        metadata = {"creator": "Tester", "version": "1.0"}
        tags = ["test", "meta-prompt"]
        
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback.",
                metadata=metadata,
                tags=tags
            )
        )
        
        # Serialize to a temporary file
        file_path = tmp_path / "meta_prompt.json"
        meta_prompt.save(file_path)
        
        # Deserialize from the temporary file
        loaded_meta_prompt = MetaPrompt.load(file_path)
        
        # Assertions
        assert loaded_meta_prompt.template == meta_prompt.template
        assert loaded_meta_prompt.variable_mappings == meta_prompt.variable_mappings
        assert loaded_meta_prompt.primary_instruction == meta_prompt.primary_instruction
        assert loaded_meta_prompt.feedback_instruction == meta_prompt.feedback_instruction
        assert loaded_meta_prompt.metadata == meta_prompt.metadata
        assert loaded_meta_prompt.tags == meta_prompt.tags
        
        # Test prompt formatting with loaded instance
        formatted_prompt = loaded_meta_prompt.format_prompt(
            generated_prompt="Generate a summary."
        )
        expected = (
            "Create a prompt.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback."
        )
        assert formatted_prompt == expected

    def test_serialization_with_callable_fields(self, tmp_path):
        """
        Test that serialization handles callable fields appropriately.
        Note: Callable fields like 'output_parser' cannot be serialized directly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                output_parser=mock_output_parser,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        file_path = tmp_path / "meta_prompt_with_callable.json"
        with pytest.raises(TypeError):
            meta_prompt.save(file_path)  # Should raise because 'output_parser' is not serializable

    # 8. Edge Case Tests

    def test_template_with_no_placeholders(self):
        """
        Test that a template without any placeholders still works if no variables are needed.
        """
        custom_template = "This is a static prompt with no variables."
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                primary_instruction="Create a static prompt.",
                feedback_instruction="No feedback needed."
            )
        )
        formatted_prompt = meta_prompt.format_prompt()
        assert formatted_prompt == "This is a static prompt with no variables."

    # def test_template_with_unmapped_placeholders(self):
    #     """
    #     This is not required because Optional Variable set as empty string if not pass into Variable mapping.
    #     Test that having placeholders in the template without corresponding variable mappings raises an error.
    #     """
    #     custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nExtra: {unmapped_placeholder}"
    #     var_mappings = {
    #         "primary_instruction": "instruction",
    #         "generated_prompt": "prompt",
    #         "feedback_instruction": "feedback"
    #     }
    #     meta_prompt = (
    #         PromptFactory
    #         .create_prompt(PromptTechnique.META_PROMPTING)
    #         .configure(
    #             template=custom_template,
    #             variable_mappings=var_mappings,
    #             primary_instruction="Create a prompt.",
    #             feedback_instruction="Provide feedback."
    #         )
    #     )
    #     with pytest.raises(ValueError) as exc_info:
    #         meta_prompt.format_prompt(
    #             generated_prompt="Generate a summary."
    #         )
    #     assert "Missing variable in template: 'unmapped_placeholder'" in str(exc_info.value)

    def test_partial_variables_override(self):
        """
        Test that partial variables can be overridden during prompt formatting.
        """
        def default_user() -> str:
            return "DefaultUser"

        custom_template = "Hello {user}, welcome to {platform}!"
        var_mappings = {
            "user_name": "user",
            "platform_name": "platform"
        }
        partial_variables = {
            "user_name": default_user
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                partial_variables=partial_variables,
                primary_instruction="Greet the user.",
                feedback_instruction="Ensure platform name is included."
            )
        )
        # Without overriding partial variable
        formatted_prompt_default = meta_prompt.format_prompt(
            platform_name="ChatGPT"
        )
        assert formatted_prompt_default == "Hello DefaultUser, welcome to ChatGPT!"

        # Overriding partial variable
        formatted_prompt_override = meta_prompt.format_prompt(
            user_name="Eve",
            platform_name="ChatGPT"
        )
        assert formatted_prompt_override == "Hello Eve, welcome to ChatGPT!"

    def test_handling_of_whitespace_only_inputs(self):
        """
        Test that inputs containing only whitespace are treated as empty.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="   ",  # Whitespace only
                feedback_instruction="\t"  # Whitespace only
            )
        )
        with pytest.raises(ValueError) as exc_info:
            meta_prompt.format_prompt(
                generated_prompt="Generate a summary."
            )
        assert "Missing required field 'primary_instruction' or it's empty." in str(exc_info.value)

    def test_dynamic_template_changes(self):
        """
        Test that changing the template after configuration updates the placeholders correctly.
        """
        original_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        new_template = "{primary_instruction}\n\nNew Prompt: {generated_prompt}\nAdditional Feedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=original_template,
                variable_mappings=var_mappings,
                primary_instruction="Initial instruction.",
                feedback_instruction="Initial feedback."
            )
        )
        # Change the template
        meta_prompt.template = new_template
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Update the prompt format."
        )
        expected = (
            "Initial instruction.\n\n"
            "New Prompt: Update the prompt format.\n"
            "Additional Feedback: Initial feedback."
        )
        assert formatted_prompt == expected

    def test_empty_generated_prompt_field(self):
        """
        Test that an empty generated_prompt field is handled appropriately.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="   "  # Whitespace only
        )
        expected = (
            "Create a prompt.\n\n"
            "Prompt:    \n"
            "Feedback: Provide feedback."
        )
        assert formatted_prompt == expected

    def test_handling_extra_optional_variables(self):
        """
        Test that extra optional variables are handled correctly without affecting required mappings.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nExtra: {extra_var}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback.",
                # 'extra_var' is not mapped, should be added to optional_variables
            )
        )
        assert "extra_var" in meta_prompt.optional_variables

        # Providing the extra_var
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a summary.",
            extra_var="This is an extra optional field."
        )
        expected = (
            "Create a prompt.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback.\n"
            "Extra: This is an extra optional field."
        )
        assert formatted_prompt == expected

    def test_multiple_iterations_of_refinement(self):
        """
        Test that multiple refinements work correctly and sequentially update the prompt.
        """
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                primary_instruction="Create a prompt for writing.",
                feedback_instruction="Include basic structure."
            )
        )
        # First refinement
        refined_prompt_1 = meta_prompt.refine_prompt(
            feedback="Include sections for introduction and conclusion.",
            current_prompt="Write a document."
        )
        expected_1 = (
            "Create a prompt for writing.\n\n"
            "Generated Prompt:\n"
            "Write a document.\n\n"
            "Feedback:\n"
            "Include sections for introduction and conclusion."
        )
        assert refined_prompt_1 == expected_1

        # Second refinement: only pass the new `feedback` and retain the `generated_prompt`
        refined_prompt_2 = meta_prompt.refine_prompt(
            feedback="Ensure the inclusion of detailed methodologies.",
            current_prompt="Write a document."  # The actual `generated_prompt`
        )
        expected_2 = (
            "Create a prompt for writing.\n\n"
            "Generated Prompt:\n"
            "Write a document.\n\n"
            "Feedback:\n"
            "Ensure the inclusion of detailed methodologies."
        )
        assert refined_prompt_2 == expected_2


    def test_handling_empty_feedback_instruction(self):
        """
        Test that an empty feedback_instruction raises an error if it is required.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt."
            )
        )
        
        # Expecting ValueError due to empty feedback_instruction
        with pytest.raises(ValueError, match="Missing required field 'feedback_instruction' or it's empty."):
            meta_prompt.format_prompt(
                generated_prompt="Generate a summary."
            )


    def test_handling_multiple_placeholder_occurrences(self):
        """
        Test that multiple occurrences of the same placeholder are handled correctly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nRepeat: {generated_prompt}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a summary."
        )
        expected = (
            "Create a prompt.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback.\n"
            "Repeat: Generate a summary."
        )
        assert formatted_prompt == expected

    def test_handling_large_number_of_optional_variables(self):
        """
        Test that a large number of optional variables are handled correctly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nOptional1: {opt1}\nOptional2: {opt2}\nOptional3: {opt3}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "optional_var1": "opt1",
            "optional_var2": "opt2",
            "optional_var3": "opt3"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        # Providing all optional variables
        formatted_prompt_all_optional = meta_prompt.format_prompt(
            generated_prompt="Generate a summary.",
            optional_var1="Extra 1",
            optional_var2="Extra 2",
            optional_var3="Extra 3"
        )
        expected_all_optional = (
            "Create a prompt.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback.\n"
            "Optional1: Extra 1\n"
            "Optional2: Extra 2\n"
            "Optional3: Extra 3"
        )
        assert formatted_prompt_all_optional == expected_all_optional

        # Providing only some optional variables
        formatted_prompt_partial_optional = meta_prompt.format_prompt(
            generated_prompt="Generate a summary.",
            optional_var1="Extra 1"
        )
        expected_partial_optional = (
            "Create a prompt.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback.\n"
            "Optional1: Extra 1\n"
            "Optional2: \n"
            "Optional3: "
        )
        assert formatted_prompt_partial_optional == expected_partial_optional

    def test_handle_numeric_and_boolean_variables(self):
        """
        Test that numeric and boolean variables are handled correctly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nCount: {count}\nActive: {active}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "count": "count",
            "active": "active"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt with numeric and boolean variables.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a report.",
            count=5,
            active=True
        )
        expected = (
            "Create a prompt with numeric and boolean variables.\n\n"
            "Prompt: Generate a report.\n"
            "Feedback: Provide feedback.\n"
            "Count: 5\n"
            "Active: True"
        )
        assert formatted_prompt == expected

    def test_handling_special_characters_in_variables(self):
        """
        Test that special characters in variable values are handled correctly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nNotes: {notes}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "notes": "notes"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt with special characters.",
                feedback_instruction="Provide feedback."
            )
        )
        special_notes = "Ensure to handle characters like @, #, $, %, &, *, etc."
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a complex report.",
            notes=special_notes
        )
        expected = (
            "Create a prompt with special characters.\n\n"
            "Prompt: Generate a complex report.\n"
            "Feedback: Provide feedback.\n"
            "Notes: Ensure to handle characters like @, #, $, %, &, *, etc."
        )
        assert formatted_prompt == expected

    def test_handling_none_values_in_optional_variables(self):
        """
        Test that None values in optional variables are handled gracefully.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nOptional: {optional_var}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "optional_var": "optional_var"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt with optional variables.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a summary.",
            optional_var=None
        )
        expected = (
            "Create a prompt with optional variables.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback.\n"
            "Optional: "
        )
        assert formatted_prompt == expected

    def test_handling_duplicate_placeholders(self):
        """
        Test that duplicate placeholders in the template are handled correctly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nRepeat Prompt: {generated_prompt}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a summary."
        )
        expected = (
            "Create a prompt.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback.\n"
            "Repeat Prompt: Generate a summary."
        )
        assert formatted_prompt == expected

    def test_large_template_with_many_placeholders(self):
        """
        Test that a large template with many placeholders is handled correctly.
        """
        custom_template = (
            "{primary_instruction}\n\n"
            "Prompt: {generated_prompt}\n"
            "Feedback: {feedback_instruction}\n"
            "Section1: {sec1}\n"
            "Section2: {sec2}\n"
            "Section3: {sec3}\n"
            "Section4: {sec4}\n"
            "Section5: {sec5}"
        )
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "sec1": "section1",
            "sec2": "section2",
            "sec3": "section3",
            "sec4": "section4",
            "sec5": "section5"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a comprehensive prompt.",
                feedback_instruction="Provide detailed feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate an extensive report.",
            section1="Introduction",
            section2="Methodology",
            section3="Results",
            section4="Discussion",
            section5="Conclusion"
        )
        expected = (
            "Create a comprehensive prompt.\n\n"
            "Prompt: Generate an extensive report.\n"
            "Feedback: Provide detailed feedback.\n"
            "Section1: Introduction\n"
            "Section2: Methodology\n"
            "Section3: Results\n"
            "Section4: Discussion\n"
            "Section5: Conclusion"
        )
        assert formatted_prompt == expected

    def test_handling_non_string_input_variables(self):
        """
        Test that non-string input variables are converted to strings correctly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nNumber: {number}\nBoolean: {boolean}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "number": "number",
            "boolean": "boolean"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt with various data types.",
                feedback_instruction="Ensure all data types are handled."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a summary.",
            number=42,
            boolean=False
        )
        expected = (
            "Create a prompt with various data types.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Ensure all data types are handled.\n"
            "Number: 42\n"
            "Boolean: False"
        )
        assert formatted_prompt == expected

    def test_handling_none_generated_prompt(self):
        """
        Test that passing None as generated_prompt is treated as an empty string.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt=None  # None should be treated as ""
        )
        expected = (
            "Create a prompt.\n\n"
            "Prompt: \n"  # Empty generated_prompt
            "Feedback: Provide feedback."
        )
        assert formatted_prompt == expected


    def test_format_prompt_with_callable_partial_variables(self):
        """
        Test that callable partial_variables are executed correctly during prompt formatting.
        """
        def default_prompt() -> str:
            return "Default generated prompt."

        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        partial_variables = {
            "generated_prompt": default_prompt
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                partial_variables=partial_variables,
                primary_instruction="Create a prompt with default generated prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        # Without overriding the partial variable
        formatted_prompt_default = meta_prompt.format_prompt()
        assert formatted_prompt_default == (
            "Create a prompt with default generated prompt.\n\n"
            "Prompt: Default generated prompt.\n"
            "Feedback: Provide feedback."
        )

        # Overriding the partial variable
        formatted_prompt_override = meta_prompt.format_prompt(
            generated_prompt="Override the generated prompt."
        )
        assert formatted_prompt_override == (
            "Create a prompt with default generated prompt.\n\n"
            "Prompt: Override the generated prompt.\n"
            "Feedback: Provide feedback."
        )

    def test_handling_multiple_variable_mappings(self):
        """
        Test that multiple variable mappings are applied correctly without interference.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nAdditional: {additional_prompt}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "additional_prompt": "additional_prompt"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a primary prompt.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate the main content.",
            additional_prompt="Generate supplementary content."
        )
        expected = (
            "Create a primary prompt.\n\n"
            "Prompt: Generate the main content.\n"
            "Feedback: Provide feedback.\n"
            "Additional: Generate supplementary content."
        )
        assert formatted_prompt == expected

    def test_handling_unicode_characters_in_variables(self):
        """
        Test that Unicode characters in variable values are handled correctly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nEmoji: {emoji}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "emoji": "emoji"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt with emojis.",
                feedback_instruction="Provide feedback."
            )
        )
        emoji_char = "😊🚀✨"
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a creative summary.",
            emoji=emoji_char
        )
        expected = (
            "Create a prompt with emojis.\n\n"
            "Prompt: Generate a creative summary.\n"
            "Feedback: Provide feedback.\n"
            "Emoji: 😊🚀✨"
        )
        assert formatted_prompt == expected

    def test_format_prompt_with_all_possible_fields(self):
        """
        Test that format_prompt works correctly when all possible fields are provided.
        """
        def uppercase_text(text: str) -> str:
            return text.upper()

        custom_template = (
            "{primary_instruction}\n\n"
            "Prompt: {generated_prompt}\n"
            "Feedback: {feedback_instruction}\n"
            "Extra1: {extra1}\n"
            "Extra2: {extra2}"
        )
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "extra_var1": "extra1",
            "extra_var2": "extra2"
        }
        func_mappings = {
            "extra_var1": uppercase_text
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                function_mappings=func_mappings,
                primary_instruction="Create a comprehensive prompt.",
                feedback_instruction="Provide detailed feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate an in-depth analysis.",
            extra_var1="additional info",
            extra_var2="more info"
        )
        expected = (
            "Create a comprehensive prompt.\n\n"
            "Prompt: Generate an in-depth analysis.\n"
            "Feedback: Provide detailed feedback.\n"
            "Extra1: ADDITIONAL INFO\n"
            "Extra2: more info"
        )
        assert formatted_prompt == expected

    def test_format_prompt_with_no_optional_variables_and_no_functions(self):
        """
        Test that format_prompt works correctly when there are no optional variables and no function mappings.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a simple prompt.",
                feedback_instruction="Provide basic feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a short summary."
        )
        expected = (
            "Create a simple prompt.\n\n"
            "Prompt: Generate a short summary.\n"
            "Feedback: Provide basic feedback."
        )
        assert formatted_prompt == expected

    def test_format_prompt_with_complex_variable_values(self):
        """
        Test that complex data structures (e.g., dictionaries, lists) are handled correctly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nData: {data}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "data": "data"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt with complex data.",
                feedback_instruction="Provide feedback."
            )
        )
        complex_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"subkey": "subvalue"}}
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a detailed report.",
            data=complex_data
        )
        expected = (
            "Create a prompt with complex data.\n\n"
            "Prompt: Generate a detailed report.\n"
            "Feedback: Provide feedback.\n"
            "Data: {'key1': 'value1', 'key2': [1, 2, 3], 'key3': {'subkey': 'subvalue'}}"
        )
        assert formatted_prompt == expected

    def test_format_prompt_with_boolean_fields(self):
        """
        Test that boolean fields are correctly handled in prompt formatting.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nActive: {active}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "active": "active"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt with boolean fields.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a user report.",
            active=True
        )
        expected = (
            "Create a prompt with boolean fields.\n\n"
            "Prompt: Generate a user report.\n"
            "Feedback: Provide feedback.\n"
            "Active: True"
        )
        assert formatted_prompt == expected

    def test_format_prompt_with_numeric_and_special_characters(self):
        """
        Test that numeric values and special characters are correctly handled.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nNumber: {number}\nSymbols: {symbols}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "number": "number",
            "symbols": "symbols"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt with numbers and symbols.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a statistical analysis.",
            number=100,
            symbols="@#$%"
        )
        expected = (
            "Create a prompt with numbers and symbols.\n\n"
            "Prompt: Generate a statistical analysis.\n"
            "Feedback: Provide feedback.\n"
            "Number: 100\n"
            "Symbols: @#$%"
        )
        assert formatted_prompt == expected

    def test_format_prompt_with_list_input(self):
        """
        Test that list inputs are converted to strings correctly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nItems: {items}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "items": "items"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt with list inputs.",
                feedback_instruction="Provide feedback."
            )
        )
        items_list = ["Item1", "Item2", "Item3"]
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a to-do list.",
            items=items_list
        )
        expected = (
            "Create a prompt with list inputs.\n\n"
            "Prompt: Generate a to-do list.\n"
            "Feedback: Provide feedback.\n"
            "Items: ['Item1', 'Item2', 'Item3']"
        )
        assert formatted_prompt == expected

    def test_format_prompt_with_nested_placeholders(self):
        """
        Test that nested placeholders in the template are handled correctly.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nNested: {nested_placeholder}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "nested_placeholder": "nested"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a prompt with nested placeholders.",
                feedback_instruction="Provide feedback."
            )
        )
        nested_value = "{another_placeholder}"
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a detailed report.",
            nested_placeholder=nested_value
        )
        expected = (
            "Create a prompt with nested placeholders.\n\n"
            "Prompt: Generate a detailed report.\n"
            "Feedback: Provide feedback.\n"
            "Nested: {another_placeholder}"
        )
        assert formatted_prompt == expected

    def test_format_prompt_with_multiple_data_types(self):
        """
        Test that multiple data types are handled correctly in prompt formatting.
        """
        custom_template = (
            "{primary_instruction}\n\n"
            "Prompt: {generated_prompt}\n"
            "Feedback: {feedback_instruction}\n"
            "Count: {count}\n"
            "Active: {active}\n"
            "Tags: {tags}"
        )
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "count": "count",
            "active": "active",
            "tags": "tags"
        }
        tags_list = ["test", "meta-prompting", "validation"]
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                primary_instruction="Create a comprehensive prompt.",
                feedback_instruction="Provide detailed feedback.",
                tags = tags_list
            )
        )
        
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate an in-depth analysis.",
            count=10,
            active=True,
            tags=tags_list
        )
        expected = (
            "Create a comprehensive prompt.\n\n"
            "Prompt: Generate an in-depth analysis.\n"
            "Feedback: Provide detailed feedback.\n"
            "Count: 10\n"
            "Active: True\n"
            "Tags: ['test', 'meta-prompting', 'validation']"
        )
        assert formatted_prompt == expected

    def test_handling_empty_function_mappings(self):
        """
        Test that empty function_mappings do not interfere with prompt generation.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                function_mappings={},  # Empty function mappings
                primary_instruction="Create a prompt without functions.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a summary."
        )
        expected = (
            "Create a prompt without functions.\n\n"
            "Prompt: Generate a summary.\n"
            "Feedback: Provide feedback."
        )
        assert formatted_prompt == expected

    def test_handle_missing_optional_variables_gracefully(self):
        """
        Test that missing optional variables do not cause errors and are left empty.
        """
        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nOptional1: {optional1}\nOptional2: {optional2}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "optional1": "optional1",
            "optional2": "optional2"
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                optional_variables=["optional1", "optional2"],
                primary_instruction="Create a prompt with optional variables.",
                feedback_instruction="Provide feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(
            generated_prompt="Generate a report."
            # optional1 and optional2 are not provided
        )
        expected = (
            "Create a prompt with optional variables.\n\n"
            "Prompt: Generate a report.\n"
            "Feedback: Provide feedback.\n"
            "Optional1: \n"
            "Optional2: "
        )
        assert formatted_prompt == expected

    def test_handle_callable_partial_variables_exceptions(self):
        """
        Test that exceptions raised by callable partial variables are propagated correctly.
        """
        def faulty_partial_var() -> str:
            raise RuntimeError("Intentional Error in Partial Variable")

        custom_template = "{primary_instruction}\n\nPrompt: {generated_prompt}\nFeedback: {feedback_instruction}\nPartial: {partial_var}"
        var_mappings = {
            "primary_instruction": "instruction",
            "generated_prompt": "prompt",
            "feedback_instruction": "feedback",
            "partial_var": "partial_var"
        }
        partial_variables = {
            "partial_var": faulty_partial_var
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                partial_variables=partial_variables,
                primary_instruction="Create a prompt with faulty partial variables.",
                feedback_instruction="Provide feedback."
            )
        )
        with pytest.raises(RuntimeError) as exc_info:
            meta_prompt.format_prompt(
                generated_prompt="Generate a summary."
            )
        assert str(exc_info.value) == "Intentional Error in Partial Variable"

    def test_kwargs_function_complex_logic(self):
        """
        Test that a **kwargs function with complex logic is handled correctly,
        and required fields are validated.
        """
        def complex_kwargs_func(**kwargs):
            return sum(value for key, value in kwargs.items() if isinstance(value, int))

        custom_template = "{primary_instruction}\n\nTotal: {sum}"
        func_mappings = {
            "sum": complex_kwargs_func
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                function_mappings=func_mappings,
                primary_instruction="Calculate the total.",
                feedback_instruction="Verify correctness."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(a=10, b=20, c="text", d=5)
        expected = (
            "Calculate the total.\n\n"
            "Total: 35"
        )
        assert formatted_prompt == expected
    
    def test_kwargs_function_handles_empty_kwargs(self):
        """
        Test that a **kwargs function gracefully handles no arguments passed,
        and required fields are validated.
        """
        def kwargs_func(**kwargs):
            return "EMPTY" if not kwargs else "NON-EMPTY"

        custom_template = "Status: {status}"
        func_mappings = {
            "status": kwargs_func
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                function_mappings=func_mappings,
                primary_instruction="Ensure all fields are validated.",
                feedback_instruction="Provide validation feedback."
            )
        )
        formatted_prompt = meta_prompt.format_prompt()
        expected = "Status: NON-EMPTY"
        assert formatted_prompt == expected

    def test_required_fields_validation(self):
        """
        Test that missing required fields raise appropriate errors.
        """
        def status_func(**kwargs):
            return "ACTIVE" if kwargs else "INACTIVE"

        custom_template = "Prompt: {primary_instruction}\nFeedback: {feedback_instruction}\nStatus: {status}"
        func_mappings = {
            "status": status_func
        }

        # Missing required fields should raise errors
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                function_mappings=func_mappings
            )
        )

        with pytest.raises(ValueError) as exc_info:
            meta_prompt.format_prompt()
        assert "Missing required field 'primary_instruction'" in str(exc_info.value)
    
    def test_kwargs_function_mixed_data_types(self):
        """
        Test that a **kwargs function correctly handles mixed data types.
        """
        def mixed_data_func(**kwargs):
            # Filter out required fields to handle only user-defined inputs
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["primary_instruction", "feedback_instruction"]}
            return f"Ints: {sum(v for v in filtered_kwargs.values() if isinstance(v, int))}, Strs: {', '.join(k for k, v in filtered_kwargs.items() if isinstance(v, str))}"

        custom_template = "Result: {result}"
        func_mappings = {
            "result": mixed_data_func
        }
        meta_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.META_PROMPTING)
            .configure(
                template=custom_template,
                function_mappings=func_mappings,
                primary_instruction="Handle mixed data types.",
                feedback_instruction="Ensure mixed types are processed."
            )
        )
        formatted_prompt = meta_prompt.format_prompt(a=10, b="text", c=20, d=None, e="example")
        expected = "Result: Ints: 30, Strs: b, e"
        assert formatted_prompt == expected






