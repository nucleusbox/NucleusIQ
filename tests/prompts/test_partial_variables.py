# tests/test_partial_variables.py

import pytest
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.prompt_composer import PromptComposer

class TestPartialVariables:

    def test_partial_variables_with_callable(self):
        """
        Test that callable partial_variables are executed during prompt formatting.
        """
        custom_template = "Partial1: {partial1}\nPartial2: {partial2}"
        var_mappings = {
            "partial1": "partial1",
            "partial2": "partial2",
        }
        
        # Define callable partial
        def generate_partial2():
            return "Dynamic Value2"
        
        # Create PromptComposer with partial_variables
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                partial_variables={
                    "partial1": "Static Value1",
                    "partial2": generate_partial2
                }
            )
        )
        
        # Format prompt without supplying partials
        formatted_prompt = composer.format_prompt()
        expected = "Partial1: Static Value1\nPartial2: Dynamic Value2"
        assert formatted_prompt.strip() == expected.strip()
    
    def test_partial_variables_overridden_by_kwargs(self):
        """
        Test that partial_variables can be overridden by user-supplied kwargs.
        """
        custom_template = "Partial1: {partial1}\nPartial2: {partial2}"
        var_mappings = {
            "partial1": "partial1",
            "partial2": "partial2",
        }
        
        # Define callable partial
        def generate_partial2():
            return "Dynamic Value2"
        
        # Create PromptComposer with partial_variables
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                partial_variables={
                    "partial1": "Static Value1",
                    "partial2": generate_partial2
                }
            )
        )
        
        # Override partial2 via kwargs
        formatted_prompt = composer.format_prompt(partial2="Overridden Value2")
        expected = "Partial1: Static Value1\nPartial2: Overridden Value2"
        assert formatted_prompt.strip() == expected.strip()
