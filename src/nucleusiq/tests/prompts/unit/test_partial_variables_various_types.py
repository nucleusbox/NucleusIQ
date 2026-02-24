# tests/test_partial_variables_various_types.py

import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from typing import Any, Dict

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique


class TestPartialVariablesVariousTypes:
    def test_partial_variables_with_various_types(self):
        """
        Test that partial_variables can handle different data types.
        """
        custom_template = "String Partial: {str_partial}\nInt Partial: {int_partial}\nCallable Partial: {callable_partial}"
        var_mappings = {
            "str_partial": "str_partial",
            "int_partial": "int_partial",
            "callable_partial": "callable_partial",
        }

        # Define partial variables of different types
        partial_vars = {
            "str_partial": "Static String",
            "int_partial": 42,
            "callable_partial": lambda: "Dynamic Callable",
        }

        # Create PromptComposer instance
        composer = PromptFactory.create_prompt(
            PromptTechnique.PROMPT_COMPOSER
        ).configure(
            template=custom_template,
            variable_mappings=var_mappings,
            partial_variables=partial_vars,
        )

        # Format prompt without supplying any kwargs
        formatted_prompt = composer.format_prompt()
        expected = "String Partial: Static String\nInt Partial: 42\nCallable Partial: Dynamic Callable"
        assert formatted_prompt.strip() == expected.strip()

    def test_output_parser_complex_logic(self):
        """
        Test that output_parser can handle complex parsing logic.
        """
        custom_template = "Data: {data}"
        var_mappings = {
            "data": "data",
        }

        # Define a complex output parser that parses JSON data
        def parse_json_output(output: str) -> Dict[str, Any]:
            import json

            return json.loads(output.split("Data: ")[-1])

        # Create PromptComposer with output_parser
        composer = PromptFactory.create_prompt(
            PromptTechnique.PROMPT_COMPOSER
        ).configure(
            template=custom_template,
            variable_mappings=var_mappings,
            output_parser=parse_json_output,
        )

        # Define JSON data
        json_data = '{"key1": "value1", "key2": "value2"}'

        # Format prompt
        formatted_prompt = composer.format_prompt(data=json_data)

        # Apply output_parser
        parsed_output = composer.output_parser(formatted_prompt)
        expected = {"key1": "value1", "key2": "value2"}
        assert parsed_output == expected
