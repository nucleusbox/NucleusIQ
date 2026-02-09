# tests/test_output_parser.py

import os
import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.prompt_composer import PromptComposer

class TestOutputParser:

    def test_output_parser_functionality(self):
        """
        Test that the output_parser correctly processes the formatted prompt.
        """
        custom_template = "Echo: {message}"
        var_mappings = {
            "message": "message",
        }
        
        # Define an output parser
        def parse_echo(output: str) -> str:
            # For simplicity, let's extract the message after 'Echo: '
            return output.split("Echo: ")[-1].lower()
        
        # Create PromptComposer with output_parser
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                output_parser=parse_echo
            )
        )
        
        # Format prompt
        formatted_prompt = composer.format_prompt(message="HELLO WORLD")
        # Apply output_parser
        parsed_output = composer.output_parser(formatted_prompt)
        expected = "hello world"
        assert parsed_output == expected
    
    def test_output_parser_with_missing_placeholder(self):
        """
        Test that the output_parser handles missing placeholders gracefully.
        """
        custom_template = "Echo: {message}"
        var_mappings = {
            "message": "message",
        }
        
        # Define an output parser that expects 'message'
        def parse_echo(output: str) -> str:
            return output.split("Echo: ")[-1].lower()
        
        # Create PromptComposer without supplying 'message'
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings,
                output_parser=parse_echo
            )
        )
        
        with pytest.raises(ValueError) as exc:
            formatted_prompt = composer.format_prompt()
            parsed_output = composer.output_parser(formatted_prompt)
        
        assert "Missing variable in template: 'message'" in str(exc.value)
