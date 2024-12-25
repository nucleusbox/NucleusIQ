# tests/test_unrecognized_fields.py

import pytest
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.prompt_composer import PromptComposer

class TestUnrecognizedFields:

    def test_configure_with_unrecognized_field(self):
        """
        Test that configuring with an unrecognized field raises a ValueError.
        """
        custom_template = "Hello {name}"
        var_mappings = {
            "name": "name",
        }
        
        # Create PromptComposer without any extra fields
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings
            )
        )
        
        # Attempt to configure with an unrecognized field
        with pytest.raises(ValueError) as exc:
            composer.configure(unrecognized_field="some_value")
        
        assert "Field 'unrecognized_field' is not recognized" in str(exc.value)
