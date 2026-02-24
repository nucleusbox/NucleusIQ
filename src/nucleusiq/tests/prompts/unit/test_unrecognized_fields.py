# tests/test_unrecognized_fields.py

import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique


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
        composer = PromptFactory.create_prompt(
            PromptTechnique.PROMPT_COMPOSER
        ).configure(template=custom_template, variable_mappings=var_mappings)

        # Attempt to configure with an unrecognized field
        with pytest.raises(ValueError) as exc:
            composer.configure(unrecognized_field="some_value")

        assert "Field 'unrecognized_field' is not recognized" in str(exc.value)
