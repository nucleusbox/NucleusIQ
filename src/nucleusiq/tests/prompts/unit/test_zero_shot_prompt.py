# tests/test_zero_shot_prompt.py

import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.zero_shot import ZeroShotPrompt


class TestZeroShotPrompt:
    def test_zero_shot_creation_success(self):
        zero_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            system="You are a helpful assistant.",
            user="Translate the following English text to French: 'Hello, how are you?'",
        )
        prompt_text = zero_shot.format_prompt()
        expected_prompt = (
            "You are a helpful assistant.\n\n"
            "Translate the following English text to French: 'Hello, how are you?'"
        )
        assert prompt_text.strip() == expected_prompt.strip()

    def test_zero_shot_missing_required_fields(self):
        with pytest.raises(ValueError) as exc_info:
            zero_shot = PromptFactory.create_prompt(
                technique=PromptTechnique.ZERO_SHOT
            ).configure(
                user="Translate the following English text to French: 'Hello, how are you?'"
                # 'system' is omitted, but it's required
            )
            zero_shot.format_prompt()
        assert "Missing required field 'system' or it's empty. " in str(exc_info.value)

    def test_zero_shot_with_empty_strings(self):
        zero_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(system="", user="")
        with pytest.raises(ValueError) as exc_info:
            prompt_text = zero_shot.format_prompt()
        assert "Missing required field 'system' or it's empty." in str(exc_info.value)

    def test_zero_shot_invalid_data_types(self):
        with pytest.raises(ValueError) as exc_info:
            zero_shot = PromptFactory.create_prompt(
                technique=PromptTechnique.ZERO_SHOT
            ).configure(
                system=123,  # Should be a string
                user=["Translate", "this"],  # Should be a string
            )
            zero_shot.format_prompt()
        # The exact error message depends on implementation
        assert "system" in str(exc_info.value) or "user" in str(exc_info.value)

    def test_zero_shot_num_clusters_boundary(self):
        with pytest.raises(TypeError) as exc_info:
            zero_shot = PromptFactory.create_prompt(
                technique=PromptTechnique.ZERO_SHOT
            ).configure(
                system="System prompt.",
                user="User prompt.",
                num_clusters=0,  # Assuming 0 is invalid
                max_questions_per_cluster=1,
            )
            zero_shot.format_prompt()
        assert "num_clusters" in str(exc_info.value)

    def test_zero_shot_max_questions_per_cluster_boundary(self):
        with pytest.raises(TypeError) as exc_info:
            zero_shot = PromptFactory.create_prompt(
                technique=PromptTechnique.ZERO_SHOT
            ).configure(
                system="System prompt.",
                user="User prompt.",
                max_questions_per_cluster=-1,  # Invalid negative number
            )
            zero_shot.format_prompt()
        assert "max_questions_per_cluster" in str(exc_info.value)

    def test_zero_shot_serialization_deserialization_json(self, tmp_path):
        zero_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            system="System prompt.",
            user="User prompt.",
            use_cot=True,
            cot_instruction="Please provide detailed reasoning.",
        )

        # Save to JSON
        json_path = tmp_path / "zero_shot.json"
        zero_shot.save(json_path)
        assert json_path.exists()

        # Load from JSON
        loaded_zero_shot = BasePrompt.load(json_path)
        assert isinstance(loaded_zero_shot, ZeroShotPrompt)
        assert loaded_zero_shot.system == "System prompt."
        assert loaded_zero_shot.user == "User prompt."
        assert loaded_zero_shot.use_cot is True
        assert loaded_zero_shot.cot_instruction == "Please provide detailed reasoning."

    def test_zero_shot_serialization_deserialization_yaml(self, tmp_path):
        zero_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            system="System prompt.",
            user="User prompt.",
            use_cot=True,
            cot_instruction="Please provide detailed reasoning.",
        )

        # Save to YAML
        yaml_path = tmp_path / "zero_shot.yaml"
        zero_shot.save(yaml_path)
        assert yaml_path.exists()

        # Load from YAML
        loaded_zero_shot_yaml = BasePrompt.load(yaml_path)
        assert isinstance(loaded_zero_shot_yaml, ZeroShotPrompt)
        assert loaded_zero_shot_yaml.system == "System prompt."
        assert loaded_zero_shot_yaml.user == "User prompt."
        assert loaded_zero_shot_yaml.use_cot is True
        assert (
            loaded_zero_shot_yaml.cot_instruction
            == "Please provide detailed reasoning."
        )

    def test_zero_shot_partial_configuration(self):
        zero_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            user="User prompt."
            # 'system' is omitted
        )
        with pytest.raises(ValueError) as exc_info:
            zero_shot.format_prompt()
        assert "Missing required field 'system' or it's empty." in str(exc_info.value)

    def test_zero_shot_partial_configuration_with_defaults(self):
        zero_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            system="System prompt.",
            user="User prompt.",
            # 'use_cot' and 'cot_instruction' are omitted; should use defaults
        )
        prompt_text = zero_shot.format_prompt()
        expected_prompt = "System prompt.\n\nUser prompt."
        assert prompt_text.strip() == expected_prompt.strip()

    def test_zero_shot_toggle_cot_after_configuration(self):
        zero_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            system="System prompt.",
            user="User prompt.",
            use_cot=False,
            cot_instruction="Custom CoT instruction.",
        )
        prompt_text = zero_shot.format_prompt()
        expected_prompt = (
            "System prompt.\n\nUser prompt."
            # 'cot_instruction' should not be included because use_cot=False
        )
        assert prompt_text.strip() == expected_prompt.strip()

        # Now enable CoT without providing a custom instruction
        zero_shot.configure(
            use_cot=True,
            cot_instruction=None,  # Should default to "Let's think step by step."
        )
        prompt_text_enabled_cot = zero_shot.format_prompt()
        expected_prompt_enabled_cot = (
            "System prompt.\n\nUser prompt.\n\nCustom CoT instruction."
        )
        assert prompt_text_enabled_cot.strip() == expected_prompt_enabled_cot.strip()

    def test_zero_shot_cot_default_instruction(self):
        zero_shot_cot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            system="System prompt.",
            user="User prompt.",
            use_cot=True,
            # 'cot_instruction' is omitted; should default
        )
        prompt_text = zero_shot_cot.format_prompt()
        expected_prompt = "System prompt.\n\nUser prompt.\n\nLet's think step by step."
        assert prompt_text.strip() == expected_prompt.strip()

    def test_zero_shot_multiple_configurations(self):
        zero_shot = PromptFactory.create_prompt(technique=PromptTechnique.ZERO_SHOT)

        # First configuration
        zero_shot.configure(
            system="Initial system prompt.", user="Initial user prompt."
        )

        # Second configuration
        zero_shot.configure(context="Additional context.", use_cot=True)

        # Third configuration
        zero_shot.configure(cot_instruction="Detailed reasoning process.")

        prompt_text = zero_shot.format_prompt()
        expected_prompt = (
            "Initial system prompt.\n\n"
            "Additional context.\n\n"
            "Initial user prompt.\n\n"
            "Detailed reasoning process."
        )
        assert prompt_text.strip() == expected_prompt.strip()

    def test_zero_shot_conflicting_configurations(self):
        zero_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            system="System prompt.",
            user="User prompt.",
            use_cot=True,
            cot_instruction="Initial CoT instruction.",
        )

        # Attempt to configure with use_cot=False and a cot_instruction
        zero_shot.configure(use_cot=False, cot_instruction="This should be ignored.")

        prompt_text = zero_shot.format_prompt()
        expected_prompt = (
            "System prompt.\n\nUser prompt."
            # 'cot_instruction' should be ignored because use_cot=False
        )
        assert prompt_text.strip() == expected_prompt.strip()
