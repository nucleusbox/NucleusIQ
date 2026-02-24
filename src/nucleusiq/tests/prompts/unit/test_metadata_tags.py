# tests/test_metadata_tags.py

import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.prompt_composer import PromptComposer


class TestMetadataTags:
    def test_metadata_preservation(self):
        """
        Test that metadata is preserved during serialization and deserialization.
        """
        custom_template = "Meta: {meta}"
        var_mappings = {
            "meta": "meta",
        }

        metadata = {"creator": "Brijesh", "project": "NucleusIQ"}

        # Create PromptComposer with metadata
        composer = PromptFactory.create_prompt(
            PromptTechnique.PROMPT_COMPOSER
        ).configure(
            template=custom_template, variable_mappings=var_mappings, metadata=metadata
        )

        # Serialize to JSON string
        serialized = composer.model_dump_json()

        # Deserialize
        loaded_composer = PromptComposer.model_validate_json(serialized)

        # Assertions
        assert loaded_composer.metadata == composer.metadata

        # Test formatting
        formatted_prompt = loaded_composer.format_prompt(meta="This is a test.")
        expected = "Meta: This is a test."
        assert formatted_prompt.strip() == expected.strip()

    def test_tags_preservation(self):
        """
        Test that tags are preserved during serialization and deserialization.
        """
        custom_template = "Tags: {tags}"
        var_mappings = {
            "tags": "tags",
        }

        tags = ["test", "serialization", "prompt"]

        # Create PromptComposer with tags
        composer = PromptFactory.create_prompt(
            PromptTechnique.PROMPT_COMPOSER
        ).configure(template=custom_template, variable_mappings=var_mappings, tags=tags)

        # Serialize to JSON string
        serialized = composer.model_dump_json()

        # Deserialize
        loaded_composer = PromptComposer.model_validate_json(serialized)

        # Assertions
        assert loaded_composer.tags == composer.tags

        # Test formatting
        formatted_prompt = loaded_composer.format_prompt(tags="unit-test")
        expected = "Tags: unit-test"
        assert formatted_prompt.strip() == expected.strip()

    def test_metadata_and_tags_combined(self):
        """
        Test that both metadata and tags are preserved and correctly formatted.
        """
        custom_template = "Meta: {meta}\nTags: {tags}"
        var_mappings = {
            "meta": "meta",
            "tags": "tags",
        }

        metadata = {"creator": "Brijesh", "project": "NucleusIQ"}
        tags = ["test", "serialization", "prompt"]

        # Create PromptComposer with metadata and tags
        composer = PromptFactory.create_prompt(
            PromptTechnique.PROMPT_COMPOSER
        ).configure(
            template=custom_template,
            variable_mappings=var_mappings,
            metadata=metadata,
            tags=tags,
        )

        # Serialize to JSON string
        serialized = composer.model_dump_json()

        # Deserialize
        loaded_composer = PromptComposer.model_validate_json(serialized)

        # Assertions
        assert loaded_composer.metadata == composer.metadata
        assert loaded_composer.tags == composer.tags

        # Test formatting
        formatted_prompt = loaded_composer.format_prompt(
            meta="This is metadata.", tags="unit-test"
        )
        expected = "Meta: This is metadata.\nTags: unit-test"
        assert formatted_prompt.strip() == expected.strip()

    def test_additional_tags_after_initialization(self):
        """
        Test that tags can be added after initial configuration.
        """
        custom_template = "Tags: {tags}"
        var_mappings = {
            "tags": "tags",
        }

        initial_tags = ["initial"]
        additional_tags = ["additional", "tags"]

        # Create PromptComposer with initial tags
        composer = PromptFactory.create_prompt(
            PromptTechnique.PROMPT_COMPOSER
        ).configure(
            template=custom_template, variable_mappings=var_mappings, tags=initial_tags
        )

        # Update tags
        composer.tags.extend(additional_tags)

        # Format prompt
        formatted_prompt = composer.format_prompt(tags=", ".join(composer.tags))
        expected = "Tags: initial, additional, tags"
        assert formatted_prompt.strip() == expected.strip()

    def test_removing_tags(self):
        """
        Test that tags can be removed correctly.
        """
        custom_template = "Tags: {tags}"
        var_mappings = {
            "tags": "tags",
        }

        tags = ["to_remove", "keep"]

        # Create PromptComposer with tags
        composer = PromptFactory.create_prompt(
            PromptTechnique.PROMPT_COMPOSER
        ).configure(template=custom_template, variable_mappings=var_mappings, tags=tags)

        # Remove a tag
        composer.tags.remove("to_remove")

        # Format prompt
        formatted_prompt = composer.format_prompt(tags=", ".join(composer.tags))
        expected = "Tags: keep"
        assert formatted_prompt.strip() == expected.strip()

    def test_updating_metadata(self):
        """
        Test that metadata can be updated after initial configuration.
        """
        custom_template = "Metadata: {meta}"
        var_mappings = {
            "meta": "meta",
        }

        initial_metadata = {"version": "1.0"}
        updated_metadata = {"version": "2.0", "author": "Brijesh"}

        # Create PromptComposer with initial metadata
        composer = PromptFactory.create_prompt(
            PromptTechnique.PROMPT_COMPOSER
        ).configure(
            template=custom_template,
            variable_mappings=var_mappings,
            metadata=initial_metadata,
        )

        # Update metadata
        composer.metadata.update(updated_metadata)

        # Format prompt
        formatted_prompt = composer.format_prompt(meta="Updated metadata.")
        expected = "Metadata: Updated metadata."
        assert formatted_prompt.strip() == expected.strip()

        # Assert metadata is updated
        assert composer.metadata == updated_metadata

    def test_adding_new_metadata_field(self):
        """
        Test that new metadata fields can be added.
        """
        custom_template = "Metadata: {meta}"
        var_mappings = {
            "meta": "meta",
        }

        metadata = {"creator": "Brijesh"}

        # Create PromptComposer with metadata
        composer = PromptFactory.create_prompt(
            PromptTechnique.PROMPT_COMPOSER
        ).configure(
            template=custom_template, variable_mappings=var_mappings, metadata=metadata
        )

        # Add new metadata field
        composer.metadata["project"] = "NucleusIQ"

        # Format prompt
        formatted_prompt = composer.format_prompt(meta="Project: NucleusIQ")
        expected = "Metadata: Project: NucleusIQ"
        assert formatted_prompt.strip() == expected.strip()

        # Assert metadata is updated
        assert composer.metadata["project"] == "NucleusIQ"
