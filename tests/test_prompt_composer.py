# tests/test_prompt_composer.py

import pytest
import os
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.base import BasePrompt

class TestPromptComposer:

    # 1) Basic usage with variable mappings
    def test_variable_mappings_basic(self):
        custom_template = "System: {sys}\nUser: {usr}\nQuery: {qry}"
        var_mappings = {
            "system": "sys",
            "user_query": "usr",
            "questions": "qry",
        }
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=custom_template,
                variable_mappings=var_mappings
            )
        )
        # Now format
        result = composer.format_prompt(
            system="System prompt here",
            user_query="User says hello",
            questions="Any question"
        )
        expected = "System: System prompt here\nUser: User says hello\nQuery: Any question"
        assert result.strip() == expected.strip()

    # 2) Missing required variables
    def test_missing_required_variables(self):
        custom_tmpl = "Hello {name}, you said: {msg}"
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(template=custom_tmpl)
        )
        # Suppose we define name/msg as required
        composer.input_variables = ["name", "msg"]

        with pytest.raises(ValueError) as exc:
            composer.format_prompt(name="Alice")
            # 'msg' is missing => error
        assert "Missing required field 'msg' or it's empty." in str(exc.value)

    # 3) Using function mappings
    def test_function_mappings(self):
        custom_tmpl = "Tasks:\n{tasks}\n\nUser: {user_query}"

        def bullet_points(**kwargs):
            lines = kwargs["tasks"].split("\n")
            return "\n".join("- " + line for line in lines)

        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=custom_tmpl,
                function_mappings={"tasks": bullet_points}
            )
        )
        prompt_text = composer.format_prompt(
            tasks="Wash dishes\nClean house",
            user_query="Please get them done soon."
        )
        expected = "Tasks:\n- Wash dishes\n- Clean house\n\nUser: Please get them done soon."
        assert prompt_text.strip() == expected.strip()

    # 4) Invalid variable mappings referencing a field not used in the template
    def test_invalid_variable_mappings(self):
        custom_tmpl = "Greeting: {hello}"
        # 'foo' does not appear in input_variables or template placeholders
        var_mappings = {"foo": "bar"}

        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=custom_tmpl,
                variable_mappings=var_mappings
            )
        )
        with pytest.raises(ValueError) as exc:
            # The template needs 'hello' => we never supply it => KeyError => ValueError
            composer.format_prompt()
        err = str(exc.value)
        assert "Missing variable in template: 'hello'" in err

    # 5) Serialization & Deserialization
    def test_serialization_deserialization(self, tmp_path):
        custom_tmpl = "System says: {sys}, user says: {usr}"
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=custom_tmpl,
                variable_mappings={"system": "sys", "user_query": "usr"}
            )
        )
        # Save to JSON
        file_path = tmp_path / "composer_test.json"
        composer.save(str(file_path))
        # Load back
        loaded = BasePrompt.load(str(file_path))

        prompt_text = loaded.format_prompt(system="SystemHere", user_query="UserHere")
        expected = "System says: SystemHere, user says: UserHere"
        assert prompt_text.strip() == expected.strip()


    # 7) Empty template => base class refuses
    def test_empty_template(self):
        composer = PromptFactory.create_prompt(PromptTechnique.PROMPT_COMPOSER)
        with pytest.raises(ValueError) as exc:
            # Attempt to format without specifying a non-empty template
            composer.format_prompt()
        assert "Template cannot be empty" in str(exc.value)

    # 8) Non-string template => cause base class error
    def test_non_string_template(self):
        composer = PromptFactory.create_prompt(PromptTechnique.PROMPT_COMPOSER)
        with pytest.raises(ValueError) as exc:
            # Not a string => base class complains
            composer.configure(template=123)  
        # Might be "Template cannot be empty." or a type error message
        err = str(exc.value)
        assert "Template cannot be empty." in err or "must be a string" in err

    # 9) No input vars => if template placeholders are missing => error
    def test_no_input_vars_missing_placeholder(self):
        tmpl = "Welcome {who}"
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(template=tmpl)
        )
        with pytest.raises(ValueError) as exc:
            composer.format_prompt()  # 'who' not supplied
        assert "Missing variable in template: 'who'" in str(exc.value)

    # 10) No variable/function mappings => direct placeholders
    def test_no_mappings_direct_placeholders(self):
        tmpl = "Hello, {name}. You are {role}."
        composer = PromptFactory.create_prompt(PromptTechnique.PROMPT_COMPOSER).configure(template=tmpl)
        result = composer.format_prompt(name="Alice", role="Engineer")
        expected = "Hello, Alice. You are Engineer."
        assert result.strip() == expected

    # 11) Conflicting variable mappings
    def test_conflicting_variable_mappings(self):
        tmpl = "Field1: {f1}, Field2: {f2}"
        var_mappings = {
            "fieldA": "f1",
            "fieldB": "f1"  # both mapped to same placeholder => overshadow
        }
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=tmpl,
                variable_mappings=var_mappings
            )
        )
        with pytest.raises(ValueError) as exc:
            composer.format_prompt(fieldA="ValA", fieldB="ValB")
        assert "Missing variable in template: 'f2'" in str(exc.value)
        
    # 12) Configure with multiple fields
    def test_configure_multiple_fields(self):
        composer = PromptFactory.create_prompt(PromptTechnique.PROMPT_COMPOSER)
        composer.configure(
            template="Here: {one}, {two}, {three}",
            system="System???",
            examples="Ex??",
            chain_of_thought="COT??",
            user_query="User??"
        )
        result = composer.format_prompt(one="1", two="2", three="3")
        assert "System???" not in result  # not in template
        assert result.strip() == "Here: 1, 2, 3"

    # 13) Configure with an unrecognized field => error
    def test_configure_unrecognized_field(self):
        composer = PromptFactory.create_prompt(PromptTechnique.PROMPT_COMPOSER)
        with pytest.raises(ValueError) as exc:
            composer.configure(
                template="My tmpl",
                non_existent="???"
            )
        assert "Field 'non_existent' is not recognized" in str(exc.value)

    # 14) Function mappings referencing missing fields
    def test_function_mapping_missing_field(self):
        def title_case(**kwargs):
            val = kwargs["missing_key"]  # We'll attempt to read something that doesn't exist
            return val.title()

        composer = PromptFactory.create_prompt(PromptTechnique.PROMPT_COMPOSER).configure(
            template="Hello {greet}",
            function_mappings={"greet": title_case}
        )
        with pytest.raises(ValueError) as exc:
            composer.format_prompt()
        assert "Missing variable in template: 'greet'" in str(exc.value)

    # 15) Successful usage with both variable & function mappings
    def test_variable_and_function_mappings(self):
        def exclaim(**kwargs):
            return kwargs["orig"] + "!!!"

        custom_template = "System: {sys}\nExamples: {ex}\nQuery: {q}"
        composer = (
            PromptFactory
            .create_prompt(PromptTechnique.PROMPT_COMPOSER)
            .configure(
                template=custom_template,
                variable_mappings={"system": "sys", "examples": "ex", "user_query": "q"},
                function_mappings={"examples": exclaim}
            )
        )
        result = composer.format_prompt(system="SysText", user_query="UserText", orig="BaseExamples")
        # 'examples' => ex => exclaim => "BaseExamples!!!"
        expected = "System: SysText\nExamples: BaseExamples!!!\nQuery: UserText"
        assert result.strip() == expected.strip()
