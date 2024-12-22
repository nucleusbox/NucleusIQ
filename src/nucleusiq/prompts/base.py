# src/nucleusiq/prompts/base.py

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Self, Callable

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

# Define a TypeVar for format output type
FormatOutputType = TypeVar("FormatOutputType")


class BasePrompt(BaseModel, ABC):
    """
    Abstract base class for all prompt templates in NucleusIQ.
    Provides a unified interface for constructing, validating, serializing,
    and deserializing prompts.
    """

    template: str
    """The prompt template string containing placeholders for variables."""

    input_variables: List[str]
    """A list of the names of the variables required as inputs to the prompt."""

    optional_variables: List[str] = Field(default_factory=list)
    """A list of the names of the variables that are optional in the prompt."""

    partial_variables: Dict[str, Union[Any, Callable[[], Any]]] = Field(default_factory=dict)
    """A dictionary of partial variables that pre-fill parts of the prompt."""

    output_parser: Optional[Any] = Field(default=None, description="Parser to process the output from the LLM.")
    """An optional parser to process the output from the LLM."""

    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata for tracing and logging purposes.")
    """Metadata for tracing and logging purposes."""

    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for categorizing and filtering prompts.")
    """Tags for categorizing and filtering prompts."""

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @property
    @abstractmethod
    def technique_name(self) -> str:
        """
        Returns the name of the prompting technique.
        Must be implemented by subclasses.
        """
        pass

    @field_validator('template')
    def validate_template_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Template cannot be empty.")
        return v

    @field_validator('input_variables', mode='before')
    def validate_input_variables(cls, v):
        if not isinstance(v, list):
            raise ValueError("input_variables must be a list of strings.")
        for var in v:
            if not isinstance(var, str) or not var.isidentifier():
                raise ValueError(f"Invalid input variable name: '{var}'. Must be a valid identifier.")
        return v

    @field_validator('optional_variables', mode='before')
    def validate_optional_variables(cls, v):
        if not isinstance(v, list):
            raise ValueError("optional_variables must be a list of strings.")
        for var in v:
            if not isinstance(var, str) or not var.isidentifier():
                raise ValueError(f"Invalid optional variable name: '{var}'. Must be a valid identifier.")
        return v

    @field_validator('tags', mode='before')
    def validate_tags(cls, v):
        if not isinstance(v, list):
            raise ValueError("tags must be a list of strings.")
        for tag in v:
            if not isinstance(tag, str):
                raise ValueError("Each tag must be a string.")
        return v

    @field_validator('output_parser')
    def validate_output_parser(cls, v):
        if v is not None and not callable(v):
            raise ValueError("output_parser must be a callable.")
        return v

    @abstractmethod
    def _construct_prompt(self, **kwargs) -> str:
        """
        Construct the prompt string by replacing placeholders with actual values.

        Args:
            **kwargs: Variable values to fill into the prompt.

        Returns:
            str: The constructed prompt string.
        """
        pass

    def format_prompt(self, **kwargs) -> str:
        """
        Format the prompt with provided variables, handling partial variables.

        Args:
            **kwargs: Variable values for the prompt.

        Returns:
            str: The formatted prompt string.

        Raises:
            ValueError: If required variables are missing.
        """
        combined_vars = self._merge_partial_variables(**kwargs)

        # Include all input_variables and optional_variables from the instance's attributes
        for var in self.input_variables + self.optional_variables:
            if hasattr(self, var):
                combined_vars[var] = getattr(self, var)

        missing_vars = set(self.input_variables) - set(combined_vars.keys())
        if missing_vars:
            raise ValueError(
                f"Missing variables for prompt: {missing_vars}. "
                f"Received variables: {list(combined_vars.keys())}"
            )

        return self._construct_prompt(**combined_vars)

    def save(self, file_path: Union[Path, str]) -> None:
        """
        Save the prompt template to a file in JSON or YAML format.

        Args:
            file_path (Path or str): The path to save the prompt template.

        Raises:
            ValueError: If the file extension is not supported.
        """
        prompt_dict = self.model_dump()
        prompt_dict["_type"] = self.technique_name  # Use technique_name instead of class name

        save_path = Path(file_path)
        if save_path.suffix == ".json":
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix in {".yaml", ".yml"}:
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

    @classmethod
    def load(cls: Type[BasePrompt], file_path: Union[Path, str]) -> BasePrompt:
        """
        Load a prompt template from a JSON or YAML file.

        Args:
            file_path (Path or str): The path to load the prompt template from.

        Returns:
            BasePrompt: An instance of the loaded prompt template.

        Raises:
            ValueError: If the file format is not supported or if the prompt type is unrecognized.
        """
        load_path = Path(file_path)
        if load_path.suffix == ".json":
            with open(load_path, "r", encoding="utf-8") as f:
                prompt_dict = json.load(f)
        elif load_path.suffix in {".yaml", ".yml"}:
            with open(load_path, "r", encoding="utf-8") as f:
                prompt_dict = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

        prompt_type = prompt_dict.pop("_type", None)
        if not prompt_type:
            raise ValueError("Prompt type '_type' is missing in the prompt file.")

        # Use the PromptFactory to instantiate the correct prompt class
        from .factory import PromptFactory

        prompt_class = PromptFactory.prompt_classes.get(prompt_type.lower())
        if not prompt_class:
            available = ", ".join(PromptFactory.prompt_classes.keys())
            raise ValueError(
                f"Unsupported prompt type '{prompt_type}'. Available techniques: {available}."
            )

        return prompt_class(**prompt_dict)

    def set_metadata(self, metadata: Dict[str, Any]) -> BasePrompt:
        """
        Set metadata for the prompt.

        Args:
            metadata (Dict[str, Any]): Metadata information.

        Returns:
            BasePrompt: The updated prompt instance.
        """
        self.metadata = metadata
        return self

    def add_tags(self, tags: List[str]) -> BasePrompt:
        """
        Add tags to the prompt.

        Args:
            tags (List[str]): List of tags to add.

        Returns:
            BasePrompt: The updated prompt instance.
        """
        self.tags.extend(tags)
        return self

    def set_output_parser(self, parser: Any) -> BasePrompt:
        """
        Set the output parser for the prompt.

        Args:
            parser (Any): A parser object or function.

        Returns:
            BasePrompt: The updated prompt instance.
        """
        self.output_parser = parser
        return self

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePrompt:
        """
        (Deprecated) Return a partial of the prompt template with some variables pre-filled.
        Use specific methods like `set_parameters`, `add_example`, etc.

        Args:
            **kwargs: Partial variables to set.

        Returns:
            BasePrompt: A new prompt template instance with partial variables.
        """
        # Deprecated method; inform the user
        import warnings
        warnings.warn(
            ".partial() is deprecated. Use specific methods like set_parameters(), add_example(), etc.",
            DeprecationWarning
        )
        new_partial_vars = {k: v for k, v in kwargs.items()}
        combined_partial_vars = {**self.partial_variables, **new_partial_vars}

        # Remove input variables that have been pre-filled
        new_input_vars = [var for var in self.input_variables if var not in new_partial_vars]

        return self.model_copy(
            update={
                "input_variables": new_input_vars,
                "partial_variables": combined_partial_vars
            }
        )

    def _merge_partial_variables(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Merge partial variables with user-provided variables.

        Args:
            **kwargs: User-provided variable values.

        Returns:
            Dict[str, Any]: Combined variable values.
        """
        partial_vars = {
            k: (v() if callable(v) else v) for k, v in self.partial_variables.items()
        }
        return {**partial_vars, **kwargs}
