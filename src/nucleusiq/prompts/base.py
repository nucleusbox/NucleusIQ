# src/nucleusiq/prompts/base.py

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Self, Callable

import yaml
from pydantic import BaseModel, Field, model_validator

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

    output_parser: Optional[Any] = None
    """An optional parser to process the output from the LLM."""

    metadata: Optional[Dict[str, Any]] = None
    """Metadata for tracing and logging purposes."""

    tags: Optional[List[str]] = Field(default_factory=list)
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

    @model_validator(mode="before")
    def _check_variable_conflicts(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure that variable names do not conflict with reserved keywords
        and that there are no overlapping input and partial variables.
        """
        reserved = {"stop", "metadata", "tags"}
        input_vars = set(values.get("input_variables", []))
        optional_vars = set(values.get("optional_variables", []))
        partial_vars = set(values.get("partial_variables", {}).keys())

        # Check for reserved keywords
        conflicts = (input_vars | optional_vars | partial_vars) & reserved
        if conflicts:
            raise ValueError(
                f"Variable names {conflicts} are reserved and cannot be used."
            )

        # Check for overlapping input and partial variables
        overlap = input_vars & partial_vars
        if overlap:
            raise ValueError(
                f"Overlapping variables between input and partial variables: {overlap}"
            )

        return values

    @model_validator(mode="after")
    def _validate_template_variables(self) -> Self:
        """
        Validate that all variables in the template are defined in input_variables,
        optional_variables, or partial_variables.
        """
        # Extract variables from the template using regex
        pattern = re.compile(r"\{(\w+)\}")
        variables_in_template = set(pattern.findall(self.template))

        allowed_vars = set(self.input_variables) | set(
            self.optional_variables
        ) | set(self.partial_variables.keys())

        missing_vars = variables_in_template - allowed_vars
        if missing_vars:
            raise ValueError(
                f"Variables {missing_vars} found in template are not defined in "
                f"input_variables, optional_variables, or partial_variables."
            )

        return self

    @abstractmethod
    def construct_prompt(self, **kwargs) -> str:
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

        missing_vars = set(self.input_variables) - set(combined_vars.keys())
        if missing_vars:
            raise ValueError(
                f"Missing variables for prompt: {missing_vars}. "
                f"Received variables: {list(combined_vars.keys())}"
            )

        return self.construct_prompt(**combined_vars)

    def save(self, file_path: Union[Path, str]) -> None:
        """
        Save the prompt template to a file in JSON or YAML format.

        Args:
            file_path (Path or str): The path to save the prompt template.

        Raises:
            ValueError: If the file extension is not supported.
        """
        prompt_dict = self.dict()
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
                f"Unsupported prompt type '{prompt_type}'. Available types: {available}."
            )

        return prompt_class(**prompt_dict)

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePrompt:
        """
        Return a partial of the prompt template with some variables pre-filled.

        Args:
            **kwargs: Partial variables to set.

        Returns:
            BasePrompt: A new prompt template instance with partial variables.
        """
        new_partial_vars = {k: v for k, v in kwargs.items()}
        combined_partial_vars = {**self.partial_variables, **new_partial_vars}

        # Remove input variables that have been pre-filled
        new_input_vars = [var for var in self.input_variables if var not in new_partial_vars]

        return self.copy(
            update={
                "input_variables": new_input_vars,
                "partial_variables": combined_partial_vars
            }
        )
    
    def dict(self, **kwargs: Any) -> dict:
        """
        Return a dictionary representation of the prompt template.

        Args:
            **kwargs: Additional arguments for serialization.

        Returns:
            dict: Dictionary representation of the prompt template.
        """
        prompt_dict = super().dict(**kwargs)
        prompt_dict["_type"] = self.technique_name  # Ensure consistency
        return prompt_dict

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
