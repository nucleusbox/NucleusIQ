from collections.abc import Callable
from typing import Any

from nucleusiq.prompts.base import BasePrompt
from pydantic import ConfigDict, Field, field_validator


class PromptComposer(BasePrompt):
    """
    A flexible prompt class that:
      - Uses 'template' with placeholders
      - Maps logical vars -> placeholders via 'variable_mappings'
      - Applies 'function_mappings' to transform fields at runtime
      - Has extra placeholders: 'examples', 'chain_of_thought', 'user_query'
      - Raises an error if a placeholder is missing or a function key is missing
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    # Extended placeholders
    examples: str | None = Field(default=None)
    chain_of_thought: str | None = Field(default=None)
    user_query: str | None = Field(default=None)

    # Mappings
    variable_mappings: dict[str, str] = Field(default_factory=dict)
    function_mappings: dict[str, Callable[..., str]] = Field(default_factory=dict)

    # Overridden base fields
    template: str = Field(default="")
    input_variables: list[str] = Field(default_factory=list)
    optional_variables: list[str] = Field(
        default_factory=lambda: ["system", "examples", "chain_of_thought", "user_query"]
    )

    @property
    def technique_name(self) -> str:
        return "prompt_composer"

    #
    # Validate 'template' at assignment
    #
    @field_validator("template", mode="before")
    def ensure_template_not_empty(cls, v):
        if not isinstance(v, str):
            raise ValueError(f"Template must be a string, got {type(v).__name__}.")
        if not v.strip():
            raise ValueError("Template cannot be empty.")
        return v

    #
    # .configure(...) to set fields, plus variable/function mappings
    #
    def configure(self, **kwargs) -> "PromptComposer":
        vmaps = kwargs.pop("variable_mappings", None)
        fmaps = kwargs.pop("function_mappings", None)

        super().configure(**kwargs)

        from nucleusiq.prompts.errors import PromptConfigError

        if vmaps is not None:
            if not isinstance(vmaps, dict):
                raise PromptConfigError(
                    f"variable_mappings must be a dict, not {type(vmaps).__name__}.",
                    technique=self.__class__.__name__,
                )
            self.variable_mappings = vmaps

        if fmaps is not None:
            if not isinstance(fmaps, dict):
                raise PromptConfigError(
                    f"function_mappings must be a dict, not {type(fmaps).__name__}.",
                    technique=self.__class__.__name__,
                )
            self.function_mappings = fmaps

        return self

    #
    # format_prompt() merges partials, checks required fields, calls _construct_prompt
    #
    def format_prompt(self, **kwargs) -> str:
        merged_vars = self._merge_partial_variables(**kwargs)

        # Override from self if present & non-empty
        for var in self.input_variables + self.optional_variables:
            val = getattr(self, var, None)
            if val not in (None, ""):
                merged_vars[var] = val

        # If user never sets template or if it was cleared => error
        from nucleusiq.prompts.errors import PromptTemplateError

        if not self.template.strip():
            raise PromptTemplateError(
                "Template cannot be empty at format time.",
                technique=self.__class__.__name__,
            )

        # Provide empty string if 'examples' was never set
        if "examples" not in merged_vars or merged_vars["examples"] is None:
            merged_vars["examples"] = ""

        # Check required input vars
        for req in self.input_variables:
            val = merged_vars.get(req, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                raise PromptTemplateError(
                    f"Missing required field '{req}' or it's empty. "
                    f"{self.__class__.__name__} requires that field to be set and non-empty.",
                    technique=self.__class__.__name__,
                )

        self._pre_format_validation(merged_vars)
        return self._construct_prompt(**merged_vars)

    #
    # Optional extra checks (like conflict checking)
    #
    def _pre_format_validation(self, combined_vars: dict[str, Any]) -> None:
        # If you wanted to detect duplicates in variable_mappings, do it here.
        pass

    #
    # The actual final step: apply function mappings -> variable mappings -> template
    #
    def _construct_prompt(self, **kwargs) -> str:
        from nucleusiq.prompts.errors import PromptTemplateError

        # 1) Apply function mappings
        for key, func in self.function_mappings.items():
            if key in kwargs and callable(func):
                try:
                    kwargs[key] = func(**kwargs)
                except KeyError:
                    raise PromptTemplateError(
                        f"Function mapping references missing key: '{key}'",
                        technique=self.__class__.__name__,
                    )

        # 2) Apply variable mappings
        final_data = {}
        for logic_var, val in kwargs.items():
            placeholder = self.variable_mappings.get(logic_var, logic_var)
            final_data[placeholder] = val

        # 3) Try formatting => if missing => raise "Missing variable in template: 'xxx'"
        try:
            return self.template.format(**final_data)
        except KeyError as ex:
            from nucleusiq.prompts.errors import PromptTemplateError

            raise PromptTemplateError(
                f"Missing variable in template: '{ex.args[0]}'",
                technique=self.__class__.__name__,
            ) from ex
