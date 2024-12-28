# src/nucleusiq/prompts/meta_prompt.py

from typing import Optional, Callable, Dict, List, Any
from pydantic import Field, ConfigDict, field_validator, model_validator
import string

from nucleusiq.prompts.base import BasePrompt  # Ensure this import is correct


class MetaPrompt(BasePrompt):
    """
    Implements Meta-Prompting for dynamic prompt generation.
    Focuses on creating, evaluating, and refining task-specific prompts.
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True
    )

    # Specific fields for Meta-Prompting
    primary_instruction: Optional[str] = Field(
        default=None,
        description="Primary instruction for generating a secondary prompt."
    )
    feedback_instruction: Optional[str] = Field(
        default=None,
        description="Instruction for refining the generated prompt based on feedback."
    )
    generated_prompt: Optional[str] = Field(
        default=None,
        description="The prompt generated by the MetaPrompt based on the primary instruction."
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback to refine the generated prompt."
    )

    output_parser: Optional[Callable[[str], Any]] = Field(
        default=None,
        description="Custom parser to process generated output."
    )

    # Inherited fields
    partial_variables: Dict[str, Callable[[], Any]] = Field(
        default_factory=dict,
        description="Default or partial variables that can be overridden."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the prompt."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing and retrieving prompts."
    )

    # New fields: variable_mappings and function_mappings
    variable_mappings: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of logical variable names to template placeholders."
    )
    function_mappings: Dict[str, Callable] = Field(
        default_factory=dict,
        description="Mapping of variables to transformation functions."
    )

    # Override some base fields
    template: str = Field(
        default="{primary_instruction}\n\nGenerated Prompt:\n{generated_prompt}\n\nFeedback:\n{feedback_instruction}",
        description="Template for Meta-Prompting."
    )
    input_variables: List[str] = Field(
        default_factory=lambda: ["primary_instruction", "feedback_instruction"],
        description="Variables required for Meta-Prompting."
    )
    # optional_variables: List[str] = Field(
    #     default_factory=lambda: [],
    #     description="Optional variables for Meta-Prompting."
    # )

    @property
    def technique_name(self) -> str:
        return "meta_prompting"

    #
    # 1) Validate the `template` upon assignment
    #
    @field_validator("template", mode="before")
    def ensure_template_is_valid(cls, v):
        if not isinstance(v, str):
            raise ValueError(f"Template must be a string, got {type(v).__name__}.")
        if not v.strip():
            raise ValueError("Template cannot be empty.")
        return v

    #
    # 2) Automatically detect and handle placeholders
    #
    @model_validator(mode='after')
    def process_template_placeholders(cls, values: "MetaPrompt") -> "MetaPrompt":
        template = values.template or ''
        formatter = string.Formatter()
        placeholders = {fname for _, fname, _, _ in formatter.parse(template) if fname}

        mapped_placeholders = set(values.variable_mappings.values())
        required_placeholders = set(values.input_variables)
        current_optional = set(values.optional_variables)
        additional_optionals = placeholders - mapped_placeholders - required_placeholders - current_optional

        if additional_optionals:
            values.optional_variables.extend(set(additional_optionals))

        return values

    #
    # 3) .configure(...) to set fields, plus variable/function mappings
    #
    def configure(self, **kwargs) -> "MetaPrompt":
        """
        Configure multiple parameters at once.

        Args:
            primary_instruction: Instruction for generating a secondary prompt.
            feedback_instruction: Instruction for refining the generated prompt.
            variable_mappings: Mapping of logical vars to template placeholders.
            function_mappings: Mapping of vars to transformation functions.
            partial_variables: Default or partial variables that can be overridden.
            output_parser: Function to parse the output.
            metadata: Additional metadata for the prompt.
            tags: Tags for categorizing the prompt.

        Returns:
            Self: The updated MetaPrompt instance.
        """
        vmaps = kwargs.pop("variable_mappings", None)
        fmaps = kwargs.pop("function_mappings", None)

        # Set fields directly using setattr
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Handle variable_mappings
        if vmaps is not None:
            if not isinstance(vmaps, dict):
                raise ValueError(f"'variable_mappings' must be a dict, got {type(vmaps).__name__}.")
            self.variable_mappings.update(vmaps)

        # Handle function_mappings
        if fmaps is not None:
            if not isinstance(fmaps, dict):
                raise ValueError(f"'function_mappings' must be a dict, got {type(fmaps).__name__}.")
            self.function_mappings.update(fmaps)

        return self

    #
    # 4) format_prompt() merges partials & instance fields, checks required fields, calls _construct_prompt
    #
    def format_prompt(self, **kwargs) -> str:
        combined_vars = self._merge_partial_variables(**kwargs)

        # Override user-supplied only if instance field is not (None or "")
        for var in self.input_variables + self.optional_variables:
            if hasattr(self, var):
                val = getattr(self, var)
                if val not in (None, ""):
                    combined_vars[var] = val

        # Ensure all function_mappings keys are present in combined_vars and callable
        for func_key, func in self.function_mappings.items():
            if func_key not in combined_vars:
                combined_vars[func_key] = None  # Trigger function mapping
            if not callable(func):  # Check if the function mapping is callable
                raise TypeError(f"Function mapping for '{func_key}' must be callable, got {type(func).__name__}")

        # If template was cleared or empty, raise at runtime
        if not self.template.strip():
            raise ValueError("Template cannot be empty at format time.")

        # Check required input vars
        for req in self.input_variables:
            val = combined_vars.get(req, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ValueError(
                    f"Missing required field '{req}' or it's empty. "
                    f"{self.__class__.__name__} requires that field to be set and non-empty."
                )

        # Run any subclass checks before building
        self._pre_format_validation(combined_vars)

        return self._construct_prompt(**combined_vars)
    
    #
    # 5) _pre_format_validation: optionally check for conflicting placeholders
    #
    def _pre_format_validation(self, combined_vars: Dict[str, Any]) -> None:
        """
        Subclass hook to impose additional checks:
          - For conflicting placeholders in variable_mappings
        """
        # Check for duplicate placeholders
        placeholder_counts = {}
        for logical_var, placeholder in self.variable_mappings.items():
            placeholder_counts[placeholder] = placeholder_counts.get(placeholder, 0) + 1
        duplicates = [p for p, c in placeholder_counts.items() if c > 1]
        if duplicates:
            raise ValueError(f"Conflicting variable mappings, multiple fields map to {duplicates}")

    #
    # 6) _construct_prompt: apply function mappings -> variable mappings -> template
    #
    def _construct_prompt(self, **kwargs) -> str:
        # 1) Apply function mappings
        kwargs = self._apply_function_mappings(kwargs)
        
        # 2) Apply variable mappings
        final_data = kwargs.copy()  # Create a copy of kwargs
        for key, value in self.variable_mappings.items():
            if key in final_data:
                final_data[value] = final_data.pop(key)
            # Add the original variable to final_data as well
            if value in final_data:
                final_data[key] = final_data[value]

        # 3) Handle optional variables
        print("self.optional_variables ", self.optional_variables)
        for opt_var in self.optional_variables:
            if opt_var not in final_data or final_data[opt_var] is None:
                final_data[opt_var] = ""  # Default to empty string

        print("Final Data ", final_data)
        # 3) Try formatting => if missing => raise "Missing variable in template: 'xxx'"
        try:
            return self.template.format(**final_data)  # Use the copied dictionary
        except KeyError as ex:
            raise ValueError(f"Missing variable in template: '{ex.args[0]}'")

    #
    # 7) refine_prompt method for iterative refinement
    #
    def refine_prompt(self, feedback: str, current_prompt: Optional[str] = None) -> str:
        """
        Refines the generated prompt using provided feedback.

        Args:
            feedback (str): Feedback to improve the generated prompt.
            current_prompt (Optional[str]): The current generated prompt to be refined.

        Returns:
            str: The refined prompt.
        """
        self.feedback = feedback  # Update the feedback attribute
        self.feedback_instruction = feedback
        if current_prompt:
            self.generated_prompt = current_prompt
        return self.format_prompt()
    
    
    def _apply_function_mappings(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies function mappings to the provided variables.

        Args:
            kwargs (Dict[str, Any]): Input variables to process.

        Returns:
            Dict[str, Any]: Updated variables after applying function mappings.
        """
        import inspect

        for key, func in self.function_mappings.items():
            if key in kwargs and callable(func):
                try:
                    # Analyze the function's signature
                    signature = inspect.signature(func)
                    parameters = signature.parameters

                    if len(parameters) == 0:
                        # Function expects no arguments
                        kwargs[key] = func()
                    elif len(parameters) == 1 and "kwargs" not in parameters:
                        # Function expects a single argument
                        kwargs[key] = func(kwargs[key])
                    elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                        # Function accepts **kwargs, pass all variables
                        relevant_kwargs = {
                            k: v for k, v in kwargs.items()
                            if k in self.variable_mappings or v is not None
                        }
                        kwargs[key] = func(**relevant_kwargs)
                    else:
                        # Filter required arguments for the function
                        required_args = {
                            param: kwargs[param]
                            for param in parameters
                            if param in kwargs
                        }
                        kwargs[key] = func(**required_args)
                except KeyError as e:
                    raise ValueError(f"Error in function mapping for '{key}': Missing key '{e.args[0]}'") from e
                except Exception as e:
                    raise ValueError(f"Error in function mapping for '{key}': {e}") from e

        return kwargs