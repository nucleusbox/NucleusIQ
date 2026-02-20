"""
StructuredOutputHandler — centralizes structured-output resolution,
LLM-kwargs generation, and result wrapping.

Extracted from ``Agent`` to satisfy the Single Responsibility Principle.
All execution modes call these through the Agent, which now delegates.
"""

from typing import Any, Dict, Optional

from nucleusiq.llms.base_llm import BaseLLM


class StructuredOutputHandler:
    """Handles structured-output concerns on behalf of an Agent.

    Constructed once per Agent.  Stateless — every call receives the
    current ``response_format``, ``llm``, and related values.
    """

    def resolve_response_format(
        self,
        response_format: Any,
        llm: Optional[BaseLLM],
    ) -> Any:
        """Resolve ``response_format`` to an ``OutputSchema`` (or ``None``)."""
        if response_format is None:
            return None

        from nucleusiq.agents.structured_output import (
            resolve_output_config,
            get_provider_from_llm,
        )

        model_name = (
            getattr(llm, "model_name", "") if llm else ""
        )
        provider = get_provider_from_llm(llm)

        return resolve_output_config(
            response_format,
            model_name=model_name,
            provider=provider,
        )

    def get_call_kwargs(
        self,
        output_config: Any,
        response_format: Any,
        llm: Optional[BaseLLM],
    ) -> Dict[str, Any]:
        """Return LLM call kwargs for structured output (may be empty)."""
        if output_config is None:
            return {}

        from nucleusiq.agents.structured_output import (
            OutputMode,
            OutputSchema,
            get_provider_from_llm,
        )

        if output_config._resolved_mode != OutputMode.NATIVE:
            from nucleusiq.agents.structured_output import OutputMode as OM
            OM.validate_mode(output_config._resolved_mode)
            return {}

        if isinstance(response_format, OutputSchema):
            provider = get_provider_from_llm(llm) or "openai"
            provider_format = output_config.for_provider(provider)
            return {
                "response_format": (
                    provider_format,
                    output_config.schema,
                )
            }
        else:
            return {"response_format": output_config.schema}

    def wrap_result(
        self,
        response: Any,
        output_config: Any,
    ) -> Any:
        """Wrap the LLM response with structured-output metadata."""
        if output_config is None:
            return response

        from nucleusiq.agents.structured_output import OutputMode

        if output_config._resolved_mode == OutputMode.NATIVE:
            if not hasattr(response, "choices"):
                return {
                    "output": response,
                    "schema": output_config.schema_name,
                    "mode": "native",
                }

        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            content = (
                msg.get("content")
                if isinstance(msg, dict)
                else getattr(msg, "content", None)
            )
            return content

        return response
