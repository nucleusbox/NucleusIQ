"""Typed Pydantic models for internal data flow.

These models replace raw ``Dict[str, Any]`` in:
- API payload construction (Chat Completions / Responses API)
- Tool format conversion
- Response normalization input items
- Structured output format configs

External boundaries (``BaseLLM.call`` signature, ``BaseTool.get_spec``)
remain ``dict`` per the framework contract.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nucleusiq_openai._shared.model_config import (
    is_strict_defaults_model,
    uses_max_completion_tokens,
)

# ======================================================================== #
# Tool models                                                               #
# ======================================================================== #


class FunctionToolSpec(BaseModel):
    """Chat Completions function-tool format (nested ``function`` key)."""

    type: str = "function"
    function: dict[str, Any]


class ResponsesFunctionTool(BaseModel):
    """Responses API function-tool format (flat keys)."""

    type: str = "function"
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool = True


# ======================================================================== #
# Responses API input items                                                 #
# ======================================================================== #


class MessageInputItem(BaseModel):
    """User or assistant message sent to the Responses API.

    ``content`` may be a plain string **or** a list of content parts
    (e.g. text + image_url + file) for multimodal inputs.
    """

    role: str
    content: str | list[dict[str, Any]]


class FunctionCallOutput(BaseModel):
    """Tool result sent back to the Responses API."""

    type: str = "function_call_output"
    call_id: str
    output: str


# ======================================================================== #
# Structured output format configs                                          #
# ======================================================================== #


class JsonSchemaFormat(BaseModel):
    """Responses API ``text.format`` for JSON schema mode."""

    type: str = "json_schema"
    name: str = "response"
    strict: bool = True
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")

    model_config = ConfigDict(populate_by_name=True)


class TextFormatConfig(BaseModel):
    """Responses API ``text`` parameter wrapping a format."""

    format: JsonSchemaFormat | dict[str, Any]


# ======================================================================== #
# Chat Completions payload                                                  #
# ======================================================================== #


class ChatCompletionsPayload(BaseModel):
    """Typed payload for ``client.chat.completions.create(**payload)``."""

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    stream_options: dict[str, Any] | None = None

    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    max_tokens: int | None = None
    max_completion_tokens: int | None = None

    logit_bias: dict[str, float] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    stop: list[str] | None = None
    response_format: dict[str, Any] | None = None

    seed: int | None = None
    n: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    parallel_tool_calls: bool | None = None
    modalities: list[str] | None = None
    audio: dict[str, Any] | None = None
    metadata: dict[str, str] | None = None
    store: bool | None = None
    service_tier: str | None = None
    reasoning_effort: str | None = None
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def build(
        cls,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        max_tokens: int = 1024,
        temperature: float | None = None,
        default_temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        logit_bias: dict[str, float] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        **extra: Any,
    ) -> ChatCompletionsPayload:
        """Construct a payload with model-aware parameter mapping."""
        sampling = {}
        if not is_strict_defaults_model(model):
            sampling = {
                "temperature": (
                    temperature if temperature is not None else default_temperature
                ),
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }

        token_key = (
            "max_completion_tokens"
            if uses_max_completion_tokens(model)
            else "max_tokens"
        )

        return cls(
            model=model,
            messages=messages,
            stream=stream,
            **sampling,
            **{token_key: max_tokens},
            logit_bias=logit_bias,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            response_format=response_format,
            **{
                k: v
                for k, v in extra.items()
                if v is not None and k in cls.model_fields
            },
        )

    def to_api_kwargs(self) -> dict[str, Any]:
        """Serialize to kwargs for ``chat.completions.create()``, dropping ``None`` values."""
        return {
            k: v for k, v in self.model_dump(exclude_none=True).items() if v is not None
        }


# ======================================================================== #
# Responses API payload                                                     #
# ======================================================================== #


class ResponsesPayload(BaseModel):
    """Typed payload for ``client.responses.create(**payload)``."""

    model: str
    input: list[MessageInputItem | FunctionCallOutput | dict[str, Any]]

    tools: list[dict[str, Any]] | None = None
    instructions: str | None = None
    previous_response_id: str | None = None

    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None

    tool_choice: Any | None = None
    stream: bool | None = None
    text: TextFormatConfig | dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None

    service_tier: str | None = None
    metadata: dict[str, Any] | None = None
    store: bool | None = None
    truncation: str | None = None
    max_tool_calls: int | None = None
    parallel_tool_calls: bool | None = None
    safety_identifier: str | None = None
    seed: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_api_kwargs(self) -> dict[str, Any]:
        """Serialize to kwargs for ``responses.create()``, dropping ``None`` values."""
        data = {}
        for k, v in self.model_dump(exclude_none=True).items():
            if v is not None:
                data[k] = v
        return data
