"""
OpenAI provider for NucleusIQ.

This module provides OpenAI client that supports both async and sync modes.
The mode is determined by the `async_mode` parameter during initialization.

Structured Output Support:
    response = await llm.call(
        model="gpt-4o",
        messages=[...],
        response_format=MyPydanticModel  # Returns validated MyPydanticModel instance
    )
"""

from __future__ import annotations

import os
import asyncio
import time
import logging
import httpx
import json
import dataclasses
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints
from pydantic import BaseModel
import openai
from nucleusiq.core.llms.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class _Choice(BaseModel):
    """Minimal wrapper so we match BaseLLM expectation."""
    message: Dict[str, Any]


class _LLMResponse(BaseModel):
    choices: List[_Choice]


class BaseOpenAI(BaseLLM):
    """
    OpenAI client for ChatCompletion with tool-calling support.
    
    Supports both async and sync modes based on `async_mode` parameter.
    Default is async mode (True).
    """
    
    def _convert_tool_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert generic BaseTool spec to OpenAI format.
        
        Args:
            spec: Generic tool spec from BaseTool.get_spec()
                {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}  # JSON Schema
                }
        
        Returns:
            OpenAI tool spec:
                {
                    "type": "function",
                    "function": {
                        "name": "...",
                        "description": "...",
                        "parameters": {...}  # OpenAI-compatible JSON Schema
                    }
                }
        """
        # Check if it's already in OpenAI format (from OpenAITool)
        if "type" in spec:
            return spec
        
        # Convert generic spec to OpenAI format
        parameters = spec.get("parameters", {})
        # Add additionalProperties: False for OpenAI
        if "additionalProperties" not in parameters:
            parameters = {**parameters, "additionalProperties": False}
        
        return {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": parameters,
            }
        }

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.7,
        logit_bias: Optional[Dict[str, float]] = None,
        async_mode: bool = True,
    ) -> None:
        """
        Initialize OpenAI chat client with sensible defaults.
        
        Args:
            async_mode: If True, uses async client. If False, uses sync client.
                       Default is True (async).
        
        Values can be overridden via arguments or environment variables:
          - OPENAI_API_KEY
          - OPENAI_API_BASE
          - OPENAI_ORG_ID
        """
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.logit_bias = logit_bias
        self.async_mode = async_mode
        self._logger = logging.getLogger("BaseOpenAI")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")

        # Create appropriate client based on mode
        if self.async_mode:
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        else:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

    # ---------- public helpers ----------
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string."""
        import tiktoken
        enc = tiktoken.encoding_for_model(self.model_name)
        return len(enc.encode(text))

    # ---------- BaseLLM impl ----------
    def _uses_max_completion_tokens(self, model: str) -> bool:
        """
        Some newer OpenAI models reject `max_tokens` and require `max_completion_tokens`.

        Example error:
            "Unsupported parameter: 'max_tokens' is not supported with this model.
             Use 'max_completion_tokens' instead."
        """
        m = (model or "").lower()
        return m.startswith("gpt-5")

    def _is_strict_defaults_model(self, model: str) -> bool:
        """
        Some models only support default sampling parameters (e.g., temperature=1)
        and may reject explicit non-default values (or sometimes explicit params at all).
        """
        m = (model or "").lower()
        return m.startswith("gpt-5")

    async def call(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        response_format: Optional[Union[Type[BaseModel], Type, Dict[str, Any]]] = None,
    ) -> Any:
        """
        Call OpenAI API with optional structured output.
        
        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini")
            messages: Chat messages
            tools: Tool definitions for function calling
            tool_choice: Tool choice strategy
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            response_format: Structured output schema. Can be:
                - Pydantic model class: Returns validated model instance
                - Python dataclass: Returns dataclass instance
                - Dict with JSON schema: Returns parsed dict
                - {"type": "json_object"}: Simple JSON mode
                
        Returns:
            If response_format is a Pydantic/dataclass: Validated instance
            Otherwise: Standard LLM response object
            
        Example:
            class Person(BaseModel):
                name: str
                age: int
            
            result = await llm.call(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Extract: John is 30"}],
                response_format=Person
            )
            print(result.name)  # "John"
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,  # Use modern tool_calls format directly
            "stream": stream,
        }

        # Sampling params: some models (e.g., gpt-5-*) only support defaults.
        if not self._is_strict_defaults_model(model):
            payload.update(
                {
                    "temperature": temperature if temperature is not None else self.temperature,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                }
            )

        # Token limit parameter name depends on model family
        if self._uses_max_completion_tokens(model):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        
        # Only include optional parameters if they are provided
        if self.logit_bias is not None:
            payload["logit_bias"] = self.logit_bias
        if tools is not None:
            payload["tools"] = tools
        # `tool_choice` support varies across models/endpoints; we pass it through when provided.
        # If the API rejects it, we handle that with a one-time retry in the error handler.
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if stop is not None:
            payload["stop"] = stop
        
        # Handle structured output via response_format
        output_schema_type = None  # Store the type for parsing response
        if response_format is not None:
            # Handle tuple format: (provider_format_dict, schema_type)
            # This allows passing explicit provider config while still getting parsed response
            if isinstance(response_format, tuple) and len(response_format) == 2:
                provider_format, schema_type = response_format
                payload["response_format"] = provider_format
                if isinstance(schema_type, type):
                    output_schema_type = schema_type
            else:
                openai_response_format = self._build_response_format(response_format)
                if openai_response_format:
                    payload["response_format"] = openai_response_format
                    # Only parse if it's a class (Pydantic/dataclass), not raw dict
                    if isinstance(response_format, type):
                        output_schema_type = response_format

        attempt = 0
        while True:
            try:
                if self.async_mode:
                    # Async mode
                    if stream:
                        async for chunk in self._client.chat.completions.create(**payload):
                            first = chunk.choices[0].delta
                            return _LLMResponse(
                                choices=[_Choice(message=first.model_dump())]
                            )
                    else:
                        resp = await self._client.chat.completions.create(**payload)
                        msg_dict = resp.choices[0].message.model_dump()
                        # If structured output requested, parse and validate
                        if output_schema_type is not None:
                            return self._parse_structured_response(msg_dict, output_schema_type)
                        # Return modern tool_calls format directly (no legacy conversion)
                        return _LLMResponse(choices=[_Choice(message=msg_dict)])
                else:
                    # Sync mode - run in executor to make it async-compatible
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if stream:
                        # For sync streaming, we need to handle it differently
                        chunks = []
                        for chunk in self._client.chat.completions.create(**payload):
                            chunks.append(chunk)
                        if chunks:
                            first = chunks[0].choices[0].delta
                            return _LLMResponse(
                                choices=[_Choice(message=first.model_dump())]
                            )
                    else:
                        resp = await loop.run_in_executor(
                            None,
                            lambda: self._client.chat.completions.create(**payload)
                        )
                        msg_dict = resp.choices[0].message.model_dump()
                        # If structured output requested, parse and validate
                        if output_schema_type is not None:
                            return self._parse_structured_response(msg_dict, output_schema_type)
                        # Return modern tool_calls format directly (no legacy conversion)
                        return _LLMResponse(choices=[_Choice(message=msg_dict)])
            except openai.RateLimitError as e:
                # Rate limit errors - retry with exponential backoff
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error(
                        f"Rate limit exceeded after {self.max_retries} retries: {e}"
                    )
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"Rate limit hit ({e}); retry {attempt}/{self.max_retries} in {backoff}s"
                )
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except openai.APIConnectionError as e:
                # Connection errors - retry with exponential backoff
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error(
                        f"Connection error after {self.max_retries} retries: {e}"
                    )
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"Connection error ({e}); retry {attempt}/{self.max_retries} in {backoff}s"
                )
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except openai.AuthenticationError as e:
                # Authentication errors - don't retry, fail immediately
                self._logger.error(f"Authentication failed: {e}")
                raise ValueError(f"Invalid API key or authentication failed: {e}") from e
            except openai.PermissionDeniedError as e:
                # Permission errors - don't retry, fail immediately
                self._logger.error(f"Permission denied: {e}")
                raise ValueError(f"Permission denied: {e}") from e
            except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
                # Invalid request errors - usually don't retry.
                # But some endpoints/models may reject optional parameters like `tool_choice`.
                # If we sent tool_choice, retry once without it.
                if tool_choice is not None and "tool_choice" in payload:
                    self._logger.warning(f"Invalid request (will retry once without tool_choice): {e}")
                    payload.pop("tool_choice", None)
                    tool_choice = None
                    continue
                self._logger.error(f"Invalid request: {e}")
                raise ValueError(f"Invalid request parameters: {e}") from e
            except openai.APIError as e:
                # API errors (500, 502, 503, etc.) - retry with exponential backoff
                # This must come after all specific APIError subclasses
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error(
                        f"API error after {self.max_retries} retries: {e}"
                    )
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"API error ({e}); retry {attempt}/{self.max_retries} in {backoff}s"
                )
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except httpx.HTTPError as e:
                # HTTP errors (timeouts, network issues) - retry with exponential backoff
                attempt += 1
                if attempt > self.max_retries:
                    self._logger.error(
                        f"HTTP error after {self.max_retries} retries: {e}"
                    )
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"HTTP error ({e}); retry {attempt}/{self.max_retries} in {backoff}s"
                )
                if self.async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
            except Exception as e:
                # Unexpected errors - log and re-raise
                self._logger.error(f"Unexpected error during OpenAI call: {e}", exc_info=True)
                raise

    # ========================================================================
    # STRUCTURED OUTPUT HELPERS
    # ========================================================================
    
    def _build_response_format(
        self, 
        schema: Union[Type[BaseModel], Type, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a schema type to OpenAI's response_format parameter.
        
        Supports:
        - Pydantic models (recommended)
        - Python dataclasses
        - Raw dict with JSON schema
        - Simple {"type": "json_object"} mode
        """
        # Already a raw response_format dict (e.g., {"type": "json_object"})
        if isinstance(schema, dict):
            # Simple JSON mode or explicit response_format - pass through
            return schema
        
        # Pydantic model
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()
            # Clean up schema for OpenAI (remove $defs, add additionalProperties)
            clean_schema = self._clean_schema_for_openai(json_schema)
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": clean_schema
                }
            }
        
        # Dataclass
        if dataclasses.is_dataclass(schema) and isinstance(schema, type):
            json_schema = self._dataclass_to_schema(schema)
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": json_schema
                }
            }
        
        # TypedDict or other types - try to build schema
        if hasattr(schema, '__annotations__'):
            json_schema = self._annotations_to_schema(schema)
            name = getattr(schema, '__name__', 'response')
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": json_schema
                }
            }
        
        self._logger.warning(f"Unknown schema type: {type(schema)}, skipping response_format")
        return None
    
    def _clean_schema_for_openai(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean Pydantic schema for OpenAI structured outputs.
        
        OpenAI strict mode requires:
        - additionalProperties: false
        - ALL properties must be in required array
        - No unsupported keys (title, $schema, default, etc.)
        """
        import copy
        schema = copy.deepcopy(schema)
        
        # OpenAI requires additionalProperties: false for strict mode
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        
        # Inline $defs references FIRST (before cleaning properties)
        defs = schema.pop("$defs", {})
        if defs:
            schema = self._inline_refs(schema, defs)
        
        # Remove unsupported keys
        schema.pop("title", None)
        schema.pop("$schema", None)
        schema.pop("description", None)
        
        # OpenAI strict mode: ALL properties MUST be in required array
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
            for key, prop in schema["properties"].items():
                if isinstance(prop, dict):
                    schema["properties"][key] = self._clean_property(prop, defs)
        
        return schema
    
    def _clean_property(self, prop: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a property schema recursively."""
        import copy
        prop = copy.deepcopy(prop)
        prop.pop("title", None)
        prop.pop("default", None)  # OpenAI strict mode doesn't support defaults
        prop.pop("description", None)
        prop.pop("minimum", None)  # OpenAI doesn't support these in strict mode
        prop.pop("maximum", None)
        prop.pop("minLength", None)
        prop.pop("maxLength", None)
        prop.pop("ge", None)
        prop.pop("le", None)
        
        # Handle anyOf (Optional types)
        if "anyOf" in prop:
            # Keep anyOf as-is but clean each option
            prop["anyOf"] = [self._clean_property(opt, defs) for opt in prop["anyOf"]]
        
        # Handle nested objects
        if prop.get("type") == "object" and "properties" in prop:
            prop["additionalProperties"] = False
            prop["required"] = list(prop["properties"].keys())
            for key, nested in prop["properties"].items():
                if isinstance(nested, dict):
                    prop["properties"][key] = self._clean_property(nested, defs)
        
        # Handle arrays
        if prop.get("type") == "array" and "items" in prop:
            prop["items"] = self._clean_property(prop["items"], defs)
        
        return prop
    
    def _inline_refs(self, obj: Any, defs: Dict[str, Any]) -> Any:
        """Recursively inline $ref references."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in defs:
                        return self._inline_refs(
                            self._clean_schema_for_openai(defs[def_name]), 
                            defs
                        )
            return {k: self._inline_refs(v, defs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._inline_refs(item, defs) for item in obj]
        return obj
    
    def _dataclass_to_schema(self, cls: Type) -> Dict[str, Any]:
        """Convert a dataclass to JSON Schema."""
        hints = get_type_hints(cls)
        fields = dataclasses.fields(cls)
        
        properties = {}
        required = []
        
        for field in fields:
            prop_schema = self._type_to_schema(hints.get(field.name, str))
            properties[field.name] = prop_schema
            
            if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
                required.append(field.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def _annotations_to_schema(self, cls: Type) -> Dict[str, Any]:
        """Convert type annotations to JSON Schema."""
        hints = get_type_hints(cls)
        properties = {}
        required = list(hints.keys())
        
        for name, hint in hints.items():
            properties[name] = self._type_to_schema(hint)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def _type_to_schema(self, type_hint: Type) -> Dict[str, Any]:
        """Convert Python type hint to JSON Schema."""
        from typing import get_origin, get_args, Union
        
        origin = get_origin(type_hint)
        args = get_args(type_hint)
        
        # Handle Optional (Union[X, None])
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                inner = self._type_to_schema(non_none[0])
                return {"anyOf": [inner, {"type": "null"}]}
        
        # Handle List
        if origin is list:
            item_schema = self._type_to_schema(args[0]) if args else {"type": "string"}
            return {"type": "array", "items": item_schema}
        
        # Handle Dict
        if origin is dict:
            return {"type": "object"}
        
        # Basic types
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
        }
        return type_map.get(type_hint, {"type": "string"})
    
    def _parse_structured_response(
        self, 
        message: Dict[str, Any], 
        schema_type: Union[Type[BaseModel], Type, Dict[str, Any]]
    ) -> Any:
        """
        Parse LLM response into the requested structured type.
        
        Returns:
        - Pydantic model instance if schema_type is Pydantic
        - Dataclass instance if schema_type is dataclass
        - Dict if schema_type is dict/TypedDict
        """
        content = message.get("content", "")
        if not content:
            raise ValueError("LLM returned empty content for structured output")
        
        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {e}")
        
        # Pydantic model
        if isinstance(schema_type, type) and issubclass(schema_type, BaseModel):
            return schema_type.model_validate(data)
        
        # Dataclass
        if dataclasses.is_dataclass(schema_type) and isinstance(schema_type, type):
            return schema_type(**data)
        
        # TypedDict or raw dict
        if isinstance(schema_type, dict) or hasattr(schema_type, '__annotations__'):
            return data
        
        return data
