# src/nucleusiq/agents/structured_output/parser.py
"""
Schema parsing and validation utilities for NucleusIQ.
"""

from __future__ import annotations

import json
import re
import dataclasses
import sys
from typing import (
    Any, Dict, List, Optional, Type, TypeVar, Union,
    get_type_hints, get_origin, get_args,
)

from pydantic import BaseModel, ValidationError

from .types import SchemaType
from .errors import SchemaValidationError, SchemaParseError

T = TypeVar("T")

# TypedDict detection
if sys.version_info >= (3, 11):
    from typing import is_typeddict
else:
    try:
        from typing_extensions import is_typeddict
    except ImportError:
        def is_typeddict(tp: Type) -> bool:
            return (
                hasattr(tp, '__annotations__') and
                hasattr(tp, '__total__') and
                hasattr(tp, '__required_keys__')
            )


# ============================================================================
# TYPE DETECTION
# ============================================================================

def is_pydantic(schema: Type) -> bool:
    """Check if type is a Pydantic BaseModel."""
    try:
        return isinstance(schema, type) and issubclass(schema, BaseModel)
    except TypeError:
        return False


def is_dataclass(schema: Type) -> bool:
    """Check if type is a dataclass."""
    return dataclasses.is_dataclass(schema) and isinstance(schema, type)


def is_typed_dict(schema: Type) -> bool:
    """Check if type is a TypedDict."""
    return is_typeddict(schema)


def is_json_schema(schema: Any) -> bool:
    """Check if value is a JSON Schema dict."""
    return isinstance(schema, dict) and "type" in schema


# ============================================================================
# SCHEMA TO JSON CONVERSION
# ============================================================================

def schema_to_json(
    schema: SchemaType,
    *,
    strict: bool = False,
    for_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a schema to JSON Schema format.
    
    Args:
        schema: Pydantic model, dataclass, TypedDict, or JSON Schema dict
        strict: Apply strict mode constraints
        for_provider: Optimize for specific provider (openai, anthropic)
        
    Returns:
        JSON Schema dictionary
    """
    if isinstance(schema, dict):
        # Already JSON Schema
        result = _clean_schema(schema, strict=strict, for_provider=for_provider)
    elif is_pydantic(schema):
        result = _pydantic_to_json(schema, strict=strict, for_provider=for_provider)
    elif is_dataclass(schema):
        result = _dataclass_to_json(schema, strict=strict, for_provider=for_provider)
    elif is_typed_dict(schema):
        result = _typeddict_to_json(schema, strict=strict, for_provider=for_provider)
    elif hasattr(schema, '__annotations__'):
        result = _annotations_to_json(schema, strict=strict, for_provider=for_provider)
    else:
        raise ValueError(f"Cannot convert {type(schema)} to JSON Schema")
    
    return result


def _pydantic_to_json(
    schema: Type[BaseModel],
    *,
    strict: bool = False,
    for_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert Pydantic model to JSON Schema."""
    json_schema = schema.model_json_schema()
    return _clean_schema(json_schema, strict=strict, for_provider=for_provider)


def _dataclass_to_json(
    schema: Type,
    *,
    strict: bool = False,
    for_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert dataclass to JSON Schema."""
    hints = get_type_hints(schema)
    fields = dataclasses.fields(schema)
    
    properties = {}
    required = []
    
    for f in fields:
        prop = _type_to_json(hints.get(f.name, str))
        properties[f.name] = prop
        
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)
    
    result = {
        "type": "object",
        "properties": properties,
    }
    
    if required:
        result["required"] = required
    
    return _clean_schema(result, strict=strict, for_provider=for_provider)


def _typeddict_to_json(
    schema: Type,
    *,
    strict: bool = False,
    for_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert TypedDict to JSON Schema."""
    hints = get_type_hints(schema)
    required_keys = getattr(schema, '__required_keys__', frozenset())
    
    properties = {}
    required = []
    
    for name, hint in hints.items():
        properties[name] = _type_to_json(hint)
        if name in required_keys:
            required.append(name)
    
    result = {
        "type": "object", 
        "properties": properties,
    }
    
    if required:
        result["required"] = required
    
    return _clean_schema(result, strict=strict, for_provider=for_provider)


def _annotations_to_json(
    schema: Type,
    *,
    strict: bool = False,
    for_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert type annotations to JSON Schema."""
    hints = get_type_hints(schema)
    
    properties = {name: _type_to_json(hint) for name, hint in hints.items()}
    
    result = {
        "type": "object",
        "properties": properties,
        "required": list(hints.keys()),
    }
    
    return _clean_schema(result, strict=strict, for_provider=for_provider)


def _type_to_json(type_hint: Type) -> Dict[str, Any]:
    """Convert Python type to JSON Schema type."""
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    
    # None
    if type_hint is type(None):
        return {"type": "null"}
    
    # Optional[X] -> anyOf[X, null]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _type_to_json(non_none[0])
            return {"anyOf": [inner, {"type": "null"}]}
        return {"anyOf": [_type_to_json(a) for a in args]}
    
    # List[X]
    if origin is list:
        items = _type_to_json(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": items}
    
    # Dict[K, V]
    if origin is dict:
        return {"type": "object"}
    
    # Literal["a", "b"]
    try:
        from typing import Literal
        if origin is Literal:
            return {"type": "string", "enum": list(args)}
    except ImportError:
        pass
    
    # Nested types
    if is_pydantic(type_hint):
        return type_hint.model_json_schema()
    if is_dataclass(type_hint):
        return _dataclass_to_json(type_hint)
    
    # Basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "format": "binary"},
    }
    
    return type_map.get(type_hint, {"type": "string"})


def _clean_schema(
    schema: Dict[str, Any],
    *,
    strict: bool = False,
    for_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Clean and transform JSON Schema for target provider."""
    import copy
    schema = copy.deepcopy(schema)
    
    # Inline $defs references
    defs = schema.pop("$defs", {})
    if defs:
        schema = _inline_refs(schema, defs)
    
    # Remove metadata keys
    for key in ["title", "$schema", "description"]:
        schema.pop(key, None)
    
    # Provider-specific transformations
    if for_provider == "openai":
        # OpenAI strict mode: additionalProperties=false, all fields required
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
            for key, prop in schema["properties"].items():
                schema["properties"][key] = _clean_property(prop, for_provider="openai")
    
    if strict:
        schema["additionalProperties"] = False
    
    return schema


def _clean_property(prop: Dict[str, Any], *, for_provider: Optional[str] = None) -> Dict[str, Any]:
    """Clean a property schema."""
    import copy
    prop = copy.deepcopy(prop)
    
    # Remove unsupported keys
    for key in ["title", "default", "description", "minimum", "maximum", 
                "minLength", "maxLength"]:
        prop.pop(key, None)
    
    if "anyOf" in prop:
        prop["anyOf"] = [_clean_property(opt, for_provider=for_provider) for opt in prop["anyOf"]]
    
    if prop.get("type") == "object" and "properties" in prop:
        if for_provider == "openai":
            prop["additionalProperties"] = False
            prop["required"] = list(prop["properties"].keys())
        for key, nested in prop["properties"].items():
            prop["properties"][key] = _clean_property(nested, for_provider=for_provider)
    
    if prop.get("type") == "array" and "items" in prop:
        prop["items"] = _clean_property(prop["items"], for_provider=for_provider)
    
    return prop


def _inline_refs(obj: Any, defs: Dict[str, Any]) -> Any:
    """Inline $ref references."""
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref = obj["$ref"]
            if ref.startswith("#/$defs/"):
                name = ref.split("/")[-1]
                if name in defs:
                    return _inline_refs(defs[name], defs)
        return {k: _inline_refs(v, defs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_inline_refs(item, defs) for item in obj]
    return obj


# ============================================================================
# PARSING & VALIDATION
# ============================================================================

def extract_json(text: str) -> str:
    """
    Extract JSON from text that may contain markdown or other content.
    
    Handles:
    - JSON in ```json code blocks
    - Raw JSON objects/arrays
    - JSON mixed with text
    """
    # Try code blocks first
    code_block = re.findall(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE)
    for match in code_block:
        match = match.strip()
        if match.startswith(("{", "[")):
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
    
    # Try raw JSON
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        for match in re.findall(pattern, text):
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
    
    # Try whole text
    try:
        json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        raise SchemaParseError(
            f"Could not extract valid JSON from response",
            raw_output=text[:500],
        )


def parse_schema(text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response text.
    
    Args:
        text: Raw LLM response (may contain markdown, etc.)
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        SchemaParseError: If JSON cannot be parsed
    """
    json_str = extract_json(text)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise SchemaParseError(
            str(e),
            raw_output=json_str,
            parse_position=e.pos,
        )


def validate_output(
    data: Any,
    schema: SchemaType,
    *,
    schema_name: Optional[str] = None,
) -> Any:
    """
    Validate data against a schema and return typed instance.
    
    Args:
        data: Raw data (dict or JSON string)
        schema: Target schema type
        schema_name: Name for error messages
        
    Returns:
        Validated instance (Pydantic model, dataclass, dict)
        
    Raises:
        SchemaValidationError: If validation fails
    """
    if isinstance(data, str):
        data = parse_schema(data)
    
    schema_name = schema_name or (
        schema.get("title", "Schema") if isinstance(schema, dict)
        else getattr(schema, "__name__", "Schema")
    )
    
    try:
        if is_pydantic(schema):
            return schema.model_validate(data)
        
        if is_dataclass(schema):
            return schema(**data)
        
        # TypedDict and JSON Schema return dict
        return data
        
    except ValidationError as e:
        errors = e.errors()
        field_errors = [
            {"field": ".".join(str(x) for x in err["loc"]), "error": err["msg"]}
            for err in errors
        ]
        raise SchemaValidationError(
            "; ".join(f"{e['field']}: {e['error']}" for e in field_errors),
            schema_name=schema_name,
            raw_output=data,
            field_errors=field_errors,
        )
    except TypeError as e:
        raise SchemaValidationError(
            str(e),
            schema_name=schema_name,
            raw_output=data,
        )
    except Exception as e:
        raise SchemaValidationError(
            f"Validation failed: {e}",
            schema_name=schema_name,
            raw_output=data,
        )

