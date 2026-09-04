"""Normalizing an inbound ``response_format`` into a bare JSON schema.

The core agent's ``StructuredOutputHandler`` hands structured output to the
provider as a ``response_format`` **call kwarg**, in one of several shapes
depending on how the caller declared it.  Without this adapter that kwarg
would flow straight through the generic parameter filter and onto the wire,
bypassing :mod:`~nucleusiq_openai_compatible.structured_output.policy`
entirely — which on vLLM means ``response_format`` and ``tools`` are sent
together and tool calls are silently suppressed.

Recovering the underlying schema lets the policy make the same decision it
would have made for a provider-native ``response_schema=`` argument, so
structured output behaves identically however the caller asked for it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["InboundFormat", "normalize_response_format"]

_DEFAULT_NAME = "response"


class InboundFormat:
    """The outcome of interpreting a ``response_format`` value.

    Exactly one of *schema* and *passthrough* is set.  A *schema* goes to the
    policy for a routing decision; a *passthrough* (such as
    ``{"type": "json_object"}``) carries no schema to reason about and is
    forwarded unchanged.
    """

    __slots__ = ("name", "passthrough", "schema")

    def __init__(
        self,
        *,
        schema: dict[str, Any] | None = None,
        name: str = _DEFAULT_NAME,
        passthrough: dict[str, Any] | None = None,
    ) -> None:
        self.schema = schema
        self.name = name
        self.passthrough = passthrough

    def __repr__(self) -> str:
        if self.schema is not None:
            return (
                f"InboundFormat(schema=<{len(self.schema)} keys>, name={self.name!r})"
            )
        return f"InboundFormat(passthrough={self.passthrough!r})"


def _looks_like_json_schema(value: dict[str, Any]) -> bool:
    """Whether *value* is a bare JSON schema rather than a wire directive."""
    if "properties" in value or "$ref" in value or "anyOf" in value:
        return True
    return value.get("type") == "object"


def normalize_response_format(value: Any) -> InboundFormat | None:
    """Interpret a ``response_format`` call kwarg.

    Handles every shape the core handler and direct callers produce:

    * ``(provider_format, schema)`` — a tuple, emitted when the caller passed
      an ``OutputSchema``; the raw schema is the second element.
    * ``{"type": "json_schema", "json_schema": {"name", "schema"}}`` — the
      OpenAI wire form; the inner schema and name are recovered.
    * ``{"type": "json_object"}`` — no schema, forwarded as-is.
    * A bare JSON schema — used directly.
    * A Pydantic model class — its ``model_json_schema()`` is used.

    Args:
        value: The ``response_format`` argument, in any of the above shapes.

    Returns:
        An :class:`InboundFormat`, or ``None`` if *value* is empty or is not
        a shape this provider can interpret — in which case the caller leaves
        it alone rather than guessing.
    """
    if value is None:
        return None

    if isinstance(value, tuple):
        # (provider_format, schema) from StructuredOutputHandler.get_call_kwargs.
        # Either element can carry the schema depending on the provider string
        # the core resolved, so prefer whichever yields one.
        results = [normalize_response_format(part) for part in reversed(value)]
        with_schema = next(
            (r for r in results if r is not None and r.schema is not None), None
        )
        if with_schema is not None:
            return with_schema
        return next((r for r in results if r is not None), None)

    model_schema = getattr(value, "model_json_schema", None)
    if callable(model_schema):
        try:
            generated = model_schema()
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(generated, dict):
            return InboundFormat(
                schema=generated, name=getattr(value, "__name__", _DEFAULT_NAME)
            )
        return None

    if not isinstance(value, dict) or not value:
        return None

    kind = value.get("type")

    if kind == "json_schema":
        block = value.get("json_schema")
        if isinstance(block, dict):
            inner = block.get("schema")
            if isinstance(inner, dict):
                name = block.get("name")
                return InboundFormat(
                    schema=inner,
                    name=name if isinstance(name, str) and name else _DEFAULT_NAME,
                )
        return InboundFormat(passthrough=value)

    if kind == "json":
        # NucleusIQ's provider-neutral shape from OutputSchema.for_provider().
        # No OpenAI-compatible server understands type="json", so the schema
        # must be recovered rather than forwarded.
        inner = value.get("schema")
        if isinstance(inner, dict):
            return InboundFormat(schema=inner)
        return None

    if kind == "json_object":
        return InboundFormat(passthrough=value)

    if _looks_like_json_schema(value):
        title = value.get("title")
        return InboundFormat(
            schema=value,
            name=title if isinstance(title, str) and title else _DEFAULT_NAME,
        )

    return InboundFormat(passthrough=value)
