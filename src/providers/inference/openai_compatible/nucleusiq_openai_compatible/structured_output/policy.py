"""Structured-output policies — resolving the vLLM tools/schema conflict.

vLLM applies constrained decoding for ``response_format``.  When a request
carries **both** ``tools`` and ``response_format`` with
``tool_choice="auto"``, the grammar forces JSON content and the model never
emits tool calls: the response comes back with ``tool_calls: []`` and a JSON
body (vLLM issue #39929).  Grammar composition landed only for the ``hermes``
and ``minimax`` parsers and is skipped when a reasoning parser is attached.
OpenAI cloud does not behave this way.

For an agent framework this is severe — enabling structured output would
silently disable the tool loop, and the agent would look broken for reasons
invisible in the logs.

Three policies, selected by ``structured_output_with_tools=``, expressed as
substitutable strategies rather than an ``if/elif`` chain so a fourth is a
new class and not an edit to existing behavior:

===========  =======================================================
``prompt``   Default. Omit ``response_format``, inject the schema into
             the prompt, validate the final message. Tools keep working
             **and** structured output stays available.
``drop``     Omit ``response_format`` and warn. Matches the current
             ``nucleusiq-groq`` behavior.
``error``    Raise before the HTTP call, for callers who would rather
             fail loudly than degrade.
===========  =======================================================
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from nucleusiq.llms.errors import InvalidRequestError

__all__ = [
    "DropPolicy",
    "ErrorPolicy",
    "PolicyDecision",
    "PromptPolicy",
    "StructuredOutputPolicy",
    "build_policy",
    "render_schema_instruction",
]

_logger = logging.getLogger(__name__)
_PROVIDER = "openai_compatible"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """What to do with a structured-output request for one call.

    Attributes:
        response_format: Value to send as ``response_format``, or ``None``
            to omit the parameter entirely.
        prompt_instruction: Text to append to the system message so the
            model still knows the required shape, or ``None``.
        reason: Human-readable explanation, recorded in call metadata so a
            surprised user can see why the wire payload differs from what
            they asked for.
    """

    response_format: dict[str, Any] | None
    prompt_instruction: str | None
    reason: str


def render_schema_instruction(schema: dict[str, Any]) -> str:
    """Render a JSON schema as a prompt instruction.

    Used whenever the schema cannot travel as ``response_format`` — either
    the server does not implement ``json_schema``, or tools are present and
    the policy is ``prompt``.
    """
    body = json.dumps(schema, indent=2, sort_keys=True)
    return (
        "You must reply with a single JSON object that validates against "
        "this JSON Schema. Output only the JSON — no prose, no markdown "
        "code fences.\n\nJSON Schema:\n"
        f"{body}"
    )


def _native_format(schema: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


@runtime_checkable
class StructuredOutputPolicy(Protocol):
    """Strategy interface for reconciling structured output with tools."""

    @property
    def name(self) -> str:
        """Policy identifier, as accepted by ``structured_output_with_tools``."""
        ...

    def decide(
        self,
        *,
        schema: dict[str, Any],
        schema_name: str,
        has_tools: bool,
        supports_json_schema: bool,
        suppresses_tools: bool,
    ) -> PolicyDecision:
        """Decide how this call should carry its output schema.

        Args:
            schema: The requested JSON schema.
            schema_name: Name to attach to a native ``json_schema`` block.
            has_tools: Whether the request carries any tools.
            supports_json_schema: Whether the server implements native
                ``json_schema`` response format.
            suppresses_tools: Whether this engine's constrained decoding
                suppresses tool calls when both are sent.

        Raises:
            InvalidRequestError: Policy is ``error`` and the combination is
                unsafe.
        """
        ...


class _BasePolicy:
    """Shared handling for the no-tools case, which is policy-independent."""

    def _without_tools(
        self,
        *,
        schema: dict[str, Any],
        schema_name: str,
        supports_json_schema: bool,
    ) -> PolicyDecision:
        if supports_json_schema:
            return PolicyDecision(
                response_format=_native_format(schema, schema_name),
                prompt_instruction=None,
                reason="native json_schema (no tools present)",
            )
        return PolicyDecision(
            response_format={"type": "json_object"},
            prompt_instruction=render_schema_instruction(schema),
            reason=(
                "server does not support json_schema; using json_object plus "
                "a prompt-injected schema"
            ),
        )


class PromptPolicy(_BasePolicy):
    """Keep tools working by moving the schema into the prompt (default).

    The only policy under which a tool-using agent can also produce
    structured output on a vLLM-family server.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        return "prompt"

    def decide(
        self,
        *,
        schema: dict[str, Any],
        schema_name: str,
        has_tools: bool,
        supports_json_schema: bool,
        suppresses_tools: bool,
    ) -> PolicyDecision:
        if not has_tools:
            return self._without_tools(
                schema=schema,
                schema_name=schema_name,
                supports_json_schema=supports_json_schema,
            )
        _logger.debug(
            "Tools present; omitting response_format and injecting the schema "
            "into the prompt to keep tool calling functional."
        )
        return PolicyDecision(
            response_format=None,
            prompt_instruction=render_schema_instruction(schema),
            reason=(
                "tools present; response_format omitted and schema injected "
                "into the prompt so tool calls are not suppressed"
            ),
        )


class DropPolicy(_BasePolicy):
    """Drop the schema entirely when tools are present, with a warning."""

    __slots__ = ()

    @property
    def name(self) -> str:
        return "drop"

    def decide(
        self,
        *,
        schema: dict[str, Any],
        schema_name: str,
        has_tools: bool,
        supports_json_schema: bool,
        suppresses_tools: bool,
    ) -> PolicyDecision:
        if not has_tools:
            return self._without_tools(
                schema=schema,
                schema_name=schema_name,
                supports_json_schema=supports_json_schema,
            )
        message = (
            "Structured output was requested together with tools. "
            "response_format has been dropped because constrained decoding "
            "would suppress tool calls. Use "
            "structured_output_with_tools='prompt' to keep the schema as a "
            "prompt instruction."
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        _logger.warning(message)
        return PolicyDecision(
            response_format=None,
            prompt_instruction=None,
            reason="tools present; response_format dropped per 'drop' policy",
        )


class ErrorPolicy(_BasePolicy):
    """Refuse the request rather than silently changing its semantics."""

    __slots__ = ()

    @property
    def name(self) -> str:
        return "error"

    def decide(
        self,
        *,
        schema: dict[str, Any],
        schema_name: str,
        has_tools: bool,
        supports_json_schema: bool,
        suppresses_tools: bool,
    ) -> PolicyDecision:
        if not has_tools:
            return self._without_tools(
                schema=schema,
                schema_name=schema_name,
                supports_json_schema=supports_json_schema,
            )
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=(
                "Structured output cannot be combined with tools on this "
                "engine: constrained decoding suppresses tool calls, so the "
                "agent's tool loop would silently stop working. Either drop "
                "the tools for this call, or set "
                "structured_output_with_tools='prompt' to carry the schema "
                "as a prompt instruction instead."
            ),
        )


_POLICIES: dict[str, StructuredOutputPolicy] = {
    "prompt": PromptPolicy(),
    "drop": DropPolicy(),
    "error": ErrorPolicy(),
}


def build_policy(mode: str) -> StructuredOutputPolicy:
    """Return the policy registered under *mode*.

    Raises:
        InvalidRequestError: Unknown mode. Validated at construction too, so
            this is a defensive second gate.
    """
    policy = _POLICIES.get(mode)
    if policy is None:
        raise InvalidRequestError.from_provider_error(
            provider=_PROVIDER,
            message=(
                f"structured_output_with_tools must be one of "
                f"{', '.join(sorted(_POLICIES))}; got {mode!r}"
            ),
        )
    return policy
