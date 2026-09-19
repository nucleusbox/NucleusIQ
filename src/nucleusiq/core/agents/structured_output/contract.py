"""Structured-output **contract**: the schema is the deliverable.

Design (docs/design/AUTONOMOUS_HARNESS_HARDENING.md, WS-8).  When a
user sets ``Agent.response_format`` the framework used to *request* the
schema (``response_format`` on every call) but never *checked* it:
prose that happened to come back was returned as success.  This module
closes that gap with one object every layer consults:

* :meth:`StructuredOutputContract.check` — parse + validate a candidate
  against the schema, returning a :class:`SchemaCheck` with the typed
  value or a retry-ready error message.
* :meth:`StructuredOutputContract.finalizer_instruction` — the
  tools-free "emit only the JSON" instruction used by the structured
  finalizer pass (replaces the prose synthesis pass when a schema is
  set).
* :meth:`StructuredOutputContract.schema_hint` — a compact schema
  description the Critic and Refiner prompts include so they judge and
  revise against the contract, not against "well-structured prose".

Everything here is pure and side-effect free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .parser import parse_schema, schema_to_json, validate_output

_SCHEMA_HINT_MAX_CHARS = 3_000


@dataclass(frozen=True)
class SchemaCheck:
    """Outcome of validating one candidate against the contract."""

    valid: bool
    value: Any = None
    raw: str = ""
    errors: str = ""
    field_errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def canonical_json(self) -> str | None:
        """Compact JSON of the validated value (``None`` when invalid)."""
        if not self.valid:
            return None
        return _to_json(self.value)


def _to_json(value: Any) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), ensure_ascii=False)
    try:
        import dataclasses

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return json.dumps(dataclasses.asdict(value), ensure_ascii=False)
    except Exception:
        pass
    return json.dumps(value, ensure_ascii=False, default=str)


class StructuredOutputContract:
    """Schema-checking facade over a resolved ``OutputSchema``."""

    def __init__(self, output_config: Any) -> None:
        self._config = output_config
        self.schema = output_config.schema
        self.schema_name: str = getattr(output_config, "schema_name", "Schema")
        self.mode: str = str(
            getattr(getattr(output_config, "_resolved_mode", None), "value", "auto")
        )

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def check(self, candidate: Any) -> SchemaCheck:
        """Validate ``candidate`` (text, dict, or typed instance)."""
        raw = candidate if isinstance(candidate, str) else _safe_text(candidate)
        if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
            return SchemaCheck(
                valid=False, raw=raw, errors="Empty output — expected a JSON object."
            )
        if isinstance(candidate, str) and candidate.lstrip().startswith("Error:"):
            return SchemaCheck(valid=False, raw=raw, errors=candidate.strip()[:300])

        # Already-typed instance of the target schema.
        if isinstance(self.schema, type) and isinstance(candidate, self.schema):
            return SchemaCheck(valid=True, value=candidate, raw=raw)

        data: Any = candidate
        try:
            if isinstance(candidate, str):
                data = parse_schema(candidate)
            elif hasattr(candidate, "model_dump"):
                data = candidate.model_dump()
            value = validate_output(data, self.schema, schema_name=self.schema_name)
        except Exception as exc:
            field_errors = tuple(getattr(exc, "field_errors", None) or ())
            message = _format_error(exc)
            return SchemaCheck(
                valid=False, raw=raw, errors=message, field_errors=field_errors
            )
        return SchemaCheck(valid=True, value=value, raw=raw)

    # ------------------------------------------------------------------ #
    # Prompt fragments                                                     #
    # ------------------------------------------------------------------ #

    def schema_hint(self) -> str:
        """Compact JSON Schema for prompts (bounded)."""
        try:
            text = json.dumps(schema_to_json(self.schema), separators=(",", ":"))
        except Exception:
            text = f"<schema {self.schema_name}>"
        if len(text) > _SCHEMA_HINT_MAX_CHARS:
            text = text[: _SCHEMA_HINT_MAX_CHARS - 3] + "..."
        return text

    def critic_criterion(self) -> str:
        return (
            f"**Schema Compliance (REQUIRED)** — The answer MUST be a single JSON "
            f"object valid against the `{self.schema_name}` schema below. Prose, "
            "markdown, summaries, or JSON wrapped in commentary are FAIL. Every "
            "required field must be present with the right type; values must be "
            "supported by the evidence.\n"
            f"Schema: {self.schema_hint()}"
        )

    def refiner_instruction(self) -> str:
        return (
            f"OUTPUT CONTRACT: return ONLY a single JSON object valid against the "
            f"`{self.schema_name}` schema — no prose, no markdown fences, no "
            "commentary before or after. Keep every field the Critic did not "
            "flag; fix the flagged ones.\n"
            f"Schema: {self.schema_hint()}"
        )

    def finalizer_instruction(self, errors: str | None = None) -> str:
        head = (
            "Data gathering is complete. Produce the final deliverable now as a "
            f"single JSON object valid against the `{self.schema_name}` schema. "
            "Output ONLY the JSON — no prose, no markdown fences, no explanation. "
            "Fill every required field from the evidence in this conversation; "
            "use null or an empty list only where the schema allows it and the "
            "evidence is genuinely missing."
        )
        if errors:
            head += f"\n\nYour previous output was rejected by the schema validator:\n{errors[:1_500]}"
        return f"{head}\nSchema: {self.schema_hint()}"

    def retry_message(self, check: SchemaCheck) -> str:
        return (
            "Your previous answer did not satisfy the required output schema "
            f"`{self.schema_name}`:\n{check.errors[:1_500]}\n\n"
            "Return ONLY a single JSON object that validates against the schema. "
            "Do not add prose or markdown."
        )

    # ------------------------------------------------------------------ #
    # Result metadata                                                      #
    # ------------------------------------------------------------------ #

    def describe(self, check: SchemaCheck | None, *, finalizer_runs: int) -> dict:
        return {
            "schema": self.schema_name,
            "mode": self.mode,
            "valid": bool(check.valid) if check is not None else False,
            "finalizer_runs": finalizer_runs,
            "errors": (
                check.errors[:500] if check is not None and check.errors else None
            ),
        }


def _safe_text(value: Any) -> str:
    try:
        return _to_json(value)
    except Exception:
        return str(value)


def _format_error(exc: Exception) -> str:
    fmt = getattr(exc, "format_for_retry", None)
    if callable(fmt):
        try:
            return str(fmt())
        except Exception:
            pass
    return f"{type(exc).__name__}: {exc}"


__all__ = ["SchemaCheck", "StructuredOutputContract"]
