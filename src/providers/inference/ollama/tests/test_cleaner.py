"""Schema cleaner."""

from __future__ import annotations

from nucleusiq_ollama.structured_output.cleaner import clean_schema_for_ollama


def test_clean_schema_inlines_ref() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "Sub": {
                "type": "object",
                "properties": {"k": {"type": "string"}},
            }
        },
        "properties": {
            "item": {"$ref": "#/$defs/Sub"},
        },
    }
    out = clean_schema_for_ollama(schema)
    assert "item" in out.get("properties", {})
