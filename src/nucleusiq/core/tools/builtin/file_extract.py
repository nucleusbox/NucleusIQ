"""FileExtractTool — extract structured data from common data files.

Supports CSV, JSON, JSONL/NDJSON, TSV, YAML, XML, and TOML via a
pluggable ``_FORMAT_HANDLERS`` registry.  Uses only the Python standard
library where possible; YAML requires ``PyYAML`` (optional).

Returns a structured summary: headers/keys, row count, sample rows,
and optional query context.  This is the lightweight alternative to
pandas for agent data exploration.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sys
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict

from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.builtin.workspace import (
    WorkspaceSecurityError,
    format_file_size,
    resolve_safe_path,
)

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_ROWS = 5


# ------------------------------------------------------------------ #
# Per-format extraction handlers                                       #
# ------------------------------------------------------------------ #

FormatHandler = Callable[["FileExtractTool", str, str, int, int, str], str]


def _extract_csv(
    tool: FileExtractTool,
    path: str,
    text: str,
    size: int,
    max_sample: int,
    query: str,
) -> str:
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        return f"File: {path} | Empty CSV file."

    headers = rows[0]
    data_rows = rows[1:]
    total = len(data_rows)

    parts: list[str] = [
        f"File: {path} | CSV | {format_file_size(size)}",
        f"Columns ({len(headers)}): {', '.join(headers)}",
        f"Rows: {total}",
    ]

    if query:
        parts.append(f"Query context: {query}")

    if data_rows:
        sample = data_rows[:max_sample]
        parts.append(f"\nSample data (first {len(sample)} rows):")
        col_widths = [len(h) for h in headers]
        for row in sample:
            for i, val in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(val))

        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        parts.append(f"  {header_line}")
        parts.append(f"  {'-+-'.join('-' * w for w in col_widths)}")
        for row in sample:
            padded = [
                (row[i] if i < len(row) else "").ljust(
                    col_widths[i] if i < len(col_widths) else 0
                )
                for i in range(len(headers))
            ]
            parts.append(f"  {' | '.join(padded)}")

    return "\n".join(parts)


def _extract_tsv(
    tool: FileExtractTool,
    path: str,
    text: str,
    size: int,
    max_sample: int,
    query: str,
) -> str:
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    rows = list(reader)

    if not rows:
        return f"File: {path} | Empty TSV file."

    headers = rows[0]
    data_rows = rows[1:]
    total = len(data_rows)

    parts: list[str] = [
        f"File: {path} | TSV | {format_file_size(size)}",
        f"Columns ({len(headers)}): {', '.join(headers)}",
        f"Rows: {total}",
    ]

    if query:
        parts.append(f"Query context: {query}")

    if data_rows:
        sample = data_rows[:max_sample]
        parts.append(f"\nSample data (first {len(sample)} rows):")
        for row in sample:
            parts.append("  " + "\t".join(row))

    return "\n".join(parts)


def _extract_json(
    tool: FileExtractTool,
    path: str,
    text: str,
    size: int,
    max_sample: int,
    query: str,
) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"Error: Invalid JSON in '{path}': {exc}"

    parts: list[str] = [
        f"File: {path} | JSON | {format_file_size(size)}",
    ]

    if query:
        parts.append(f"Query context: {query}")

    if isinstance(data, list):
        parts.append(f"Type: Array with {len(data)} items")
        if data and isinstance(data[0], dict):
            keys = list(data[0].keys())
            parts.append(f"Item keys ({len(keys)}): {', '.join(keys)}")
            parts.append(f"\nSample items (first {min(max_sample, len(data))}):")
            for item in data[:max_sample]:
                parts.append(f"  {json.dumps(item, ensure_ascii=False, default=str)}")
        elif data:
            parts.append(f"\nSample items (first {min(max_sample, len(data))}):")
            for item in data[:max_sample]:
                parts.append(f"  {json.dumps(item, ensure_ascii=False, default=str)}")
    elif isinstance(data, dict):
        keys = list(data.keys())
        parts.append(f"Type: Object with {len(keys)} keys")
        parts.append(f"Keys: {', '.join(keys)}")
        parts.append("\nValues summary:")
        for key in keys[:max_sample]:
            val = data[key]
            if isinstance(val, list):
                parts.append(f"  {key}: Array[{len(val)}]")
            elif isinstance(val, dict):
                parts.append(f"  {key}: Object{{{', '.join(list(val.keys())[:3])}...}}")
            else:
                s = json.dumps(val, ensure_ascii=False, default=str)
                if len(s) > 80:
                    s = s[:77] + "..."
                parts.append(f"  {key}: {s}")
    else:
        parts.append(f"Type: {type(data).__name__}")
        parts.append(f"Value: {json.dumps(data, ensure_ascii=False, default=str)}")

    return "\n".join(parts)


def _extract_jsonl(
    tool: FileExtractTool,
    path: str,
    text: str,
    size: int,
    max_sample: int,
    query: str,
) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    total = len(lines)
    parts: list[str] = [
        f"File: {path} | JSONL | {format_file_size(size)}",
        f"Lines: {total}",
    ]

    if query:
        parts.append(f"Query context: {query}")

    if not lines:
        parts.append("(empty file)")
        return "\n".join(parts)

    try:
        first = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        return f"Error: Invalid JSONL in '{path}' (line 1): {exc}"

    if isinstance(first, dict):
        parts.append(f"Item keys ({len(first)}): {', '.join(first.keys())}")

    parts.append(f"\nSample lines (first {min(max_sample, total)}):")
    for ln in lines[:max_sample]:
        try:
            obj = json.loads(ln)
            parts.append(f"  {json.dumps(obj, ensure_ascii=False, default=str)}")
        except json.JSONDecodeError:
            parts.append(f"  (invalid JSON) {ln[:120]}")

    return "\n".join(parts)


def _extract_yaml(
    tool: FileExtractTool,
    path: str,
    text: str,
    size: int,
    max_sample: int,
    query: str,
) -> str:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return "Error: PyYAML is not installed. Install with: pip install pyyaml"

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return f"Error: Invalid YAML in '{path}': {exc}"

    parts: list[str] = [
        f"File: {path} | YAML | {format_file_size(size)}",
    ]

    if query:
        parts.append(f"Query context: {query}")

    if isinstance(data, list):
        parts.append(f"Type: Array with {len(data)} items")
        if data and isinstance(data[0], dict):
            parts.append(f"Item keys: {', '.join(data[0].keys())}")
        parts.append(f"\nSample items (first {min(max_sample, len(data))}):")
        for item in data[:max_sample]:
            parts.append(f"  {json.dumps(item, ensure_ascii=False, default=str)}")
    elif isinstance(data, dict):
        keys = list(data.keys())
        parts.append(f"Type: Object with {len(keys)} keys")
        parts.append(f"Keys: {', '.join(str(k) for k in keys)}")
        parts.append("\nValues summary:")
        for key in keys[:max_sample]:
            val = data[key]
            s = json.dumps(val, ensure_ascii=False, default=str)
            if len(s) > 80:
                s = s[:77] + "..."
            parts.append(f"  {key}: {s}")
    elif data is None:
        parts.append("(empty document)")
    else:
        parts.append(f"Type: {type(data).__name__}")
        parts.append(f"Value: {data}")

    return "\n".join(parts)


def _extract_xml(
    tool: FileExtractTool,
    path: str,
    text: str,
    size: int,
    max_sample: int,
    query: str,
) -> str:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return f"Error: Invalid XML in '{path}': {exc}"

    parts: list[str] = [
        f"File: {path} | XML | {format_file_size(size)}",
        f"Root element: <{root.tag}>",
    ]

    if query:
        parts.append(f"Query context: {query}")

    if root.attrib:
        parts.append(f"Root attributes: {dict(root.attrib)}")

    children = list(root)
    parts.append(f"Child elements: {len(children)}")

    if children:
        tag_counts: dict[str, int] = {}
        for child in children:
            tag_counts[child.tag] = tag_counts.get(child.tag, 0) + 1
        parts.append(
            f"Child tags: {', '.join(f'{t} ({c})' for t, c in tag_counts.items())}"
        )

        parts.append(f"\nSample elements (first {min(max_sample, len(children))}):")
        for child in children[:max_sample]:
            attrs = f" {dict(child.attrib)}" if child.attrib else ""
            child_text = (child.text or "").strip()
            if child_text:
                if len(child_text) > 80:
                    child_text = child_text[:77] + "..."
                parts.append(f"  <{child.tag}{attrs}> {child_text}")
            else:
                sub_count = len(list(child))
                parts.append(f"  <{child.tag}{attrs}> ({sub_count} sub-elements)")
    elif root.text and root.text.strip():
        t = root.text.strip()
        if len(t) > 200:
            t = t[:197] + "..."
        parts.append(f"Text content: {t}")

    return "\n".join(parts)


def _extract_toml(
    tool: FileExtractTool,
    path: str,
    text: str,
    size: int,
    max_sample: int,
    query: str,
) -> str:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef,import-untyped]
            except ImportError:
                return (
                    "Error: TOML parsing requires Python 3.11+ or the 'tomli' package. "
                    "Install with: pip install tomli"
                )

    try:
        data = tomllib.loads(text)
    except Exception as exc:
        return f"Error: Invalid TOML in '{path}': {exc}"

    parts: list[str] = [
        f"File: {path} | TOML | {format_file_size(size)}",
    ]

    if query:
        parts.append(f"Query context: {query}")

    keys = list(data.keys())
    parts.append(f"Type: Object with {len(keys)} top-level keys")
    parts.append(f"Keys: {', '.join(keys)}")
    parts.append("\nValues summary:")
    for key in keys[:max_sample]:
        val = data[key]
        s = json.dumps(val, ensure_ascii=False, default=str)
        if len(s) > 80:
            s = s[:77] + "..."
        parts.append(f"  {key}: {s}")

    return "\n".join(parts)


# ------------------------------------------------------------------ #
# Format registry                                                       #
# ------------------------------------------------------------------ #

_FORMAT_HANDLERS: dict[str, FormatHandler] = {
    ".csv": _extract_csv,
    ".tsv": _extract_tsv,
    ".json": _extract_json,
    ".jsonl": _extract_jsonl,
    ".ndjson": _extract_jsonl,
    ".yaml": _extract_yaml,
    ".yml": _extract_yaml,
    ".xml": _extract_xml,
    ".toml": _extract_toml,
}


def register_extract_format(extension: str, handler: FormatHandler) -> None:
    """Register a custom format handler for ``FileExtractTool``.

    Parameters
    ----------
    extension : str
        File extension including the dot (e.g. ``".parquet"``).
    handler : FormatHandler
        Callable with signature
        ``(tool, path, text, size, max_sample, query) -> str``.
    """
    _FORMAT_HANDLERS[extension.lower()] = handler


# ------------------------------------------------------------------ #
# Tool class                                                            #
# ------------------------------------------------------------------ #


class FileExtractTool(BaseTool):
    """Extract structured data from data files.

    Auto-detects format from file extension.  Returns headers/keys,
    row/item count, and sample rows so the LLM can understand the
    data structure before deciding next steps.

    Supported formats: CSV, TSV, JSON, JSONL/NDJSON, YAML, XML, TOML.
    New formats can be added via ``register_extract_format()``.

    Parameters accepted by ``execute()``:
        path (str): File path relative to *workspace_root*.
        query (str, optional): Natural-language hint about what to
            extract (passed through as context -- the tool always
            returns the structural summary).
        max_sample_rows (int, optional): Number of sample rows
            to include (default 5).
    """

    def __init__(
        self,
        workspace_root: str,
        *,
        max_sample_rows: int = DEFAULT_SAMPLE_ROWS,
        name: str = "file_extract",
        description: str = (
            "Extract structured data from data files (CSV, TSV, JSON, JSONL, "
            "YAML, XML, TOML). Returns headers, row count, and sample rows "
            "for data exploration."
        ),
    ) -> None:
        super().__init__(name=name, description=description)
        self.workspace_root = workspace_root
        self.max_sample_rows = max_sample_rows

    async def initialize(self) -> None:
        return

    async def execute(self, **kwargs: Any) -> str:
        path: str = kwargs.get("path", "")
        query: str = kwargs.get("query", "")
        max_sample: int = kwargs.get("max_sample_rows", self.max_sample_rows)

        if not path:
            return "Error: 'path' parameter is required."

        try:
            resolved = resolve_safe_path(self.workspace_root, path)
        except WorkspaceSecurityError as exc:
            return f"Error: {exc}"

        if not resolved.is_file():
            return f"Error: '{path}' is not a file or does not exist."

        ext = resolved.suffix.lower()
        size = resolved.stat().st_size

        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"Error reading file: {exc}"

        handler = _FORMAT_HANDLERS.get(ext)
        if handler is None:
            supported = ", ".join(sorted(_FORMAT_HANDLERS.keys()))
            return (
                f"Error: Unsupported file format '{ext}'. "
                f"FileExtractTool supports: {supported}"
            )

        return handler(self, path, text, size, max_sample, query)

    def get_spec(self) -> Dict[str, Any]:
        supported = ", ".join(sorted(_FORMAT_HANDLERS.keys()))
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            f"File path relative to workspace root. "
                            f"Supported: {supported}"
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional hint about what data to focus on "
                            "(e.g. 'revenue figures', 'user emails')."
                        ),
                    },
                    "max_sample_rows": {
                        "type": "integer",
                        "description": "Number of sample rows to include (default: 5).",
                        "default": 5,
                    },
                },
                "required": ["path"],
            },
        }
