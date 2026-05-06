"""Deterministic synthesis package builder.

The package is a bounded final-answer input assembled from curated run state
instead of the entire compacted conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier
from nucleusiq.agents.context.workspace import InMemoryWorkspace


@dataclass(frozen=True)
class SynthesisPackage:
    """Bounded synthesis input plus omission metadata."""

    text: str
    metadata: dict[str, Any]


def _cap_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _section(title: str, body: str) -> str:
    body = body.strip()
    if not body:
        return ""
    return f"## {title}\n{body}"


def _evidence_section(evidence: InMemoryEvidenceDossier) -> str:
    items = evidence.list()
    supported = [item for item in items if item.status != "gap"]
    if not supported:
        return ""
    lines: list[str] = []
    for item in supported:
        locator = f" {item.source.locator}" if item.source.locator else ""
        tags = f" [{', '.join(item.tags)}]" if item.tags else ""
        lines.append(
            f"- {item.status}: {item.claim} "
            f"(source={item.source.ref}{locator}, confidence={item.confidence:.2f}){tags}"
        )
    return "\n".join(lines)


def _gap_section(evidence: InMemoryEvidenceDossier) -> str:
    gaps = evidence.list(status="gap")
    if not gaps:
        return ""
    lines = []
    for item in gaps:
        reason = item.metadata.get("reason") or item.quote or ""
        tags = f" [{', '.join(item.tags)}]" if item.tags else ""
        lines.append(f"- gap: {item.claim} (reason={reason}){tags}")
    return "\n".join(lines)


def _source_index(evidence: InMemoryEvidenceDossier) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for item in evidence.list():
        if item.source.ref == "gap" or item.source.ref in seen:
            continue
        seen.add(item.source.ref)
        title = f" — {item.source.title}" if item.source.title else ""
        locator = f" {item.source.locator}" if item.source.locator else ""
        lines.append(f"- {item.source.ref}{locator}{title}")
    return "\n".join(lines)


def _snippet_section(recalled_snippets: tuple[str, ...]) -> str:
    lines = []
    for idx, snippet in enumerate(recalled_snippets, start=1):
        lines.append(f"- snippet {idx}: {' '.join(snippet.split())}")
    return "\n".join(lines)


def build_synthesis_package(
    *,
    task: str,
    output_shape: str = "",
    workspace: InMemoryWorkspace | None = None,
    evidence: InMemoryEvidenceDossier | None = None,
    recalled_snippets: tuple[str, ...] = (),
    max_chars: int = 12_000,
) -> SynthesisPackage:
    """Build a bounded package from task, evidence, workspace, and snippets."""
    omitted_sections: list[str] = []
    sections: list[tuple[str, str, bool]] = [
        ("Task And Constraints", task, True),
        ("Required Output Shape", output_shape, True),
    ]

    if evidence is not None:
        sections.extend(
            [
                ("Supported Evidence", _evidence_section(evidence), True),
                ("Known Gaps And Conflicts", _gap_section(evidence), True),
            ]
        )
    if workspace is not None:
        sections.append(
            ("Selected Workspace Notes", workspace.summarize(max_chars=2500), False)
        )
    if recalled_snippets:
        sections.append(
            ("Selected Recalled Snippets", _snippet_section(recalled_snippets), False)
        )
    if evidence is not None:
        sections.append(("Source Index", _source_index(evidence), False))

    rendered: list[str] = []
    for title, body, required in sections:
        section = _section(title, body)
        if not section:
            continue
        candidate = "\n\n".join([*rendered, section])
        if len(candidate) <= max_chars:
            rendered.append(section)
            continue
        if required:
            remaining = max_chars - len("\n\n".join(rendered))
            if rendered:
                remaining -= 2
            capped = _section(
                title, _cap_text(body, max(0, remaining - len(title) - 5))
            )
            if capped:
                rendered.append(capped)
            omitted_sections.append(title)
            break
        omitted_sections.append(title)

    text = "\n\n".join(rendered)
    if len(text) > max_chars:
        text = _cap_text(text, max_chars)
    metadata: dict[str, Any] = {
        "char_count": len(text),
        "max_chars": max_chars,
        "omitted_sections": omitted_sections,
        "section_count": len(rendered),
    }
    return SynthesisPackage(text=text, metadata=metadata)
