"""Deterministic synthesis package builder.

The package is a bounded final-answer input assembled from curated run state
instead of the entire compacted conversation.  It is consumed by three
roles — the synthesis pass, the Critic and the Refiner — and one property
matters more than its size: **it must never look complete when it is not**
(design invariant I-10, "the verifier sees what the generator saw, or knows
it is seeing less").

Two rules follow from that:

* Lists (evidence items, workspace notes, gaps, snippets) degrade by
  dropping *whole items* and appending an explicit ``[N more … omitted]``
  line.  A list cut mid-item reads as a shorter, complete list — that is
  exactly how a Critic once concluded "the evidence only contains three
  invoices" and had six correct records deleted.
* ``metadata["complete"]`` is ``True`` only when every item of every
  section is present.  Consumers use it to decide whether the package can
  stand alone or must be accompanied by the raw trace.

The size cap is supplied by the caller from :class:`BudgetResolver`; the
default here is only a fallback for direct callers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier
from nucleusiq.agents.context.workspace import InMemoryWorkspace

#: Fallback when the caller does not pass a budget-derived cap.
DEFAULT_PACKAGE_CHARS = 12_000

#: Resources listed by name in the coverage section; beyond this the rest
#: is a count so the section stays small at any resource count.
_MAX_LISTED_RESOURCES = 60

#: A workspace note is a preview of one tool result.  This is the least a
#: note can be cut to and still identify its source and first facts.
_MIN_NOTE_CHARS = 240


@dataclass(frozen=True)
class SynthesisPackage:
    """Bounded synthesis input plus omission metadata."""

    text: str
    metadata: dict[str, Any]

    @property
    def complete(self) -> bool:
        """True when no item of any section had to be dropped or cut."""
        return bool(self.metadata.get("complete", False))

    def visibility_note(self) -> str:
        """One paragraph a consumer can put in front of the model.

        Empty when the package is complete.  Otherwise it says, in the
        model's terms, what is missing and what not to conclude from it.
        """
        if self.complete:
            return ""
        omitted = self.metadata.get("omitted_items") or {}
        included = self.metadata.get("included_items") or {}
        cut = self.metadata.get("cut_items") or {}
        parts = []
        for section, n in omitted.items():
            if n:
                shown = included.get(section, 0)
                parts.append(f"{section}: {shown} of {shown + n} shown")
        for section, n in cut.items():
            if n:
                parts.append(f"{section}: {n} shortened")
        detail = "; ".join(parts) if parts else "some sections were cut for space"
        return (
            "EVIDENCE VISIBILITY: this package is PARTIAL (" + detail + "). "
            "The generator saw the full tool results; you do not. Absence of a "
            "fact from this package is NOT evidence that it is unsupported — "
            "consult the raw trace if provided, and where you cannot verify a "
            "claim, say so instead of calling it wrong."
        )


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


def _omission_line(omitted: int, noun: str) -> str:
    return (
        f"- [{omitted} more {noun} omitted for space — this list is INCOMPLETE; "
        "absence here is not absence of evidence]"
    )


# --------------------------------------------------------------------------- #
# Item renderers                                                              #
# --------------------------------------------------------------------------- #


def _evidence_items(evidence: InMemoryEvidenceDossier) -> list[str]:
    lines: list[str] = []
    for item in evidence.list():
        if item.status == "gap":
            continue
        locator = f" {item.source.locator}" if item.source.locator else ""
        tags = f" [{', '.join(item.tags)}]" if item.tags else ""
        lines.append(
            f"- {item.status}: {item.claim} "
            f"(source={item.source.ref}{locator}, confidence={item.confidence:.2f}){tags}"
        )
    return lines


def _gap_items(evidence: InMemoryEvidenceDossier) -> list[str]:
    lines: list[str] = []
    for item in evidence.list(status="gap"):
        reason = item.metadata.get("reason") or item.quote or ""
        tags = f" [{', '.join(item.tags)}]" if item.tags else ""
        lines.append(f"- gap: {item.claim} (reason={reason}){tags}")
    return lines


def _source_items(evidence: InMemoryEvidenceDossier) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for item in evidence.list():
        if item.source.ref == "gap" or item.source.ref in seen:
            continue
        seen.add(item.source.ref)
        title = f" — {item.source.title}" if item.source.title else ""
        locator = f" {item.source.locator}" if item.source.locator else ""
        lines.append(f"- {item.source.ref}{locator}{title}")
    return lines


def _snippet_items(recalled_snippets: Sequence[str]) -> list[str]:
    return [
        f"- snippet {idx}: {' '.join(snippet.split())}"
        for idx, snippet in enumerate(recalled_snippets, start=1)
    ]


def _note_items(
    workspace: InMemoryWorkspace, *, per_note_chars: int
) -> tuple[list[str], int]:
    """Workspace entries, one per line, each cut to ``per_note_chars``.

    Returns ``(lines, cut)`` where ``cut`` counts notes whose preview had
    to be shortened.  A cut note is still *present* (it is not an
    omission) but it hides facts, so the count is surfaced in
    ``metadata["cut_items"]`` and makes the package non-complete.
    """
    lines: list[str] = []
    cut = 0
    for entry in workspace.list():
        content = " ".join(entry.content.split())
        line = f"- {entry.kind}: {entry.title} — {content}"
        if len(line) > per_note_chars:
            cut += 1
        lines.append(_cap_text(line, per_note_chars))
    return lines, cut


def coverage_facts(coverage: Mapping[str, Any]) -> str:
    """Render ``ResourceTouchTracker.to_dict()`` as harness-verified facts.

    Shared by the package's "Resources Processed" section and the Refiner
    prompt so every downstream role reads the same statement of what the
    tools actually touched.
    """
    return _coverage_body(coverage)


def _coverage_body(coverage: Mapping[str, Any]) -> str:
    resources = list(coverage.get("resources") or [])
    if not resources:
        return ""
    touched = list(coverage.get("touched") or [])
    unprocessed = list(coverage.get("unprocessed") or [])

    def _listed(items: list[str]) -> str:
        if len(items) <= _MAX_LISTED_RESOURCES:
            return ", ".join(items)
        rest = len(items) - _MAX_LISTED_RESOURCES
        return ", ".join(items[:_MAX_LISTED_RESOURCES]) + f", … (+{rest} more)"

    lines = [
        f"- {len(touched)} of {len(resources)} declared resources were read by "
        "tools during this run (harness-verified from tool traffic, not from "
        "the model's claims)."
    ]
    if touched:
        lines.append(f"- read: {_listed(touched)}")
    if unprocessed:
        lines.append(f"- NOT read: {_listed(unprocessed)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Fill                                                                        #
# --------------------------------------------------------------------------- #


def _fill_items(items: list[str], budget: int, noun: str) -> tuple[str, int, int]:
    """Return ``(body, included, omitted)`` fitting ``budget`` chars.

    Items are kept in order and dropped from the end; the omission line
    is always part of the body when anything was dropped, and its own
    cost is accounted for so the body never exceeds ``budget``.
    """
    if not items or budget <= 0:
        return "", 0, len(items)
    total = sum(len(i) + 1 for i in items) - 1
    if total <= budget:
        return "\n".join(items), len(items), 0
    included: list[str] = []
    used = 0
    for idx, item in enumerate(items):
        remaining_after = len(items) - (idx + 1)
        reserve = len(_omission_line(max(1, remaining_after), noun)) + 1
        cost = len(item) + (1 if included else 0)
        if used + cost + reserve > budget:
            break
        included.append(item)
        used += cost
    omitted = len(items) - len(included)
    body_lines = [*included, _omission_line(omitted, noun)]
    return "\n".join(body_lines), len(included), omitted


def build_synthesis_package(
    *,
    task: str,
    output_shape: str = "",
    workspace: InMemoryWorkspace | None = None,
    evidence: InMemoryEvidenceDossier | None = None,
    recalled_snippets: tuple[str, ...] = (),
    coverage: Mapping[str, Any] | None = None,
    max_chars: int = DEFAULT_PACKAGE_CHARS,
) -> SynthesisPackage:
    """Build a bounded package from task, evidence, coverage, workspace, snippets.

    Section order is priority order.  Prose sections (task, output shape,
    coverage) are required and are cut with ``...`` only as a last resort;
    list sections drop whole items and say how many.  ``max_chars`` should
    come from ``BudgetResolver.handoff_chars(...)``.
    """
    omitted_sections: list[str] = []
    omitted_items: dict[str, int] = {}
    included_items: dict[str, int] = {}
    cut_items: dict[str, int] = {}
    rendered: list[str] = []

    def used() -> int:
        return len("\n\n".join(rendered))

    def remaining() -> int:
        return max_chars - used() - (2 if rendered else 0)

    def add_prose(title: str, body: str) -> None:
        section = _section(title, body)
        if not section:
            return
        if len(section) <= remaining():
            rendered.append(section)
            return
        capped = _section(title, _cap_text(body, max(0, remaining() - len(title) - 4)))
        if capped and len(capped) <= remaining():
            rendered.append(capped)
        omitted_sections.append(title)

    def add_items(title: str, items: list[str], noun: str) -> None:
        if not items:
            return
        header_cost = len(f"## {title}\n")
        body, n_in, n_out = _fill_items(items, remaining() - header_cost, noun)
        included_items[title] = n_in
        omitted_items[title] = n_out
        if n_in == 0:
            omitted_sections.append(title)
            return
        rendered.append(_section(title, body))

    # Required prose.
    add_prose("Task And Constraints", task)
    add_prose("Required Output Shape", output_shape)
    if coverage is not None:
        add_prose("Resources Processed", _coverage_body(coverage))

    # Evidence lists.
    if evidence is not None:
        add_items("Supported Evidence", _evidence_items(evidence), "evidence items")
        add_items("Known Gaps And Conflicts", _gap_items(evidence), "gaps")

    # Workspace notes: every note present if at all possible — the per-note
    # preview shrinks with the note count before any note is dropped.
    if workspace is not None:
        entries = workspace.list()
        if entries:
            room = remaining() - len("## Selected Workspace Notes\n")
            per_note = max(_MIN_NOTE_CHARS, room // max(1, len(entries)) - 1)
            note_lines, cut = _note_items(workspace, per_note_chars=per_note)
            add_items("Selected Workspace Notes", note_lines, "notes")
            if cut:
                cut_items["Selected Workspace Notes"] = cut

    if recalled_snippets:
        add_items(
            "Selected Recalled Snippets", _snippet_items(recalled_snippets), "snippets"
        )
    if evidence is not None:
        add_items("Source Index", _source_items(evidence), "sources")

    text = "\n\n".join(rendered)
    if len(text) > max_chars:  # defensive; the fill logic should prevent this
        text = _cap_text(text, max_chars)

    complete = (
        not omitted_sections
        and not any(omitted_items.values())
        and not any(cut_items.values())
    )
    metadata: dict[str, Any] = {
        "char_count": len(text),
        "max_chars": max_chars,
        "omitted_sections": omitted_sections,
        "omitted_items": omitted_items,
        "included_items": included_items,
        "cut_items": cut_items,
        "complete": complete,
        "section_count": len(rendered),
    }
    return SynthesisPackage(text=text, metadata=metadata)
