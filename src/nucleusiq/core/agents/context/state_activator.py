"""Internal context-state activation for business tool results.

L4.5 turns context state from optional model-facing tools into framework-owned
runtime behavior: light ingest captures substantive read/search/file tool output
into workspace + optional L5 corpus; strict heuristics promote obvious facts into
the evidence dossier, without provider calls or task-specific parsers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from nucleusiq.agents.context.document_search import InMemoryDocumentCorpus
from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier
from nucleusiq.agents.context.workspace import InMemoryWorkspace
from nucleusiq.agents.context.workspace_tools import is_context_management_tool_name

_EVIDENCE_TOOL_HINTS = (
    "read",
    "search",
    "fetch",
    "lookup",
    "query",
    "report",
    "excerpt",
    "document",
    "pdf",
)
_METRIC_KEYWORDS = (
    "revenue",
    "sales",
    "margin",
    "profit",
    "income",
    "cash flow",
    "roce",
    "ebit",
    "ebitda",
    "risk",
    "strategy",
    "cloud",
)
_VALUE_RE = re.compile(
    r"(?:rs\.?|inr|usd|crore|million|billion|%|\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# Substrings matched against tool names for L4.5 *light ingest* when output is
# long enough but not strict evidence-shaped (durable workspace + L5 capture).
_INGEST_NAME_HINTS = (
    "read",
    "search",
    "list",
    "fetch",
    "load",
    "file",
    "dir",
    "query",
    "retrieve",
    "download",
    "excerpt",
    "document",
    "pdf",
    "report",
    "content",
    "scan",
    "browse",
    "open",
    "cat",
)
_ACRONYM_STOP = frozenset(
    {
        "FY",
        "AI",
        "IT",
        "UK",
        "US",
        "EU",
        "RS",
        "CEO",
        "CFO",
        "IPO",
        "NSE",
        "BSE",
        "THE",
        "AND",
        "EPS",
        "EBIT",
        "EBITDA",
        "GAAP",
        "IFRS",
    }
)


@dataclass
class ContextActivationMetrics:
    """Small cumulative telemetry snapshot for L4.5 activation."""

    activation_enabled: bool = True
    tool_results_seen: int = 0
    tool_results_activated: int = 0
    workspace_entries_created: int = 0
    evidence_items_promoted: int = 0
    evidence_gaps_promoted: int = 0
    activation_skipped_framework_tools: int = 0
    activation_skipped_non_evidence_tools: int = 0
    light_ingests: int = 0
    synthesis_package_used: bool = False
    synthesis_package_char_count: int = 0
    critic_used_package: bool = False
    raw_trace_fallback_used: bool = False
    last_skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_enabled": self.activation_enabled,
            "tool_results_seen": self.tool_results_seen,
            "tool_results_activated": self.tool_results_activated,
            "workspace_entries_created": self.workspace_entries_created,
            "evidence_items_promoted": self.evidence_items_promoted,
            "evidence_gaps_promoted": self.evidence_gaps_promoted,
            "activation_skipped_framework_tools": (
                self.activation_skipped_framework_tools
            ),
            "activation_skipped_non_evidence_tools": (
                self.activation_skipped_non_evidence_tools
            ),
            "light_ingests": self.light_ingests,
            "synthesis_package_used": self.synthesis_package_used,
            "synthesis_package_char_count": self.synthesis_package_char_count,
            "critic_used_package": self.critic_used_package,
            "raw_trace_fallback_used": self.raw_trace_fallback_used,
            "last_skip_reason": self.last_skip_reason,
        }


class ContextStateActivator:
    """Route business tool results into workspace, evidence, and optional L5 corpus.

    Strict "evidence-shaped" heuristics control dossier fact promotion only.
    Light ingest (notes + corpus indexing) applies to substantive read/search/file
    style tool output so autonomous runs retain durable state (see roadmap L4.5).
    """

    def __init__(
        self,
        *,
        workspace: InMemoryWorkspace,
        evidence: InMemoryEvidenceDossier,
        document_corpus: InMemoryDocumentCorpus | None = None,
        required_tags: tuple[str, ...] = (),
        max_inspect_chars: int = 12_000,
        max_facts_per_result: int = 8,
        max_corpus_index_chars: int = 500_000,
        ingest_min_chars: int = 200,
    ) -> None:
        self._workspace = workspace
        self._evidence = evidence
        self._document_corpus = document_corpus
        self._required_tags = required_tags
        self._max_inspect_chars = max_inspect_chars
        self._max_facts_per_result = max_facts_per_result
        self._max_corpus_index_chars = max_corpus_index_chars
        self._ingest_min_chars = ingest_min_chars
        self.metrics = ContextActivationMetrics()

    def activate_tool_result(
        self,
        *,
        tool_name: str | None,
        tool_call_id: str | None,
        tool_result: Any,
        tool_args: dict[str, Any] | None = None,
    ) -> ContextActivationMetrics:
        """Inspect one business tool result and update workspace/evidence state."""
        self.metrics.tool_results_seen += 1
        name = tool_name or "unknown_tool"

        if is_context_management_tool_name(name):
            self.metrics.activation_skipped_framework_tools += 1
            self.metrics.last_skip_reason = "framework_context_tool"
            return self.metrics

        text = _stringify_tool_result(tool_result)
        bounded = text[: self._max_inspect_chars]
        ingest, strict_shaped = _activation_gate(
            name,
            bounded,
            ingest_min_chars=self._ingest_min_chars,
        )
        if not ingest:
            self.metrics.activation_skipped_non_evidence_tools += 1
            self.metrics.last_skip_reason = "non_evidence_tool_result"
            return self.metrics

        self._index_tool_result_text_into_corpus(
            tool_name=name,
            tool_call_id=tool_call_id,
            tool_args=tool_args or {},
            text=bounded,
        )

        source_ref = _source_ref(name, tool_call_id)
        activated = False

        if bounded.strip():
            self._workspace.write_note(
                title=f"Observed {name}",
                content=_ingest_progress_note(
                    name,
                    bounded,
                    strict_evidence_shaped=strict_shaped,
                ),
                source_refs=(source_ref,),
                metadata={"tool_name": name, "tool_call_id": tool_call_id},
            )
            self.metrics.workspace_entries_created += 1
            activated = True
        if not strict_shaped:
            self.metrics.light_ingests += 1

        promoted_tags: set[str] = set()
        promoted_for_result = 0
        if strict_shaped:
            for sentence in _candidate_sentences(bounded):
                tags = _infer_tags(sentence, name, tool_args or {})
                if not tags:
                    continue
                self._evidence.add_evidence(
                    claim=sentence,
                    source_ref=source_ref,
                    title=name,
                    locator=_locator_from_args(tool_args or {}),
                    tool_name=name,
                    confidence=0.75,
                    tags=tuple(sorted(tags)),
                    quote=sentence,
                    metadata={"tool_call_id": tool_call_id},
                )
                promoted_tags.update(tags)
                self.metrics.evidence_items_promoted += 1
                promoted_for_result += 1
                activated = True
                if promoted_for_result >= self._max_facts_per_result:
                    break

        self._record_required_tag_gaps(promoted_tags, source_ref=source_ref)

        if activated:
            self.metrics.tool_results_activated += 1
            self.metrics.last_skip_reason = None
        return self.metrics

    def _index_tool_result_text_into_corpus(
        self,
        *,
        tool_name: str,
        tool_call_id: str | None,
        tool_args: dict[str, Any],
        text: str,
    ) -> None:
        if self._document_corpus is None or self._max_corpus_index_chars <= 0:
            return
        if not text.strip():
            return
        cap = min(len(text), self._max_corpus_index_chars)
        body = text[:cap]
        doc_id = _corpus_document_id(tool_name, tool_call_id, tool_args)
        meta: dict[str, Any] = {"tool_name": tool_name}
        if tool_call_id:
            meta["tool_call_id"] = tool_call_id
        for k in (
            "company",
            "entity",
            "year",
            "filename",
            "document_id",
            "path",
            "file",
        ):
            if k in tool_args and tool_args[k] is not None:
                meta[k] = tool_args[k]
        title: str | None = None
        if isinstance(tool_args.get("title"), str):
            title = tool_args["title"]
        elif isinstance(tool_args.get("filename"), str):
            title = os.path.basename(tool_args["filename"].strip().replace("\\", "/"))
        else:
            title = tool_name
        self._document_corpus.index_document(
            doc_id,
            body,
            title=title,
            metadata=meta,
        )

    def _record_required_tag_gaps(
        self,
        promoted_tags: set[str],
        *,
        source_ref: str,
    ) -> None:
        existing_gap_tags = {
            tag for item in self._evidence.list(status="gap") for tag in item.tags
        }
        for tag in self._required_tags:
            if tag in promoted_tags or tag in existing_gap_tags:
                continue
            self._evidence.add_gap(
                question=f"Need evidence for {tag}.",
                reason="No supported fact for this required tag was promoted.",
                tags=(tag,),
                metadata={"source_ref": source_ref},
            )
            self.metrics.evidence_gaps_promoted += 1


def _stringify_tool_result(tool_result: Any) -> str:
    if isinstance(tool_result, str):
        return tool_result
    try:
        return json.dumps(tool_result, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(tool_result)


def _source_ref(tool_name: str, tool_call_id: str | None) -> str:
    suffix = tool_call_id or "unknown"
    return f"obs:{tool_name}:{suffix}"


def _tool_name_eligible_for_light_ingest(tool_name: str) -> bool:
    lower = tool_name.lower()
    return any(fragment in lower for fragment in _INGEST_NAME_HINTS)


def _activation_gate(
    tool_name: str,
    text: str,
    *,
    ingest_min_chars: int,
) -> tuple[bool, bool]:
    """Return (ingest_curated_state, strict_evidence_shaped).

    Ingest is True when we should write workspace / L5 index for this result.
    strict_evidence_shaped controls dossier sentence promotion only.
    """
    strict = _looks_evidence_shaped(tool_name, text)
    if strict:
        return True, True
    stripped = text.strip()
    if not stripped:
        return False, False
    if len(stripped) < ingest_min_chars:
        return False, False
    if _tool_name_eligible_for_light_ingest(tool_name):
        return True, False
    return False, False


def _ingest_progress_note(
    tool_name: str,
    text: str,
    *,
    strict_evidence_shaped: bool,
) -> str:
    compact = " ".join(text.split())
    preview = compact[:700].rstrip()
    inspected = len(text)
    if strict_evidence_shaped:
        return (
            f"{tool_name} returned evidence-shaped content ({inspected} chars inspected). "
            f"Preview: {preview}"
        )
    return (
        f"{tool_name} returned substantive tool output ({inspected} chars inspected); "
        f"indexed for run-local retrieval. Preview: {preview}"
    )


def _text_mentions_ai_topic(lower_text: str) -> bool:
    """True for GenAI / AI-as-topic without matching ``ai`` inside unrelated words."""
    return "genai" in lower_text or bool(re.search(r"\bai\b", lower_text))


def _looks_evidence_shaped(tool_name: str, text: str) -> bool:
    lower_name = tool_name.lower()
    lower_text = text.lower()
    has_tool_hint = any(hint in lower_name for hint in _EVIDENCE_TOOL_HINTS)
    has_metric = any(keyword in lower_text for keyword in _METRIC_KEYWORDS)
    if not has_metric:
        has_metric = _text_mentions_ai_topic(lower_text)
    has_value = bool(_VALUE_RE.search(text))
    return has_tool_hint and (has_metric or has_value)


def _candidate_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for raw in _SENTENCE_RE.split(text):
        sentence = " ".join(raw.split()).strip(" -")
        if len(sentence) < 20:
            continue
        lower = sentence.lower()
        if not any(keyword in lower for keyword in _METRIC_KEYWORDS):
            if not _text_mentions_ai_topic(lower):
                continue
        if not _VALUE_RE.search(sentence):
            continue
        sentences.append(sentence[:900])
    return sentences


def _slug_from_label(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip())
    s = s.strip("_")
    return s[:120] if s else ""


def _corpus_document_id(
    tool_name: str,
    tool_call_id: str | None,
    tool_args: dict[str, Any],
) -> str:
    for key in ("document_id", "filename", "file", "path"):
        val = tool_args.get(key)
        if isinstance(val, str) and val.strip():
            base = os.path.basename(val.strip().replace("\\", "/"))
            stem, _ext = os.path.splitext(base)
            slug = _slug_from_label(stem or base)
            if slug:
                return f"doc:{slug.lower()}"
    safe_tool = _slug_from_label(tool_name) or "tool"
    suffix = _slug_from_label(tool_call_id or "unknown") or "unknown"
    return f"tool:{safe_tool.lower()}:{suffix.lower()}"


def _infer_tags(
    sentence: str,
    tool_name: str,
    tool_args: dict[str, Any],
) -> set[str]:
    haystack = " ".join([sentence, tool_name, json.dumps(tool_args)]).lower()
    tags: set[str] = set()

    company = tool_args.get("company")
    if isinstance(company, str) and company.strip():
        slug = _slug_from_label(company.strip().lower())
        if slug:
            tags.add(f"company:{slug.lower()}")

    for key in ("entity", "subject", "issuer"):
        val = tool_args.get(key)
        if isinstance(val, str) and val.strip():
            slug = _slug_from_label(val.strip().lower())
            if slug:
                tags.add(f"entity:{slug.lower()}")

    tick = tool_args.get("ticker")
    if isinstance(tick, str) and tick.strip():
        slug = _slug_from_label(tick.strip().upper())
        if slug:
            tags.add(f"ticker:{slug.lower()}")

    raw_tags = tool_args.get("tags")
    if raw_tags is None:
        raw_tags = tool_args.get("evidence_tags")
    if isinstance(raw_tags, (list, tuple)):
        for t in raw_tags:
            if isinstance(t, str) and t.strip():
                tags.add(t.strip())

    for key in ("filename", "path", "file"):
        val = tool_args.get(key)
        if isinstance(val, str) and val.strip():
            base = os.path.basename(val.strip().replace("\\", "/"))
            stem, _ = os.path.splitext(base)
            slug = _slug_from_label(stem)
            if slug:
                tags.add(f"document:{slug.lower()}")
            break

    did = tool_args.get("document_id")
    if isinstance(did, str) and did.strip() and "/" not in did and "\\" not in did:
        slug = _slug_from_label(did.strip())
        if slug:
            tags.add(f"document:{slug.lower()}")

    acronym_haystack = f"{sentence} {tool_name}"
    for m in re.finditer(r"\b([A-Z]{2,8})\b", acronym_haystack):
        tok = m.group(1)
        if tok in _ACRONYM_STOP:
            continue
        tags.add(f"entity:{tok.lower()}")

    metric_map = {
        "revenue": "metric:revenue",
        "sales": "metric:revenue",
        "margin": "metric:margin",
        "profit": "metric:profit",
        "income": "metric:profit",
        "roce": "metric:roce",
        "risk": "topic:risk",
        "strategy": "topic:strategy",
        "cloud": "topic:cloud",
    }
    for needle, tag in metric_map.items():
        if needle in haystack:
            tags.add(tag)
    if _text_mentions_ai_topic(haystack):
        tags.add("topic:ai")

    years = re.findall(r"\b(?:fy)?20\d{2}\b", haystack)
    for year in years[:2]:
        tags.add(f"year:{year}")

    return tags


def _locator_from_args(tool_args: dict[str, Any]) -> str | None:
    page = tool_args.get("page") or tool_args.get("start_page")
    if page is not None:
        return f"p.{page}"
    path = tool_args.get("path") or tool_args.get("file") or tool_args.get("filename")
    if path is not None:
        return str(path)
    return None


__all__ = ["ContextActivationMetrics", "ContextStateActivator"]
