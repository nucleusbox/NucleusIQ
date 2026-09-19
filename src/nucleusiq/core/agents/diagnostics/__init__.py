"""Run diagnostics: termination reasons, run reports, and the offline analyzer."""

from nucleusiq.agents.diagnostics.analyzer import (
    analyze,
    attach_findings,
    load_report,
    recommendations,
    registered_rules,
)
from nucleusiq.agents.diagnostics.run_report import (
    Finding,
    RunCounters,
    RunRecorder,
    RunReport,
    TerminationReason,
    TerminationRecord,
    TimelineEvent,
    recorder_for,
)

__all__ = [
    "Finding",
    "RunCounters",
    "RunRecorder",
    "RunReport",
    "TerminationReason",
    "TerminationRecord",
    "TimelineEvent",
    "analyze",
    "attach_findings",
    "load_report",
    "recommendations",
    "recorder_for",
    "registered_rules",
]
