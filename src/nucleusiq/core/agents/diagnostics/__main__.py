"""CLI: analyze a saved run report.

    python -m nucleusiq.agents.diagnostics report.json
    python -m nucleusiq.agents.diagnostics report.json --json
    python -m nucleusiq.agents.diagnostics report.json --redact
    cat report.json | python -m nucleusiq.agents.diagnostics -

``report.json`` may be ``RunReport.to_dict()`` or a whole
``AgentResult.summary()`` (the ``diagnostics`` block is picked out).
Exit status is 2 when any finding is *critical*, 1 for *warning*, else 0.
"""

from __future__ import annotations

import argparse
import json
import sys

from nucleusiq.agents.diagnostics.analyzer import attach_findings, load_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nucleusiq.agents.diagnostics",
        description="Analyze a NucleusIQ run report and print findings.",
    )
    parser.add_argument("report", help="path to a JSON report, or '-' for stdin")
    parser.add_argument(
        "--json", action="store_true", help="emit the analyzed report as JSON"
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="hash resource identifiers and drop free-text previews",
    )
    args = parser.parse_args(argv)

    source = sys.stdin.read() if args.report == "-" else args.report
    report = attach_findings(load_report(source))
    if args.redact:
        report = report.redacted()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.explain())

    severities = {f.severity for f in report.findings}
    if "critical" in severities:
        return 2
    if "warning" in severities:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
