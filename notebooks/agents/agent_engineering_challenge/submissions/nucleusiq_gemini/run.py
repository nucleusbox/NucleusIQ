"""Gemini submission: NucleusIQ + Gemini.

This submission implements the challenge in ../../TASK.md using NucleusIQ
and Google Gemini, producing a scorecard following ../../SCORECARD_SPEC.md.

Run from anywhere:

    python notebooks/agents/agent_engineering_challenge/submissions/nucleusiq_gemini/run.py

The runner auto-loads the repo .env so GEMINI_API_KEY in .env is enough.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CHALLENGE_DIR = HERE.parents[1]
DATA_DIR = CHALLENGE_DIR / "data"
TASK_FILE = CHALLENGE_DIR / "TASK.md"
REPO_ROOT = CHALLENGE_DIR.parents[2]


def _load_template_module():
    template_path = CHALLENGE_DIR / "submissions" / "_template" / "run.py"
    spec = importlib.util.spec_from_file_location("challenge01_template", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv as _load  # type: ignore

        _load(env_path)
        return
    except Exception:
        pass
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


async def _run_with_nucleusiq() -> dict[str, Any]:
    from nucleusiq.agents import Agent
    from nucleusiq.agents.config import AgentConfig, ExecutionMode
    from nucleusiq.agents.context.config import ContextConfig, ContextStrategy
    from nucleusiq.agents.task import Task
    from nucleusiq.prompts.zero_shot import ZeroShotPrompt
    from nucleusiq.tools.builtin import (
        DirectoryListTool,
        FileReadTool,
        FileSearchTool,
    )
    from nucleusiq_gemini import BaseGemini

    import nucleusiq

    objective = TASK_FILE.read_text(encoding="utf-8")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    agent = Agent(
        name="challenge01-nucleusiq-gemini",
        prompt=ZeroShotPrompt().configure(
            system=(
                "You are a careful investment-risk analyst. Use the file tools "
                "to inspect every file in the challenge data directory. Separate "
                "repeated boilerplate (dashboard headers, marketing blurbs, "
                "service-desk macros, vendor metadata) from real evidence. "
                "Produce all four required sections: recommendation, top 5 "
                "risks with evidence and source file names, and unknowns."
            )
        ),
        llm=BaseGemini(model_name=model_name),
        tools=[
            DirectoryListTool(workspace_root=str(DATA_DIR)),
            FileSearchTool(workspace_root=str(DATA_DIR)),
            FileReadTool(workspace_root=str(DATA_DIR)),
        ],
        config=AgentConfig(
            execution_mode=ExecutionMode.STANDARD,
            enable_tracing=True,
            respect_context_window=True,
            context=ContextConfig(
                strategy=ContextStrategy.PROGRESSIVE,
                optimal_budget=15_000,
                squeeze_threshold=0.0,
            ),
            max_tool_calls=20,
        ),
    )

    start = time.perf_counter()
    await agent.initialize()
    result = await agent.execute(Task(id="challenge01", objective=objective))
    duration = time.perf_counter() - start

    tel = result.context_telemetry

    # Resolve provider dynamically from telemetry if populated, else fall back to 'google'
    provider_name = (
        result.llm_calls[0].provider
        if result.llm_calls and getattr(result.llm_calls[0], "provider", None)
        else "google"
    )

    # Token accounting if exposed
    input_tokens = (
        sum(c.prompt_tokens for c in result.llm_calls) if result.llm_calls else None
    )
    if input_tokens == 0:
        input_tokens = None
    output_tokens = (
        sum(c.completion_tokens for c in result.llm_calls) if result.llm_calls else None
    )
    if output_tokens == 0:
        output_tokens = None

    return {
        "framework": "nucleusiq",
        "framework_version": getattr(nucleusiq, "__version__", "0.7.12"),
        "model": model_name,
        "provider": provider_name,
        "duration_seconds": round(duration, 2),
        "tool_calls": len(result.tool_calls),
        "llm_calls": len(result.llm_calls),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": getattr(tel, "estimated_cost_with_mgmt", None),
        "peak_context_utilization": getattr(tel, "peak_utilization", None),
        "compaction_events": getattr(tel, "compaction_count", None),
        "tokens_freed_total": getattr(tel, "tokens_freed_total", None),
        "final_answer": str(result.output or ""),
        "notes": (
            f"Run with Gemini ({model_name}) via NucleusIQ BaseGemini. "
            "Telemetry duration and call metrics captured via core execution tracer. "
            "optimal_budget=15K to observe compaction."
        ),
    }


def build_and_run() -> dict[str, Any]:
    _load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set (put it in repo .env or your shell).")
    return asyncio.run(_run_with_nucleusiq())


def main() -> None:
    template = _load_template_module()
    run_data = build_and_run()
    scorecard = template.produce_scorecard(run_data)
    out = HERE / "result.json"
    out.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    print(json.dumps(scorecard, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
