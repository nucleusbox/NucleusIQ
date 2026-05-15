#!/usr/bin/env python3
"""Agent + Claude **native** structured outputs (``output_config.format`` / JSON Schema).

Requires models that support `Claude structured outputs
<https://platform.claude.com/docs/en/build-with-claude/structured-outputs>`_ and
``ANTHROPIC_MODEL`` pointing at such a SKU (see ``examples/agents/09_anthropic_list_models.py``).

Run from ``src/providers/llms/anthropic``::

    uv sync --group full
    uv run python examples/output_parsers/anthropic_native_structured_example.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util_env import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

from nucleusiq.agents.agent import Agent  # noqa: E402
from nucleusiq.agents.config.agent_config import (  # noqa: E402
    AgentConfig,
    ExecutionMode,
)
from nucleusiq.agents.task import Task  # noqa: E402
from nucleusiq.prompts.zero_shot import ZeroShotPrompt  # noqa: E402
from nucleusiq_anthropic import BaseAnthropic  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CapsuleSummary(BaseModel):
    title: str = Field(description="Short title")
    bullets: list[str] = Field(description="2-4 key bullet points")


async def main() -> None:
    model = (
        os.getenv("ANTHROPIC_MODEL", "").strip()
        or os.getenv(
            "CLAUDE_MODEL",
            "",
        ).strip()
    )
    if not model:
        logger.error(
            "Set ANTHROPIC_MODEL to a Claude SKU that supports structured outputs."
        )
        return

    llm = BaseAnthropic(model_name=model, async_mode=True)
    agent = Agent(
        name="AnthropicStructured",
        role="Summariser",
        objective="Produce a compact capsule summary as JSON.",
        prompt=ZeroShotPrompt().configure(
            system=(
                "You reply only via the required structured JSON schema "
                "(no preamble, no markdown fences)."
            ),
        ),
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        response_format=CapsuleSummary,
    )

    task = Task(
        id="cap-1",
        objective=(
            "Summarise: NucleusIQ agents support DIRECT, STANDARD, and AUTONOMOUS "
            "execution modes with pluggable LLM providers."
        ),
    )

    result = await agent.execute(task)
    summary = result["output"]

    logger.info("Parsed type: %s", type(summary).__name__)
    logger.info("Title: %s", getattr(summary, "title", summary))
    if hasattr(summary, "bullets"):
        for b in summary.bullets:
            logger.info("  - %s", b)


if __name__ == "__main__":
    asyncio.run(main())
