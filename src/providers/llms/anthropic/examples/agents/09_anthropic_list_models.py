"""List Messages-API models visible to your API key (fixes ``404 model not found`` guessing).

Uses ``GET /v1/models`` via the Anthropic SDK. Pick an ``id`` and set::

    ANTHROPIC_MODEL=<that id>

Optional: ``ANTHROPIC_LIST_MODELS_VERBOSE=1`` appends capability blobs (noisy).

Run from ``src/providers/llms/anthropic``::

    uv run python examples/agents/09_anthropic_list_models.py

Requires ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from util_env import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

from anthropic import Anthropic  # noqa: E402


def main() -> None:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("Set ANTHROPIC_API_KEY (repo-root .env or env).")
        raise SystemExit(1)

    client = Anthropic(api_key=key)
    page = client.models.list(limit=40)
    verbose = os.getenv("ANTHROPIC_LIST_MODELS_VERBOSE", "").strip() in (
        "1",
        "true",
        "yes",
    )

    header = (
        "Models available to this key (latest first). "
        "Set ANTHROPIC_MODEL to an id column; use ANTHROPIC_LIST_MODELS_VERBOSE=1 for capabilities.\n"
    )
    print(header)
    for m in page.data:
        if verbose and getattr(m, "capabilities", None) is not None:
            extra = "\t" + str(m.capabilities)
        else:
            extra = ""
        print(f"{m.id}\t{m.display_name}{extra}")


if __name__ == "__main__":
    main()
