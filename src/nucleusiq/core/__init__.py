"""
NucleusIQ - An open-source framework for building and managing autonomous AI agents.

NucleusIQ offers diverse strategies and architectures to create intelligent chatbots,
financial tools, and multi-agent systems. With NucleusIQ, developers have the core
components and flexibility needed to develop advanced AI applications effortlessly.
"""

__version__ = "0.7.5"

# Auto-load the user's .env so API keys (OPENAI_API_KEY, etc.) are available.
# load_dotenv() with no args uses find_dotenv() which searches from the
# caller's CWD upward — works for any consumer project, not just this repo.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:
    pass
