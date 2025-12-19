"""
NucleusIQ - An open-source framework for building and managing autonomous AI agents.

NucleusIQ offers diverse strategies and architectures to create intelligent chatbots,
financial tools, and multi-agent systems. With NucleusIQ, developers have the core
components and flexibility needed to develop advanced AI applications effortlessly.
"""

__version__ = "0.1.0"

# Load environment variables from project root .env (if present).
# This makes OPENAI_API_KEY etc. available no matter which submodule is imported.
try:
    from pathlib import Path
    from dotenv import load_dotenv

    _repo_root = Path(__file__).resolve().parents[2]
    _env_path = _repo_root / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except Exception:
    # Never fail import due to dotenv loading.
    pass

