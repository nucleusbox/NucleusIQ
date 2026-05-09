"""Contract tests for :mod:`nucleusiq.llms.retry_policy` across LLM providers.

HTTP-based providers (Groq, OpenAI, and Gemini on 429) should call
``compute_rate_limit_sleep`` so ``Retry-After``, exponential backoff, and the
sleep cap behave identically. Provider packages assert that in their own
``call_with_retry`` tests; this module locks the **shared** metadata shape used
for logs and observability pipelines.
"""

from __future__ import annotations

from nucleusiq.llms.retry_policy import (
    DEFAULT_RATE_LIMIT_MAX_SLEEP_SECONDS,
    RateLimitRetryMeta,
    compute_rate_limit_sleep,
)


def test_rate_limit_retry_meta_includes_observability_fields() -> None:
    """Stable keys for log processors / metrics (do not rename lightly)."""
    _, meta = compute_rate_limit_sleep(2, "10")
    required: set[str] = {
        "attempt",
        "sleep_seconds",
        "retry_after_header_raw",
        "retry_after_seconds",
        "exponential_seconds",
        "max_sleep_cap",
        "used_retry_after",
    }
    assert required == set(meta.keys())


def test_default_cap_matches_documentation() -> None:
    assert DEFAULT_RATE_LIMIT_MAX_SLEEP_SECONDS == 120.0


def test_typed_dict_runtime_compatible_with_mapping() -> None:
    _: RateLimitRetryMeta = compute_rate_limit_sleep(1, None)[1]
