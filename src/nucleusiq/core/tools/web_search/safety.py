"""Safety filters for web-search results.

DuckDuckGo does **not** offer a full search API. Their documented
Instant Answer API (https://duckduckgo.com/api) returns topic boxes,
not ranked links, and they say they cannot syndicate SERP results.
The default adapter therefore uses ``ddgs`` for titles/URLs/snippets
only. This module is the security boundary around that untrusted data:

* never fetch a result URL (no scrape, no ``ddgs.extract``)
* accept only public ``http`` / ``https`` URLs (blocks SSRF / file / js)
* strip HTML and control characters from snippets (prompt injection)
* optional constructor-only allowlist / blocklist of hostnames
"""

from __future__ import annotations

import html
import ipaddress
import re
from collections.abc import Iterable, Sequence
from urllib.parse import urlparse

from nucleusiq.tools.web_search.types import SearchHit

_TAG = re.compile(r"<[^>]+>")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SNIPPET_MAX = 400
_TITLE_MAX = 200

_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localhost", ".onion")
_BLOCKED_HOSTS = frozenset({"localhost", "localhost.localdomain", "0.0.0.0"})


def normalize_domains(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        host = str(raw).strip().lower()
        host = host.removeprefix("https://").removeprefix("http://")
        host = host.split("/")[0].split(":")[0].lstrip(".")
        if host and host not in seen:
            seen.add(host)
            out.append(host)
    return tuple(out)


def sanitize_text(text: str, *, max_len: int) -> str:
    cleaned = html.unescape(str(text or ""))
    cleaned = _TAG.sub("", cleaned)
    cleaned = _CTRL.sub("", cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_len:
        return cleaned[: max_len - 1] + "…"
    return cleaned


def is_public_http_url(url: str) -> bool:
    """Return True for a public http(s) URL an agent may cite.

    Rejects credentials, non-http schemes, localhost, private/reserved
    IPs, and obvious intranet suffixes. Does not fetch the URL.
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.username or parsed.password:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or host in _BLOCKED_HOSTS:
        return False
    if any(host.endswith(suffix) for suffix in _BLOCKED_HOST_SUFFIXES):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return bool(ip.is_global) and not ip.is_multicast


def host_permitted(
    url: str,
    *,
    allowed: Sequence[str] = (),
    blocked: Sequence[str] = (),
) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    if blocked and any(_host_matches(host, domain) for domain in blocked):
        return False
    if allowed:
        return any(_host_matches(host, domain) for domain in allowed)
    return True


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def with_site_operators(query: str, allowed: Sequence[str]) -> str:
    """Hint the engine with ``site:`` operators. Filtering is still enforced."""
    if not allowed:
        return query
    if len(allowed) == 1:
        return f"{query} site:{allowed[0]}"
    clause = " OR ".join(f"site:{domain}" for domain in allowed)
    return f"{query} ({clause})"


def sanitize_hits(
    hits: Iterable[SearchHit],
    *,
    allowed_domains: Sequence[str] = (),
    blocked_domains: Sequence[str] = (),
) -> list[SearchHit]:
    safe: list[SearchHit] = []
    for hit in hits:
        url = (hit.url or "").strip()
        if not is_public_http_url(url):
            continue
        if not host_permitted(url, allowed=allowed_domains, blocked=blocked_domains):
            continue
        safe.append(
            SearchHit(
                title=sanitize_text(hit.title, max_len=_TITLE_MAX) or "(untitled)",
                url=url,
                snippet=sanitize_text(hit.snippet, max_len=_SNIPPET_MAX),
            )
        )
    return safe
