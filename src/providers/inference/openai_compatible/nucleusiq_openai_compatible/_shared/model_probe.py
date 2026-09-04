"""``GET /v1/models`` discovery.

Two jobs, both optional and neither ever fatal by accident:

* discover the served context length (vLLM and SGLang publish
  ``max_model_len`` on the model card), so users who forget
  ``context_window=`` still get a correct budget instead of a wrong guess;
* list the models the server actually serves, so a wrong ``model=`` produces
  *"served models are: a, b, c"* instead of a bare 404 mid-run.

Injected into the provider as ``probe=`` so tests supply a fake and never
patch the ``openai`` SDK.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ModelProbe", "ProbeResult"]

_logger = logging.getLogger(__name__)

_CONTEXT_FIELDS = (
    "max_model_len",
    "max_context_length",
    "context_length",
    "max_sequence_length",
)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of a ``/v1/models`` call.

    A failed probe is a normal, non-fatal outcome — the server may not
    implement the endpoint, or may be behind a gateway that blocks it.
    """

    reachable: bool
    model_ids: tuple[str, ...] = ()
    context_window: int | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    def has_model(self, model: str) -> bool:
        """Whether *model* appears in the served list.

        Returns ``True`` when the list is empty or unknown, so an
        uninformative probe never blocks a working configuration.
        """
        if not self.model_ids:
            return True
        return model in self.model_ids


class ModelProbe:
    """Reads the ``/v1/models`` endpoint through an ``openai`` SDK client.

    Results are cached per instance: the served model list and its context
    length do not change while a server is running, and an agent run must
    not pay a round-trip per call.
    """

    __slots__ = ("_cache", "_client_factory")

    def __init__(self, client_factory: Any) -> None:
        """Args:
        client_factory: Zero-argument callable returning an object with
            an awaitable ``models.list()``, i.e. an
            ``openai.AsyncOpenAI``.  A factory rather than a client so
            credentials resolve at probe time.
        """
        self._client_factory = client_factory
        self._cache: ProbeResult | None = None

    @property
    def cached(self) -> ProbeResult | None:
        """The cached result, if the probe has already run."""
        return self._cache

    async def probe(self, *, model: str | None = None) -> ProbeResult:
        """Fetch and cache the served model list and context length.

        Never raises: transport and protocol failures are captured in
        :attr:`ProbeResult.error` and logged at debug level, because a
        probe failure must not prevent a working server from being used.
        """
        if self._cache is not None:
            return self._cache

        try:
            client = self._client_factory()
            listing = await client.models.list()
            entries = list(getattr(listing, "data", None) or [])
        except Exception as exc:
            _logger.debug("Model probe failed (non-fatal): %s", exc)
            self._cache = ProbeResult(reachable=False, error=str(exc))
            return self._cache

        ids: list[str] = []
        window: int | None = None
        raw: dict[str, Any] = {}

        for entry in entries:
            entry_id = _get(entry, "id")
            if isinstance(entry_id, str):
                ids.append(entry_id)
            if model is not None and entry_id != model:
                continue
            found = _extract_context_window(entry)
            if found is not None:
                window = found
                raw = _as_dict(entry)

        # Single-model servers routinely serve under a name that differs from
        # what the caller passed (a local path, say), so accept the sole
        # model card's window when there is exactly one.
        if window is None and len(entries) == 1:
            window = _extract_context_window(entries[0])
            raw = _as_dict(entries[0])

        self._cache = ProbeResult(
            reachable=True,
            model_ids=tuple(ids),
            context_window=window,
            raw=raw,
        )
        _logger.debug("Model probe: %d model(s), context_window=%s", len(ids), window)
        return self._cache


def _get(entry: Any, key: str) -> Any:
    """Read *key* from a dict-style or attribute-style model card."""
    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)


def _as_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return dict(entry)
    dump = getattr(entry, "model_dump", None)
    if callable(dump):
        try:
            result = dump()
            if isinstance(result, dict):
                return result
        except Exception:  # pragma: no cover - defensive
            pass
    return {}


def _extract_context_window(entry: Any) -> int | None:
    """Pull a context length off a model card, trying known field names."""
    for name in _CONTEXT_FIELDS:
        value = _get(entry, name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    meta = _get(entry, "meta") or _get(entry, "metadata")
    if isinstance(meta, dict):
        for name in _CONTEXT_FIELDS:
            value = meta.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None
