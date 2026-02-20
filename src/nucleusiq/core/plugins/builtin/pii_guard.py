"""
PIIGuardPlugin — detect and handle PII in messages before they reach the model.

Supports built-in patterns (email, credit card, phone, IP, SSN) and custom
regex patterns. Strategies: redact, mask, block.

Usage::

    agent = Agent(
        ...,
        plugins=[
            PIIGuardPlugin(
                pii_types=["email", "credit_card"],
                strategy="redact",
            )
        ],
    )

    # Custom pattern
    agent = Agent(
        ...,
        plugins=[
            PIIGuardPlugin(
                custom_patterns={"api_key": r"sk-[a-zA-Z0-9]{32,}"},
                strategy="block",
            )
        ],
    )
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Sequence

from nucleusiq.plugins.base import BasePlugin, ModelRequest
from nucleusiq.plugins.errors import PluginHalt

logger = logging.getLogger(__name__)

BUILTIN_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "credit_card": re.compile(r"\b(?:\d[ \-]*?){13,19}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def _luhn_check(number: str) -> bool:
    """Validate a credit card number with the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


class PIIGuardPlugin(BasePlugin):
    """Detects and handles PII in messages before they reach the LLM.

    Args:
        pii_types: Built-in PII types to detect. Options: email, credit_card,
            phone, ip_address, ssn.
        custom_patterns: Dict of {name: regex_pattern} for custom PII types.
        strategy: How to handle detected PII:
            - ``"redact"``: Replace with ``[REDACTED_TYPE]`` (default)
            - ``"mask"``: Partially mask (e.g., ``****@email.com``)
            - ``"block"``: Raise ``PluginHalt`` if PII is found
        apply_to_input: Whether to check user messages. Defaults to True.
        apply_to_output: Whether to check model responses. Defaults to False.
    """

    def __init__(
        self,
        pii_types: Optional[Sequence[str]] = None,
        custom_patterns: Optional[Dict[str, str]] = None,
        strategy: str = "redact",
        apply_to_input: bool = True,
        apply_to_output: bool = False,
    ) -> None:
        if strategy not in ("redact", "mask", "block"):
            raise ValueError(f"Invalid strategy: {strategy!r}. Use 'redact', 'mask', or 'block'.")

        self._patterns: Dict[str, re.Pattern] = {}
        for pii_type in (pii_types or []):
            if pii_type not in BUILTIN_PATTERNS:
                raise ValueError(
                    f"Unknown PII type: {pii_type!r}. "
                    f"Available: {list(BUILTIN_PATTERNS.keys())}"
                )
            self._patterns[pii_type] = BUILTIN_PATTERNS[pii_type]

        for name, pattern in (custom_patterns or {}).items():
            self._patterns[name] = re.compile(pattern)

        if not self._patterns:
            raise ValueError("At least one pii_type or custom_pattern is required")

        self._strategy = strategy
        self._apply_to_input = apply_to_input
        self._apply_to_output = apply_to_output

    @property
    def name(self) -> str:
        return "pii_guard"

    def _detect(self, text: str) -> List[Dict[str, Any]]:
        """Find all PII matches in text."""
        findings: List[Dict[str, Any]] = []
        for pii_type, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                value = match.group()
                if pii_type == "credit_card" and not _luhn_check(value):
                    continue
                findings.append({
                    "type": pii_type,
                    "value": value,
                    "start": match.start(),
                    "end": match.end(),
                })
        return findings

    def _redact(self, text: str) -> str:
        """Replace PII with [REDACTED_TYPE] placeholders."""
        for pii_type, pattern in self._patterns.items():
            def replacer(m: re.Match) -> str:
                if pii_type == "credit_card" and not _luhn_check(m.group()):
                    return m.group()
                return f"[REDACTED_{pii_type.upper()}]"
            text = pattern.sub(replacer, text)
        return text

    def _mask(self, text: str) -> str:
        """Partially mask PII values."""
        for pii_type, pattern in self._patterns.items():
            def replacer(m: re.Match, _type: str = pii_type) -> str:
                val = m.group()
                if _type == "credit_card":
                    if not _luhn_check(val):
                        return val
                    digits = re.sub(r"\D", "", val)
                    return "****-****-****-" + digits[-4:]
                elif _type == "email":
                    parts = val.split("@")
                    return parts[0][0] + "***@" + parts[1]
                elif _type == "ssn":
                    return "***-**-" + val[-4:]
                elif _type == "phone":
                    clean = re.sub(r"\D", "", val)
                    return "***-***-" + clean[-4:]
                elif _type == "ip_address":
                    octets = val.split(".")
                    return f"***.***.***.{octets[-1]}"
                else:
                    if len(val) > 4:
                        return val[:2] + "*" * (len(val) - 4) + val[-2:]
                    return "*" * len(val)
            text = pattern.sub(replacer, text)
        return text

    def _sanitize(self, text: str) -> str:
        if self._strategy == "redact":
            return self._redact(text)
        elif self._strategy == "mask":
            return self._mask(text)
        return text

    def _sanitize_messages(self, messages: List[Any]) -> List[Any]:
        """Sanitize PII in a list of message objects."""
        cleaned = []
        for msg in messages:
            if hasattr(msg, "content"):
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    new_content = self._sanitize(content)
                    if new_content != content:
                        msg = msg.model_copy(update={"content": new_content}) if hasattr(msg, "model_copy") else msg
                cleaned.append(msg)
            elif isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
                if isinstance(content, str):
                    msg = {**msg, "content": self._sanitize(content)}
                cleaned.append(msg)
            else:
                cleaned.append(msg)
        return cleaned

    async def before_model(self, request: ModelRequest) -> Optional[ModelRequest]:
        if not self._apply_to_input:
            return None

        for msg in request.messages:
            content = ""
            if hasattr(msg, "content"):
                content = getattr(msg, "content", "")
            elif isinstance(msg, dict):
                content = msg.get("content", "")

            if isinstance(content, str):
                findings = self._detect(content)
                if findings and self._strategy == "block":
                    types_found = {f["type"] for f in findings}
                    raise PluginHalt(
                        f"PII detected in message: {', '.join(types_found)}. "
                        f"Request blocked by PIIGuardPlugin."
                    )

        if self._strategy in ("redact", "mask"):
            cleaned = self._sanitize_messages(request.messages)
            if cleaned != request.messages:
                return request.with_(messages=cleaned)

        return None

    async def after_model(self, request: ModelRequest, response: Any) -> Any:
        if not self._apply_to_output:
            return response

        if hasattr(response, "content") and isinstance(response.content, str):
            findings = self._detect(response.content)
            if findings:
                if self._strategy == "block":
                    raise PluginHalt(
                        f"PII detected in model response. Blocked by PIIGuardPlugin."
                    )
                sanitized = self._sanitize(response.content)
                if hasattr(response, "model_copy"):
                    return response.model_copy(update={"content": sanitized})
                logger.warning("Cannot sanitize response of type %s", type(response))

        return response
