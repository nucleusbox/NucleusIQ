"""AttachmentGuardPlugin -- validate task attachments before execution.

Mirrors the ``ToolGuardPlugin`` pattern.  Runs in the ``before_agent``
hook and raises ``PluginHalt`` when an attachment violates policy.

Usage::

    agent = Agent(
        ...,
        plugins=[
            AttachmentGuardPlugin(
                allowed_types=[AttachmentType.TEXT, AttachmentType.IMAGE_URL],
                max_file_size=10 * 1024 * 1024,  # 10 MB
                max_attachments=5,
                allowed_extensions=[".txt", ".csv", ".png", ".jpg"],
            ),
        ],
    )
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

from nucleusiq.agents.attachments import Attachment, AttachmentType
from nucleusiq.plugins.base import AgentContext, BasePlugin
from nucleusiq.plugins.errors import PluginError, PluginHalt

logger = logging.getLogger(__name__)


class AttachmentGuardPlugin(BasePlugin):
    """Validates task attachments against configurable rules.

    Operates on four independent checks (all optional):

    1. **Type filter** -- ``allowed_types`` or ``blocked_types`` (mutually
       exclusive).
    2. **Max file size** -- per-attachment byte limit.
    3. **Max count** -- total number of attachments.
    4. **Extension filter** -- ``allowed_extensions`` whitelist.

    Any violation raises ``PluginHalt`` with a human-readable message.
    """

    def __init__(
        self,
        *,
        allowed_types: Sequence[AttachmentType] | None = None,
        blocked_types: Sequence[AttachmentType] | None = None,
        max_file_size: int | None = None,
        max_attachments: int | None = None,
        allowed_extensions: Sequence[str] | None = None,
    ) -> None:
        if allowed_types and blocked_types:
            raise PluginError("Cannot specify both 'allowed_types' and 'blocked_types'")

        self._allowed_types = set(allowed_types) if allowed_types else None
        self._blocked_types = set(blocked_types) if blocked_types else None
        self._max_file_size = max_file_size
        self._max_attachments = max_attachments
        self._allowed_extensions = (
            {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in allowed_extensions
            }
            if allowed_extensions
            else None
        )

    @property
    def name(self) -> str:
        return "attachment_guard"

    async def before_agent(self, ctx: AgentContext) -> AgentContext | None:
        task = ctx.task
        attachments: list[Attachment] | None = getattr(task, "attachments", None)
        if not attachments:
            return None

        if (
            self._max_attachments is not None
            and len(attachments) > self._max_attachments
        ):
            msg = (
                f"Too many attachments: {len(attachments)} "
                f"(max {self._max_attachments})"
            )
            logger.warning("AttachmentGuard: %s", msg)
            raise PluginHalt(msg)

        for att in attachments:
            self._check_type(att)
            self._check_size(att)
            self._check_extension(att)

        logger.debug("AttachmentGuard: PASSED %d attachment(s)", len(attachments))
        return None

    def _check_type(self, att: Attachment) -> None:
        if self._blocked_types and att.type in self._blocked_types:
            msg = f"Attachment type '{att.type.value}' is blocked by policy"
            logger.warning("AttachmentGuard: %s", msg)
            raise PluginHalt(msg)
        if self._allowed_types and att.type not in self._allowed_types:
            msg = (
                f"Attachment type '{att.type.value}' is not in allowed types: "
                f"{sorted(t.value for t in self._allowed_types)}"
            )
            logger.warning("AttachmentGuard: %s", msg)
            raise PluginHalt(msg)

    def _check_size(self, att: Attachment) -> None:
        if self._max_file_size is None:
            return
        size = len(att.data) if isinstance(att.data, (bytes, str)) else 0
        if size > self._max_file_size:
            name = att.name or "(unnamed)"
            mb = self._max_file_size / (1024 * 1024)
            msg = (
                f"Attachment '{name}' is {size:,} bytes, "
                f"exceeding the {mb:.1f} MB limit"
            )
            logger.warning("AttachmentGuard: %s", msg)
            raise PluginHalt(msg)

    def _check_extension(self, att: Attachment) -> None:
        if self._allowed_extensions is None or att.name is None:
            return
        ext = os.path.splitext(att.name)[1].lower()
        if ext and ext not in self._allowed_extensions:
            msg = (
                f"Attachment '{att.name}' has extension '{ext}' which is "
                f"not in allowed extensions: {sorted(self._allowed_extensions)}"
            )
            logger.warning("AttachmentGuard: %s", msg)
            raise PluginHalt(msg)
