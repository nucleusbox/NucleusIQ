"""
ModelFallbackPlugin — automatically tries fallback models on failure.

When the primary model fails, tries each fallback in order until one
succeeds or all are exhausted.

Usage::

    agent = Agent(
        ...,
        plugins=[
            ModelFallbackPlugin(
                fallbacks=["gpt-4o-mini", "gpt-3.5-turbo"],
                retry_on=(Exception,),
            )
        ],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Sequence, Tuple, Type

from nucleusiq.plugins.base import BasePlugin, ModelHandler, ModelRequest

logger = logging.getLogger(__name__)


class ModelFallbackPlugin(BasePlugin):
    """Tries fallback models in order when the primary model fails.

    Args:
        fallbacks: Ordered list of model names to try on failure.
        retry_on: Exception types that trigger fallback. Defaults to all exceptions.
    """

    def __init__(
        self,
        fallbacks: Sequence[str],
        retry_on: Tuple[Type[Exception], ...] | Type[Exception] = (Exception,),
    ) -> None:
        if not fallbacks:
            raise ValueError("At least one fallback model is required")
        self._fallbacks = list(fallbacks)
        self._retry_on = retry_on if isinstance(retry_on, tuple) else (retry_on,)

    @property
    def name(self) -> str:
        return "model_fallback"

    async def wrap_model_call(
        self, request: ModelRequest, handler: ModelHandler
    ) -> Any:
        try:
            return await handler(request)
        except self._retry_on as primary_err:
            logger.warning(
                "Primary model '%s' failed: %s. Trying fallbacks: %s",
                request.model,
                primary_err,
                self._fallbacks,
            )
            last_err: Exception = primary_err
            for fallback_model in self._fallbacks:
                try:
                    logger.info("Trying fallback model: %s", fallback_model)
                    return await handler(request.with_(model=fallback_model))
                except self._retry_on as fallback_err:
                    logger.warning(
                        "Fallback model '%s' also failed: %s",
                        fallback_model,
                        fallback_err,
                    )
                    last_err = fallback_err
            raise last_err
