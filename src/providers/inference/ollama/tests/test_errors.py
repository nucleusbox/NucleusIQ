"""Error mapping branches."""

from __future__ import annotations

import pytest
from nucleusiq.llms.errors import (
    InvalidRequestError,
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
)
from nucleusiq_ollama._shared.errors import map_ollama_response_error
from ollama import ResponseError


@pytest.mark.parametrize(
    "code,kind",
    [
        (404, ModelNotFoundError),
        (400, InvalidRequestError),
        (409, InvalidRequestError),
        (0, ProviderConnectionError),
        (502, ProviderConnectionError),
        (503, ProviderConnectionError),
        (500, ProviderServerError),
        (401, ProviderError),
    ],
)
def test_map_response_error(code: int, kind: type) -> None:
    exc = ResponseError("x", status_code=code)
    out = map_ollama_response_error(exc)
    assert isinstance(out, kind)
