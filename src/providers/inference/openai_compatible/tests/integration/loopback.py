"""A real HTTP server that speaks the OpenAI-compatible protocol.

Every other test in this package replaces ``llm._client`` with a fake, which
proves our logic but never exercises the parts between it and the network:
the ``openai`` SDK itself, URL resolution, header transmission, SSE framing
and real HTTP status codes.  This serves actual traffic on localhost so that
whole path runs.

It is not a model — responses are scripted.  It is the thinnest thing that is
still genuinely an OpenAI-compatible server.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["LoopbackServer", "RecordedRequest", "chat_response", "sse_chunk"]


@dataclass
class RecordedRequest:
    """One request as the server actually received it."""

    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any]

    def header(self, name: str) -> str | None:
        """Case-insensitive header lookup, as HTTP requires."""
        lowered = {k.lower(): v for k, v in self.headers.items()}
        return lowered.get(name.lower())


@dataclass
class Script:
    """What the server should do next."""

    status: int = 200
    body: Any = None
    sse: list[str] = field(default_factory=list)


def chat_response(
    content: str | None = "hello",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning: str | None = None,
    finish_reason: str = "stop",
    model: str = "test-model",
) -> dict[str, Any]:
    """Build a Chat Completions body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": "chatcmpl-loopback",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def sse_chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    model: str = "test-model",
) -> str:
    """Build one ``data:`` line of a streamed response."""
    payload: dict[str, Any] = {
        "id": "chatcmpl-loopback",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": model,
        "choices": [],
    }
    if usage is not None:
        # A usage trailer legitimately carries an empty choices list.
        payload["usage"] = usage
    else:
        delta: dict[str, Any] = {}
        if content is not None:
            delta["content"] = content
        if reasoning is not None:
            delta["reasoning"] = reasoning
        if tool_calls is not None:
            delta["tool_calls"] = tool_calls
        payload["choices"] = [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ]
    return json.dumps(payload)


class _Handler(BaseHTTPRequestHandler):
    server: LoopbackServer  # type: ignore[assignment]

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # noqa: D102
        pass  # Keep pytest output clean.

    # -- helpers ------------------------------------------------------- #

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _record(self, method: str, body: dict[str, Any]) -> None:
        owner = self.server.owner
        owner.requests.append(
            RecordedRequest(
                method=method,
                path=self.path,
                headers=dict(self.headers),
                body=body,
            )
        )

    def _send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("x-request-id", "loopback-req-1")
        self.end_headers()
        self.wfile.write(raw)

    def _send_sse(self, lines: list[str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for line in lines:
            self.wfile.write(f"data: {line}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # -- routes -------------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._record("GET", {})
        owner = self.server.owner

        if self.path.rstrip("/").endswith("/models"):
            if owner.models_status != 200:
                self._send_json(owner.models_status, {"error": "unavailable"})
                return
            self._send_json(200, {"object": "list", "data": owner.model_cards})
            return

        self._send_json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = self._read_body()
        self._record("POST", body)
        owner = self.server.owner

        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"no route {self.path}"}})
            return

        script = owner.next_script()

        if script.status != 200:
            self._send_json(script.status, script.body)
            return

        if body.get("stream"):
            self._send_sse(
                script.sse or [sse_chunk(content="hi", finish_reason="stop")]
            )
            return

        self._send_json(200, script.body or chat_response())


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    owner: LoopbackServer


class LoopbackServer:
    """A scriptable OpenAI-compatible server on localhost.

    Example:
        >>> with LoopbackServer() as server:  # doctest: +SKIP
        ...     server.queue(body=chat_response("hi"))
        ...     ...  # point a provider at server.base_url
    """

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self.model_cards: list[dict[str, Any]] = [
            {"id": "test-model", "object": "model", "max_model_len": 32_768}
        ]
        self.models_status = 200
        self._scripts: list[Script] = []
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------- #

    def __enter__(self) -> LoopbackServer:
        self._httpd = _Server(("127.0.0.1", 0), _Handler)
        self._httpd.owner = self
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- configuration -------------------------------------------------- #

    @property
    def port(self) -> int:
        assert self._httpd is not None, "server not started"
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def root_url(self) -> str:
        """Base URL without the ``/v1`` suffix, to test normalization."""
        return f"http://127.0.0.1:{self.port}"

    def queue(
        self,
        *,
        status: int = 200,
        body: Any = None,
        sse: list[str] | None = None,
    ) -> None:
        """Add one scripted response to the queue."""
        self._scripts.append(Script(status=status, body=body, sse=sse or []))

    def next_script(self) -> Script:
        """Pop the next script, repeating the last one once exhausted."""
        if not self._scripts:
            return Script(body=chat_response())
        if len(self._scripts) == 1:
            return self._scripts[0]
        return self._scripts.pop(0)

    # -- assertions ------------------------------------------------------ #

    @property
    def completions(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.method == "POST"]

    @property
    def model_lists(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.method == "GET"]
