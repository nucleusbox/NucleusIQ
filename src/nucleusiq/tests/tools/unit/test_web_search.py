"""Unit tests for WebSearchTool adapters — no live network."""

from __future__ import annotations

from typing import Any

import pytest
from nucleusiq.agents.context.policy import ContextPolicy
from nucleusiq.tools.errors import ToolValidationError
from nucleusiq.tools.web_search import (
    DuckDuckGoSearch,
    SearchHit,
    WebSearchBackend,
    WebSearchBackendFactory,
    WebSearchProvider,
    WebSearchTool,
)
from nucleusiq.tools.web_search.backends._ddgs import _MISSING_DDGS, _search_sync
from nucleusiq.tools.web_search.format import (
    HARD_MAX_RESULTS,
    clamp_max_results,
    format_results,
)
from nucleusiq.tools.web_search.http import WebSearchHttpError, request_json


class _FakeDDGS:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.rows: list[dict[str, Any]] = [
            {
                "title": "Python",
                "href": "https://www.python.org/",
                "body": "The official Python website.",
            },
            {
                "title": "PEP 8",
                "href": "https://peps.python.org/pep-0008/",
                "body": "Style guide for Python code.",
            },
        ]
        self.error: Exception | None = None

    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return self.rows


def _install_ddgs(monkeypatch: pytest.MonkeyPatch, fake: _FakeDDGS) -> None:
    import sys

    def _factory(*args: Any, **kwargs: Any) -> _FakeDDGS:
        fake.init_kwargs = kwargs
        return fake

    class _Mod:
        DDGS = staticmethod(_factory)

    monkeypatch.setitem(sys.modules, "ddgs", _Mod())


@pytest.fixture()
def clean_factory() -> None:
    original = WebSearchBackendFactory._registry.copy()
    yield
    WebSearchBackendFactory._registry.clear()
    WebSearchBackendFactory._registry.update(original)


class TestFormatAndClamp:
    def test_format_numbered_title_url_snippet(self) -> None:
        text = format_results(
            "python",
            [
                SearchHit(
                    title="Python",
                    url="https://www.python.org/",
                    snippet="Official site",
                )
            ],
        )
        assert 'Search results for "python":' in text
        assert "1. Python" in text
        assert "URL: https://www.python.org/" in text
        assert "Official site" in text

    def test_format_empty(self) -> None:
        text = format_results("obscure", [])
        assert "No web results" in text
        assert "obscure" in text

    def test_clamp_bounds(self) -> None:
        assert clamp_max_results(0, 5) == 1
        assert clamp_max_results(99, 5) == HARD_MAX_RESULTS
        assert clamp_max_results("3", 5) == 3
        assert clamp_max_results("nope", 5) == 5


class TestFactoryAndConfig:
    def test_default_is_duckduckgo(self) -> None:
        tool = WebSearchTool()
        assert isinstance(tool.backend, DuckDuckGoSearch)
        assert tool.backend.name == "duckduckgo"

    def test_unknown_provider_lists_builtins(self) -> None:
        with pytest.raises(ToolValidationError, match="Unknown web-search provider"):
            WebSearchTool(provider="askjeeves")

    def test_google_requires_key_and_cse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
        monkeypatch.delenv("GOOGLE_SEARCH_ENGINE_ID", raising=False)
        with pytest.raises(ToolValidationError, match="api_key"):
            WebSearchTool(provider="google")
        with pytest.raises(ToolValidationError, match="search_engine_id"):
            WebSearchTool(provider="google", api_key="only-key")

    def test_google_accepts_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
        tool = WebSearchTool(provider=WebSearchProvider.GOOGLE)
        assert tool.backend.name == "google"

    @pytest.mark.parametrize(
        ("provider", "env"),
        [
            ("brave", "BRAVE_API_KEY"),
            ("tavily", "TAVILY_API_KEY"),
            ("serper", "SERPER_API_KEY"),
        ],
    )
    def test_paid_provider_requires_key(
        self,
        provider: str,
        env: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(env, raising=False)
        with pytest.raises(ToolValidationError, match="api_key"):
            WebSearchTool(provider=provider)

    def test_alias_ddg(self) -> None:
        tool = WebSearchTool(provider="ddg")
        assert tool.backend.name == "duckduckgo"

    def test_custom_backend_instance(self) -> None:
        class _Stub(WebSearchBackend):
            name = "stub"

            async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
                return [SearchHit(title="Hi", url="https://ex.test/", snippet=query)]

        tool = WebSearchTool(backend=_Stub())
        assert tool.backend.name == "stub"

    def test_register_custom_provider(self, clean_factory: None) -> None:
        class _Mine(WebSearchBackend):
            name = "mine"

            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

            async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
                return []

        WebSearchBackendFactory.register("mine", _Mine)
        tool = WebSearchTool(provider="mine", api_key="abc")
        assert isinstance(tool.backend, _Mine)
        assert tool.backend.kwargs["api_key"] == "abc"


class TestWebSearchTool:
    def test_spec_and_policy(self) -> None:
        tool = WebSearchTool()
        spec = tool.get_spec()
        assert spec["name"] == "web_search"
        assert spec["parameters"]["required"] == ["query"]
        assert tool.context_policy is ContextPolicy.EVIDENCE

    def test_exported_from_public_packages(self) -> None:
        from nucleusiq.tools import WebSearchBackendFactory as Factory
        from nucleusiq.tools import WebSearchTool as FromTools
        from nucleusiq.tools.builtin import WebSearchTool as FromBuiltin

        assert FromTools is WebSearchTool
        assert FromBuiltin is WebSearchTool
        assert Factory is WebSearchBackendFactory

    @pytest.mark.asyncio
    async def test_missing_query(self) -> None:
        result = await WebSearchTool().execute()
        assert "query" in result.lower()
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_duckduckgo_formats_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeDDGS()
        _install_ddgs(monkeypatch, fake)
        result = await WebSearchTool().execute(query="python pep")
        assert "1. Python" in result
        assert "https://www.python.org/" in result
        assert fake.calls[0][0] == "python pep"
        assert fake.calls[0][1]["backend"] == "duckduckgo"
        assert fake.calls[0][1]["max_results"] == 5

    @pytest.mark.asyncio
    async def test_bing_uses_bing_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeDDGS()
        _install_ddgs(monkeypatch, fake)
        await WebSearchTool(provider="bing").execute(query="news")
        assert fake.calls[0][1]["backend"] == "bing"

    @pytest.mark.asyncio
    async def test_max_results_hard_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeDDGS()
        _install_ddgs(monkeypatch, fake)
        await WebSearchTool().execute(query="x", max_results=50)
        assert fake.calls[0][1]["max_results"] == HARD_MAX_RESULTS

    @pytest.mark.asyncio
    async def test_rate_limit_is_recoverable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeDDGS()
        fake.error = Exception("202 Ratelimit")
        _install_ddgs(monkeypatch, fake)
        result = await WebSearchTool().execute(query="hot topic")
        assert "rate-limited" in result.lower()

    @pytest.mark.asyncio
    async def test_generic_failure_is_recoverable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeDDGS()
        fake.error = RuntimeError("backend exploded")
        _install_ddgs(monkeypatch, fake)
        result = await WebSearchTool().execute(query="x")
        assert "web search failed" in result
        assert "backend exploded" in result

    @pytest.mark.asyncio
    async def test_custom_backend_execute(self) -> None:
        class _Stub(WebSearchBackend):
            name = "stub"

            async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
                return [
                    SearchHit(
                        title="Custom",
                        url="https://ex.test/",
                        snippet=f"{query}:{max_results}",
                    )
                ]

        result = await WebSearchTool(backend=_Stub()).execute(query="hello")
        assert "1. Custom" in result
        assert "hello:5" in result

    def test_missing_ddgs_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "ddgs", None)
        with pytest.raises(RuntimeError, match="ddgs"):
            _search_sync(
                "q",
                max_results=1,
                region="us-en",
                safesearch="moderate",
                engine="duckduckgo",
                timeout=5,
            )
        assert "duckduckgo-search" in _MISSING_DDGS


class TestPaidHttpBackends:
    @pytest.mark.asyncio
    async def test_google_maps_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["params"]["key"] == "k"
            assert kwargs["params"]["cx"] == "cx"
            return {
                "items": [
                    {
                        "title": "Docs",
                        "link": "https://docs.example/",
                        "snippet": "Guide",
                    }
                ]
            }

        monkeypatch.setattr(
            "nucleusiq.tools.web_search.backends.google.request_json",
            _fake,
        )
        tool = WebSearchTool(
            provider="google",
            api_key="k",
            search_engine_id="cx",
        )
        result = await tool.execute(query="nucleusiq")
        assert "1. Docs" in result
        assert "https://docs.example/" in result

    @pytest.mark.asyncio
    async def test_http_429_is_recoverable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise WebSearchHttpError("HTTP 429 rate-limited", status=429)

        monkeypatch.setattr(
            "nucleusiq.tools.web_search.backends.brave.request_json",
            _boom,
        )
        result = await WebSearchTool(provider="brave", api_key="k").execute(query="x")
        assert "rate-limited" in result.lower()

    def test_request_json_builds_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _Resp:
            def read(self) -> bytes:
                return b'{"ok": true}'

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        def _urlopen(request: Any, timeout: float = 0) -> _Resp:
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            return _Resp()

        monkeypatch.setattr(
            "nucleusiq.tools.web_search.http.urllib.request.urlopen",
            _urlopen,
        )
        payload = request_json(
            "GET",
            "https://example.test/search",
            params={"q": "hello world"},
        )
        assert payload == {"ok": True}
        assert "q=hello+world" in captured["url"]
        assert captured["method"] == "GET"


class TestSearchSafety:
    @pytest.mark.asyncio
    async def test_drops_unsafe_urls_and_html(self) -> None:
        class _Stub(WebSearchBackend):
            name = "stub"

            async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
                return [
                    SearchHit("JS", "javascript:alert(1)", "bad"),
                    SearchHit("Local", "http://127.0.0.1/admin", "ssrf"),
                    SearchHit("File", "file:///etc/passwd", "disk"),
                    SearchHit(
                        "<b>Python</b>",
                        "https://www.python.org/",
                        "<script>alert(1)</script>Official site",
                    ),
                ]

        result = await WebSearchTool(backend=_Stub()).execute(query="python")
        assert "javascript" not in result
        assert "127.0.0.1" not in result
        assert "file://" not in result
        assert "<script>" not in result
        assert "<b>" not in result
        assert "1. Python" in result
        assert "Official site" in result

    @pytest.mark.asyncio
    async def test_allowed_domains_filters(self) -> None:
        seen: list[str] = []

        class _Stub(WebSearchBackend):
            name = "stub"

            async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
                seen.append(query)
                return [
                    SearchHit("Docs", "https://docs.python.org/3/", "docs"),
                    SearchHit("Random", "https://evil.example/", "nope"),
                    SearchHit("PEPs", "https://peps.python.org/pep-0008/", "pep"),
                ]

        tool = WebSearchTool(
            backend=_Stub(),
            allowed_domains=["python.org"],
        )
        result = await tool.execute(query="typing")
        assert seen[0] == "typing"
        assert "docs.python.org" in result
        assert "peps.python.org" in result
        assert "evil.example" not in result

    @pytest.mark.asyncio
    async def test_blocked_domains(self) -> None:
        class _Stub(WebSearchBackend):
            name = "stub"

            async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
                return [
                    SearchHit("Wiki", "https://en.wikipedia.org/wiki/Python", "ok"),
                    SearchHit("Spam", "https://tracker.spam.test/x", "no"),
                ]

        result = await WebSearchTool(
            backend=_Stub(),
            blocked_domains=["spam.test"],
        ).execute(query="python")
        assert "wikipedia.org" in result
        assert "spam.test" not in result

    @pytest.mark.asyncio
    async def test_allowlist_empty_message(self) -> None:
        class _Stub(WebSearchBackend):
            name = "stub"

            async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
                return [SearchHit("X", "https://other.test/", "x")]

        result = await WebSearchTool(
            backend=_Stub(),
            allowed_domains=["python.org"],
        ).execute(query="news")
        assert "allowed sites" in result
        assert "python.org" in result


class TestAgentUsesWebSearch:
    @pytest.mark.asyncio
    async def test_standard_mode_calls_web_search(self) -> None:
        import json

        from nucleusiq.agents.agent import Agent
        from nucleusiq.agents.config import AgentConfig, ExecutionMode
        from nucleusiq.agents.task import Task
        from nucleusiq.llms.mock_llm import MockLLM
        from nucleusiq.tests.conftest import make_test_prompt

        class _Stub(WebSearchBackend):
            name = "stub"

            async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
                return [
                    SearchHit(
                        "Python",
                        "https://www.python.org/",
                        f"hit for {query}",
                    )
                ]

        class _LLM(MockLLM):
            def _tool_call_response(self, messages, tools):  # type: ignore[no-untyped-def]
                fn_call = {
                    "name": "web_search",
                    "arguments": json.dumps({"query": "python typing"}),
                }
                tool_calls = [
                    {
                        "id": "call_web_1",
                        "type": "function",
                        "function": fn_call,
                    }
                ]
                msg = self.Message(
                    content=None, function_call=fn_call, tool_calls=tool_calls
                )
                return self.LLMResponse([self.Choice(msg)])

        agent = Agent(
            name="researcher",
            role="Researcher",
            objective="Answer with current web facts",
            llm=_LLM(),
            prompt=make_test_prompt(
                system="Use web_search for current facts.",
            ),
            tools=[WebSearchTool(backend=_Stub())],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )
        result = await agent.execute(Task(id="q1", objective="What is python typing?"))
        from nucleusiq.agents.agent_result import ResultStatus

        assert result.status is ResultStatus.SUCCESS
        assert result is not None
