"""Unit tests for the Yandex Search retriever (`web_search` real impl).

The actual API call is mocked: we build a fake `result.documents`
matching the SDK's ``WebSearchDocument`` shape and assert that
``YandexSearchTool.run({...})`` returns the same output schema as
the fixture-based ``WebSearchTool`` so they are interchangeable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from flybrain.runtime.tools import (
    WebSearchTool,
    YandexSearchConfig,
    YandexSearchTool,
    default_tool_registry,
)
from flybrain.runtime.tools.yandex_search import _parse_documents

# -- fakes ---------------------------------------------------------------------


@dataclass
class _FakeDoc:
    title: str
    url: str
    passages: tuple[str, ...]
    domain: str = "example.com"
    modtime: Any = None
    lang: str = "ru"
    extra: dict[str, str] | None = None


@dataclass
class _FakeResult:
    documents: tuple[_FakeDoc, ...]


def _fake_documents(n: int = 3) -> tuple[_FakeDoc, ...]:
    return tuple(
        _FakeDoc(
            title=f"Result {i}",
            url=f"https://example.com/{i}",
            passages=(f"snippet {i} part A", f"snippet {i} part B"),
        )
        for i in range(n)
    )


# -- _parse_documents ----------------------------------------------------------


def test_parse_documents_projects_to_web_search_schema() -> None:
    """Real SDK result → `{title, snippet, url}` triples."""
    fake = _FakeResult(documents=_fake_documents(3))

    out = _parse_documents(fake, max_results=3)

    assert out == [
        {
            "title": "Result 0",
            "snippet": "snippet 0 part A snippet 0 part B",
            "url": "https://example.com/0",
        },
        {
            "title": "Result 1",
            "snippet": "snippet 1 part A snippet 1 part B",
            "url": "https://example.com/1",
        },
        {
            "title": "Result 2",
            "snippet": "snippet 2 part A snippet 2 part B",
            "url": "https://example.com/2",
        },
    ]


def test_parse_documents_respects_max_results() -> None:
    fake = _FakeResult(documents=_fake_documents(10))
    out = _parse_documents(fake, max_results=2)
    assert [r["url"] for r in out] == [
        "https://example.com/0",
        "https://example.com/1",
    ]


def test_parse_documents_handles_missing_fields() -> None:
    """Missing url/title/passages must yield empty strings, not crash."""
    doc = _FakeDoc(title=None, url=None, passages=())  # type: ignore[arg-type]
    out = _parse_documents(_FakeResult(documents=(doc,)), max_results=1)
    assert out == [{"title": "", "snippet": "", "url": ""}]


def test_parse_documents_handles_no_documents_attr() -> None:
    """Defensive: if the SDK schema drifts and ``documents`` is missing,
    we should return an empty list rather than crash."""

    class _NoDocs:
        pass

    out = _parse_documents(_NoDocs(), max_results=5)
    assert out == []


# -- YandexSearchConfig.from_env ----------------------------------------------


def test_yandex_search_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YANDEX_FOLDER_ID", "fld-123")
    monkeypatch.setenv("YANDEX_SEARCH_API_KEY", "sk-foo")
    cfg = YandexSearchConfig.from_env()
    assert cfg.folder_id == "fld-123"
    assert cfg.api_key == "sk-foo"


def test_yandex_search_config_falls_back_to_yandex_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YANDEX_FOLDER_ID", "fld-2")
    monkeypatch.delenv("YANDEX_SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("YANDEX_API_KEY", "sk-fallback")
    cfg = YandexSearchConfig.from_env()
    assert cfg.api_key == "sk-fallback"


def test_yandex_search_config_raises_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YANDEX_FOLDER_ID", "fld-3")
    monkeypatch.delenv("YANDEX_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.delenv("yandex_api_key", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY"):
        YandexSearchConfig.from_env()


# -- YandexSearchTool.run -----------------------------------------------------


def test_yandex_search_tool_returns_web_search_schema() -> None:
    """End-to-end: run() produces the same output shape as `WebSearchTool`."""
    cfg = YandexSearchConfig(folder_id="fld-x", api_key="sk-x", max_results=2)
    tool = YandexSearchTool(config=cfg)

    fake = _FakeResult(documents=_fake_documents(2))

    async def fake_search(self: YandexSearchTool, query: str) -> list[dict[str, str]]:
        return _parse_documents(fake, self._get_config().max_results)

    with patch.object(YandexSearchTool, "_search", fake_search):
        result = tool.run({"query": "fly brain optimizer"})

    assert result.ok
    assert result.name == "web_search"
    assert result.output is not None
    payload = result.output
    assert payload["query"] == "fly brain optimizer"
    assert len(payload["results"]) == 2
    assert {"title", "snippet", "url"} == set(payload["results"][0].keys())


def test_yandex_search_tool_caches_repeated_queries() -> None:
    """Identical queries should hit the in-process cache (no second SDK call)."""
    cfg = YandexSearchConfig(folder_id="fld-x", api_key="sk-x")
    tool = YandexSearchTool(config=cfg)
    call_count = {"n": 0}

    async def fake_search(self: YandexSearchTool, query: str) -> list[dict[str, str]]:
        call_count["n"] += 1
        return [{"title": "T", "snippet": "S", "url": "U"}]

    with patch.object(YandexSearchTool, "_search", fake_search):
        r1 = tool.run({"query": "fly brain"})
        r2 = tool.run({"query": "fly brain"})

    assert r1.ok and r2.ok
    assert call_count["n"] == 1, "second call should hit cache"
    assert r1.output == r2.output


def test_yandex_search_tool_rejects_empty_query() -> None:
    cfg = YandexSearchConfig(folder_id="fld-x", api_key="sk-x")
    tool = YandexSearchTool(config=cfg)
    result = tool.run({"query": "   "})
    assert not result.ok
    assert result.error is not None
    assert "query" in result.error


def test_yandex_search_tool_surfaces_sdk_errors() -> None:
    """If the underlying SDK raises, the tool returns ok=False with the
    error message instead of bubbling up."""
    cfg = YandexSearchConfig(folder_id="fld-x", api_key="sk-x")
    tool = YandexSearchTool(config=cfg)

    async def boom(self: YandexSearchTool, query: str) -> list[dict[str, str]]:
        raise ValueError("simulated transport failure")

    with patch.object(YandexSearchTool, "_search", boom):
        result = tool.run({"query": "fly brain"})

    assert not result.ok
    assert result.error is not None
    assert "simulated transport failure" in result.error


# -- default_tool_registry interplay ------------------------------------------


def test_default_tool_registry_uses_fixture_by_default() -> None:
    """No flag → fixture-based stub is registered under `web_search`."""
    reg = default_tool_registry()
    web = reg.get("web_search")
    assert isinstance(web, WebSearchTool)


def test_default_tool_registry_can_swap_to_live_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`live_search=True` swaps in `YandexSearchTool` (creds picked up
    from env at call time, not at registry construction)."""
    monkeypatch.setenv("YANDEX_FOLDER_ID", "fld-test")
    monkeypatch.setenv("YANDEX_SEARCH_API_KEY", "sk-test")

    reg = default_tool_registry(live_search=True)
    web = reg.get("web_search")
    assert isinstance(web, YandexSearchTool)


# -- compatibility with the fixture-based stub --------------------------------


def test_live_and_fixture_share_output_keys() -> None:
    """Each result dict from both tools should expose exactly the same
    keys so `Retriever`/`Researcher` agents don't care which one is wired."""
    fixture_tool = WebSearchTool(
        fixture={
            "fly": [
                {
                    "title": "Drosophila connectome",
                    "snippet": "200k neurons",
                    "url": "https://example.com/fly",
                }
            ]
        }
    )
    fixture_out = fixture_tool.run({"query": "fly brain"}).output
    assert fixture_out is not None
    fixture_keys = set(fixture_out["results"][0].keys())

    live_cfg = YandexSearchConfig(folder_id="fld-x", api_key="sk-x")
    live_tool = YandexSearchTool(config=live_cfg)

    async def fake_search(self: YandexSearchTool, query: str) -> list[dict[str, str]]:
        return _parse_documents(
            _FakeResult(documents=_fake_documents(1)),
            max_results=1,
        )

    with patch.object(YandexSearchTool, "_search", fake_search):
        live_result = live_tool.run({"query": "fly brain"})

    live_out = live_result.output
    assert live_out is not None
    live_keys = set(live_out["results"][0].keys())
    assert fixture_keys == live_keys


# -- SDK gating: skip live integration without creds -------------------------


def _has_live_creds() -> bool:
    import os

    return bool(os.environ.get("YANDEX_FOLDER_ID")) and bool(
        os.environ.get("YANDEX_SEARCH_API_KEY")
    )


@pytest.fixture
def _maybe_skip_live() -> Iterator[None]:
    if not _has_live_creds():
        pytest.skip(
            "live Yandex Search creds (YANDEX_FOLDER_ID + YANDEX_SEARCH_API_KEY) not set",
        )
    yield


@pytest.mark.skipif(
    "FLYBRAIN_RUN_LIVE_SEARCH" not in __import__("os").environ,
    reason="set FLYBRAIN_RUN_LIVE_SEARCH=1 to exercise the live Yandex Search API",
)
def test_yandex_search_live(_maybe_skip_live: None) -> None:  # pragma: no cover - live
    """Live integration: actually hits Yandex Search API. Costs ~₽0.40."""
    tool = YandexSearchTool()
    result = tool.run({"query": "Drosophila connectome FlyWire"})
    assert result.ok, result.error
    assert result.output is not None
    assert len(result.output["results"]) > 0
