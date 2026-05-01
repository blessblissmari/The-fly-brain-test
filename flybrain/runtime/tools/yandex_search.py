"""`yandex_search` — real `web_search` retriever backed by Yandex Search API
(README §17 / PLAN.md §621 stretch).

The tool target shape is intentionally identical to ``WebSearchTool``
(returns ``{"query": ..., "results": [{"title", "snippet", "url"}, ...]}``)
so it can drop into the same agent graphs that the fixture-based stub
serves. The two implementations live side-by-side: agents pick the
fixture in tests/CI for determinism, and the live retriever in
production / live-LLM evaluation runs.

Authentication
--------------

The Yandex Search API expects either an IAM token or a service-account
API key. We accept three env-var spellings (in priority order):

* ``YANDEX_SEARCH_API_KEY`` — service account API key with the
  ``search-api.webSearch.user`` role, recommended for batch usage.
* ``YANDEX_API_KEY`` / ``yandex_api_key`` — falls back to the
  ``YandexClient`` credentials so a single secret unblocks both LLM
  completions and search.
* ``YANDEX_FOLDER_ID`` / ``folder_id`` / ``yandex_folder_id`` — folder
  the search request is billed against.

Cost
----

Each request is billed per page, see
https://yandex.cloud/en/docs/search-api/pricing — at the time of
writing one parsed-response page is ≈ ₽0.40, so a single Retriever
agent step costs about the same as a tiny YandexGPT-Lite reply. The
`BudgetTracker` integration is intentionally optional (search calls
do not consume LLM tokens) and lives in the calling code.

Caching
-------

Live calls are slow (300 ms – 2 s) and non-free, so the retriever
keeps an in-process query → results cache. This is reset between
benchmark runs by constructing a fresh tool instance.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from flybrain.runtime.tools.base import ToolResult


@dataclass(slots=True)
class YandexSearchConfig:
    folder_id: str
    api_key: str
    """Service-account API key with `search-api.webSearch.user` role,
    or an IAM token (the SDK accepts both)."""

    region: str = "225"
    """Yandex search region, default ``225`` = Russia. ``2`` = Saint
    Petersburg, ``213`` = Moscow, ``187`` = Kyiv, etc."""

    max_results: int = 5
    """Top-N documents extracted from the parsed response and passed
    back to the agent. The Yandex API itself returns up to 100 per
    page; we always request page 0."""

    timeout_s: float = 15.0

    @classmethod
    def from_env(cls) -> YandexSearchConfig:
        folder_id = (
            os.environ.get("YANDEX_FOLDER_ID")
            or os.environ.get("folder_id")
            or os.environ.get("yandex_folder_id")
            or ""
        )
        api_key = (
            os.environ.get("YANDEX_SEARCH_API_KEY")
            or os.environ.get("YANDEX_API_KEY")
            or os.environ.get("yandex_api_key")
            or ""
        )
        if not folder_id:
            raise RuntimeError(
                "YANDEX_FOLDER_ID is not set; cannot init Yandex search retriever",
            )
        if not api_key:
            raise RuntimeError(
                "YANDEX_SEARCH_API_KEY (or YANDEX_API_KEY) is not set; "
                "cannot init Yandex search retriever",
            )
        return cls(folder_id=folder_id, api_key=api_key)


@dataclass(slots=True)
class YandexSearchTool:
    """Real `web_search` retriever using Yandex Search API.

    Drop-in replacement for ``WebSearchTool``: the result schema
    (``{"query": str, "results": [{"title", "snippet", "url"}, ...]}``)
    is identical, so existing Retriever agents and verifier tests do
    not need to change.
    """

    name: str = "web_search"
    config: YandexSearchConfig | None = None
    _sdk: Any = field(default=None, init=False, repr=False)
    _cache: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def _get_config(self) -> YandexSearchConfig:
        if self.config is None:
            self.config = YandexSearchConfig.from_env()
        return self.config

    def _get_sdk(self) -> Any:
        if self._sdk is None:
            try:
                from yandex_ai_studio_sdk import AsyncAIStudio
            except ImportError as e:  # pragma: no cover - runtime dep
                raise RuntimeError(
                    "yandex-ai-studio-sdk is not installed; "
                    "install with `uv pip install yandex-ai-studio-sdk`",
                ) from e
            cfg = self._get_config()
            self._sdk = AsyncAIStudio(folder_id=cfg.folder_id, auth=cfg.api_key)
        return self._sdk

    async def _search(self, query: str) -> list[dict[str, str]]:
        """Run a single `parsed`-format Yandex web search request."""
        cfg = self._get_config()
        sdk = self._get_sdk()
        search = sdk.search_api.web("RU").configure(
            search_type="ru",
            family_mode="moderate",
            fix_typo_mode="on",
            group_mode="deep",
            localization="ru",
            sort_order="desc",
            sort_mode="by_relevance",
            groups_on_page=max(5, cfg.max_results),
            region=cfg.region,
        )
        result = await search.run(query, format="parsed", page=0, timeout=cfg.timeout_s)
        return _parse_documents(result, cfg.max_results)

    def run(self, args: dict[str, Any]) -> ToolResult:
        query = args.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                name=self.name,
                ok=False,
                output=None,
                error="missing `query`",
                args=args,
            )

        if query in self._cache:
            return ToolResult(
                name=self.name,
                ok=True,
                output={"query": query, "results": list(self._cache[query])},
                args=args,
            )

        try:
            results = asyncio.run(self._search(query))
        except RuntimeError as e:
            # Either `from_env` raised (missing creds) or asyncio.run
            # was called inside an existing loop. Surface it cleanly.
            return ToolResult(
                name=self.name,
                ok=False,
                output=None,
                error=f"{type(e).__name__}: {e}",
                args=args,
            )
        except Exception as e:  # pragma: no cover - defensive net
            return ToolResult(
                name=self.name,
                ok=False,
                output=None,
                error=f"{type(e).__name__}: {e}",
                args=args,
            )

        self._cache[query] = results
        return ToolResult(
            name=self.name,
            ok=True,
            output={"query": query, "results": list(results)},
            args=args,
        )


def _parse_documents(result: Any, max_results: int) -> list[dict[str, str]]:
    """Project a parsed `WebSearchResult` into the `web_search` schema.

    The SDK exposes ``result.documents`` as a tuple of
    ``WebSearchDocument(url, domain, title, modtime, lang, passages, extra)``.
    We map the first ``max_results`` of them into the
    ``{title, snippet, url}`` shape the rest of the codebase already
    consumes (matching the fixture-based ``WebSearchTool`` schema).
    """
    docs = getattr(result, "documents", None) or []
    out: list[dict[str, str]] = []
    for doc in docs[:max_results]:
        title = getattr(doc, "title", None) or ""
        url = getattr(doc, "url", None) or ""
        passages = getattr(doc, "passages", None) or ()
        snippet = " ".join(p.strip() for p in passages if isinstance(p, str)).strip()
        out.append({"title": title, "snippet": snippet, "url": url})
    return out
