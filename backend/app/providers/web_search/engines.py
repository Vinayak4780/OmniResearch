"""Web search adapters: Tavily, Brave Search, Exa, SearXNG. Each normalizes to WebSearchResponse."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import Field

from app.providers.base import (
    Capability,
    HealthResult,
    ProviderConfigBase,
    WebExtractionRequest,
    WebPage,
    WebPageExtractionProvider,
    WebSearchProvider,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
    ProviderError,
    ProviderErrorKind,
)
from app.providers.common import invalid, require
from app.providers.http import join_url


def _site_query(req: WebSearchRequest) -> str:
    q = req.query
    if req.include_domains:
        q += " (" + " OR ".join(f"site:{d}" for d in req.include_domains) + ")"
    for d in req.exclude_domains:
        q += f" -site:{d}"
    return q


def _time_range(days: int | None) -> str | None:
    if not days:
        return None
    return "day" if days <= 1 else "week" if days <= 7 else "month" if days <= 31 else "year"


class _SearchBase(WebSearchProvider):
    async def health_check(self) -> HealthResult:
        t0 = time.perf_counter()
        r = await self.search(WebSearchRequest(query="large language model inference", max_results=2))
        return HealthResult(ok=bool(r.results), latency_ms=(time.perf_counter() - t0) * 1000, detail=f"{len(r.results)} results", checked=["search"])


class TavilyConfig(ProviderConfigBase):
    base_url: str = "https://api.tavily.com"
    search_depth: str = Field(default="basic", description="basic | advanced")
    topic: str = Field(default="general", description="general | news")


class TavilyProvider(_SearchBase, WebPageExtractionProvider):
    adapter = "tavily"
    display_name = "Tavily Search + Extract"
    protocol = "native"
    capabilities = frozenset({Capability.WEB_SEARCH, Capability.WEB_EXTRACTION})
    description = "Tavily search API (/search) and content extraction (/extract)."
    limitations = ["Snippets are Tavily-generated excerpts; fetch the page for detailed claims.", "Extraction runs on Tavily servers."]
    config_model = TavilyConfig
    secret_fields = ["api_key"]
    config: TavilyConfig

    def model_for(self, capability: Capability) -> str | None:
        return f"tavily-{self.config.search_depth}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {require(self.secrets.get('api_key'), self.provider_id, 'api_key')}", "Content-Type": "application/json"}

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        body: dict[str, Any] = {
            "query": request.query,
            "max_results": request.max_results,
            "search_depth": self.config.search_depth,
            "topic": self.config.topic,
            "include_answer": False,
        }
        if request.include_domains:
            body["include_domains"] = request.include_domains
        if request.exclude_domains:
            body["exclude_domains"] = request.exclude_domains
        if tr := _time_range(request.recency_days):
            body["time_range"] = tr
        data = await self.http.request_json(self.provider_id, "POST", join_url(self.config.base_url, "search"), headers=self._headers(), json=body, timeout=self.config.timeout_s)
        items = data.get("results") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise invalid(self.provider_id, "tavily results missing")
        return WebSearchResponse(
            engine="tavily",
            results=[
                WebSearchResult(url=i["url"], title=i.get("title", ""), snippet=i.get("content", ""), published_at=i.get("published_date"), score=i.get("score"), source_engine="tavily")
                for i in items
                if isinstance(i, dict) and i.get("url")
            ],
        )

    async def extract(self, request: WebExtractionRequest) -> WebPage:
        data = await self.http.request_json(
            self.provider_id, "POST", join_url(self.config.base_url, "extract"), headers=self._headers(), json={"urls": [request.url]}, timeout=self.config.timeout_s
        )
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            failed = (data.get("failed_results") or [{}])[0] if isinstance(data, dict) else {}
            raise ProviderError(ProviderErrorKind.BAD_REQUEST, f"Tavily could not extract page: {failed.get('error', 'unknown')}", provider_id=self.provider_id)
        r = results[0]
        text = r.get("raw_content") or ""
        return WebPage(
            url=request.url,
            final_url=r.get("url", request.url),
            title=r.get("title") or "",
            text=text[: request.max_chars],
            fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            truncated=len(text) > request.max_chars,
            extractor="tavily",
        )


class BraveConfig(ProviderConfigBase):
    base_url: str = "https://api.search.brave.com/res/v1"
    country: str | None = None
    safesearch: str = "moderate"


class BraveSearchProvider(_SearchBase):
    adapter = "brave"
    display_name = "Brave Search API"
    protocol = "native"
    capabilities = frozenset({Capability.WEB_SEARCH})
    description = "Brave Web Search API (GET /web/search, X-Subscription-Token)."
    limitations = ["Domain filters are expressed with site: operators.", "Max 20 results per request."]
    config_model = BraveConfig
    secret_fields = ["api_key"]
    config: BraveConfig

    def model_for(self, capability: Capability) -> str | None:
        return "brave-web"

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        params: dict[str, Any] = {"q": _site_query(request), "count": min(request.max_results, 20), "safesearch": self.config.safesearch}
        if request.recency_days:
            d = request.recency_days
            params["freshness"] = "pd" if d <= 1 else "pw" if d <= 7 else "pm" if d <= 31 else "py"
        if self.config.country:
            params["country"] = self.config.country
        headers = {"Accept": "application/json", "X-Subscription-Token": require(self.secrets.get("api_key"), self.provider_id, "api_key")}
        data = await self.http.request_json(self.provider_id, "GET", join_url(self.config.base_url, "web/search"), headers=headers, params=params, timeout=self.config.timeout_s)
        items = ((data or {}).get("web") or {}).get("results", []) if isinstance(data, dict) else None
        if items is None:
            raise invalid(self.provider_id, "brave web.results missing")
        return WebSearchResponse(
            engine="brave",
            results=[
                WebSearchResult(url=i["url"], title=i.get("title", ""), snippet=i.get("description", ""), published_at=i.get("page_age") or i.get("age"), source_engine="brave")
                for i in items
                if i.get("url")
            ],
        )


class ExaConfig(ProviderConfigBase):
    base_url: str = "https://api.exa.ai"
    search_type: str = Field(default="auto", description="auto | neural | keyword | fast")
    snippet_chars: int = Field(default=600, ge=0, le=5000)


class ExaSearchProvider(_SearchBase):
    adapter = "exa"
    display_name = "Exa Search API"
    protocol = "native"
    capabilities = frozenset({Capability.WEB_SEARCH})
    description = "Exa /search with native domain and published-date filters."
    limitations = ["Recency uses startPublishedDate, which excludes pages without a known date."]
    config_model = ExaConfig
    secret_fields = ["api_key"]
    config: ExaConfig

    def model_for(self, capability: Capability) -> str | None:
        return f"exa-{self.config.search_type}"

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        body: dict[str, Any] = {"query": request.query, "numResults": request.max_results, "type": self.config.search_type}
        if request.include_domains:
            body["includeDomains"] = request.include_domains
        if request.exclude_domains:
            body["excludeDomains"] = request.exclude_domains
        if request.recency_days:
            body["startPublishedDate"] = (datetime.now(timezone.utc) - timedelta(days=request.recency_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if self.config.snippet_chars:
            body["contents"] = {"text": {"maxCharacters": self.config.snippet_chars}}
        headers = {"x-api-key": require(self.secrets.get("api_key"), self.provider_id, "api_key"), "Content-Type": "application/json"}
        data = await self.http.request_json(self.provider_id, "POST", join_url(self.config.base_url, "search"), headers=headers, json=body, timeout=self.config.timeout_s)
        items = data.get("results") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise invalid(self.provider_id, "exa results missing")
        return WebSearchResponse(
            engine="exa",
            results=[
                WebSearchResult(url=i["url"], title=i.get("title") or "", snippet=(i.get("text") or "")[:1000], published_at=i.get("publishedDate"), score=i.get("score"), source_engine="exa")
                for i in items
                if i.get("url")
            ],
        )


class SearXNGConfig(ProviderConfigBase):
    base_url: str = Field(..., description="SearXNG instance root with JSON format enabled, e.g. http://searxng:8080")
    engines: str = Field(default="", description="Comma-separated engines, empty = instance default")
    language: str = "en"


class SearXNGProvider(_SearchBase):
    adapter = "searxng"
    display_name = "SearXNG (self-hosted metasearch)"
    protocol = "native"
    capabilities = frozenset({Capability.WEB_SEARCH})
    description = "Self-hosted SearXNG instance via /search?format=json. No API key needed."
    limitations = ["The instance must enable the json output format.", "A local instance must be listed in OMNI_OUTBOUND_ALLOW_PRIVATE_HOSTS."]
    config_model = SearXNGConfig
    secret_fields = []
    config: SearXNGConfig

    def model_for(self, capability: Capability) -> str | None:
        return "searxng"

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        params: dict[str, Any] = {"q": _site_query(request), "format": "json", "language": self.config.language}
        if tr := _time_range(request.recency_days):
            params["time_range"] = "month" if tr == "week" else tr
        if self.config.engines:
            params["engines"] = self.config.engines
        data = await self.http.request_json(self.provider_id, "GET", join_url(self.config.base_url, "search"), params=params, headers={"Accept": "application/json"}, timeout=self.config.timeout_s)
        items = data.get("results") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise invalid(self.provider_id, "searxng results missing")
        return WebSearchResponse(
            engine="searxng",
            results=[
                WebSearchResult(url=i["url"], title=i.get("title", ""), snippet=i.get("content", ""), published_at=i.get("publishedDate"), score=i.get("score"), source_engine=i.get("engine"))
                for i in items[: request.max_results]
                if i.get("url")
            ],
        )
