"""Web research tools built on the WebSearchProvider and WebPageExtractionProvider routes.

Prompt-injection posture: the agent may only fetch URLs that came back from a search provider (or
that the user supplied). Text from fetched pages is never parsed for links to follow or for
instructions; it is only returned as evidence passages.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Literal

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from app.core.text import extract_numbers, match_key, tokenize
from app.providers.base import NoProviderAvailableError, ProviderError, Task, WebPage, WebSearchRequest, WebSearchResult
from app.providers.router import ProviderRouter
from app.repositories.db import Database, utcnow
from app.services.research.evidence import Evidence, domain_of, web_key

WEB_SETTINGS_KEY = "web_settings"


class WebSettings(BaseModel):
    enabled: bool = True
    mode: Literal["auto", "always", "never"] = "auto"
    max_results: int = Field(default=5, ge=1, le=20)
    max_pages_per_query: int = Field(default=3, ge=0, le=10)
    max_queries: int = Field(default=4, ge=1, le=12)
    recency_days: int | None = Field(default=None, ge=1, le=3650)
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    passages_per_page: int = Field(default=3, ge=1, le=10)
    page_max_chars: int = Field(default=60_000, ge=2_000, le=500_000)


def split_passages(text: str, target: int = 900) -> list[str]:
    """Split page text into passages that are exact substrings of the page text."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    cur = ""
    for p in paras:
        p = p.strip()
        if len(p) > target * 2:
            if cur:
                out.append(cur)
                cur = ""
            sents = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for s in sents:
                if buf and len(buf) + len(s) > target:
                    out.append(buf)
                    buf = s
                else:
                    buf = f"{buf} {s}".strip() if buf else s
            if buf:
                out.append(buf)
            continue
        if cur and len(cur) + len(p) > target:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return [o for o in out if len(o) > 40]


class WebResearchService:
    def __init__(self, db: Database, router: ProviderRouter):
        self.db = db
        self.router = router

    # ---------------------------------------------------------------- settings
    def settings(self) -> WebSettings:
        return WebSettings.model_validate(self.db.kv_get(WEB_SETTINGS_KEY, {}) or {})

    def save_settings(self, s: WebSettings) -> WebSettings:
        self.db.kv_set(WEB_SETTINGS_KEY, s.model_dump())
        return s

    def available(self) -> tuple[bool, str]:
        s = self.settings()
        if not s.enabled or s.mode == "never":
            return False, "web research disabled in settings"
        if not self.router.is_configured(Task.WEB_SEARCH):
            return False, "no web search provider configured"
        return True, ""

    def recent_errors(self, limit: int = 30) -> list[dict[str, Any]]:
        return self.db.all("SELECT * FROM web_fetch_log WHERE ok=0 ORDER BY id DESC LIMIT ?", (limit,))

    # ---------------------------------------------------------------- tools
    async def web_search(self, query: str, *, max_results: int | None = None) -> list[WebSearchResult]:
        """Tool: web_search."""
        s = self.settings()
        resp = await self.router.web_search(
            WebSearchRequest(
                query=query[:400],
                max_results=max_results or s.max_results,
                recency_days=s.recency_days,
                include_domains=s.include_domains,
                exclude_domains=s.exclude_domains,
            )
        )
        excluded = {d.lower() for d in s.exclude_domains}
        results = []
        for r in resp.results:
            dom = domain_of(r.url)
            if any(dom == d or dom.endswith("." + d) for d in excluded):
                continue
            if s.include_domains and not any(dom == d or dom.endswith("." + d) for d in (x.lower() for x in s.include_domains)):
                continue
            results.append(r)
        return results

    async def fetch_webpage(self, url: str) -> WebPage:
        """Tool: fetch_webpage (through the WEB_EXTRACTION route, which applies SSRF checks)."""
        s = self.settings()
        try:
            page = await self.router.extract_webpage(url, max_chars=s.page_max_chars)
            self.db.execute("INSERT INTO web_fetch_log(url, ok, status_code, error, created_at) VALUES (?,?,?,?,?)", (url, 1, page.status_code, None, utcnow()))
            return page
        except (ProviderError, NoProviderAvailableError) as exc:
            self.db.execute("INSERT INTO web_fetch_log(url, ok, status_code, error, created_at) VALUES (?,?,?,?,?)", (url, 0, getattr(exc, "status_code", None), str(exc)[:500], utcnow()))
            raise

    @staticmethod
    def extract_webpage_content(page: WebPage, query: str, k: int = 3) -> list[str]:
        """Tool: extract_webpage_content — the k passages most relevant to ``query`` (verbatim)."""
        passages = split_passages(page.text)
        if not passages:
            return []
        q = tokenize(query)
        if not q:
            return passages[:k]
        scores = BM25Okapi([tokenize(p) or ["_"] for p in passages]).get_scores(q)
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        return [p for p, sc in ranked[:k] if sc > 0] or passages[:1]

    @staticmethod
    def get_web_source_metadata(page: WebPage) -> dict[str, Any]:
        """Tool: get_web_source_metadata."""
        url = page.canonical_url or page.final_url or page.url
        return {
            "url": page.url,
            "final_url": page.final_url,
            "canonical_url": page.canonical_url,
            "domain": domain_of(url),
            "title": page.title,
            "site_name": page.site_name,
            "author": page.author,
            "published_at": page.published_at,
            "retrieved_at": page.fetched_at,
            "content_type": page.content_type,
            "truncated": page.truncated,
            "extractor": page.extractor,
        }

    @staticmethod
    def compare_web_sources(evidence: list[Evidence]) -> dict[str, Any]:
        """Tool: compare_web_sources — deterministic overview of which sources agree on numbers/terms."""
        by_domain: dict[str, list[Evidence]] = {}
        for e in evidence:
            if e.source_type == "web":
                by_domain.setdefault(e.domain or "?", []).append(e)
        numbers: dict[str, set[str]] = {}
        for dom, items in by_domain.items():
            for e in items:
                for n in extract_numbers(e.text):
                    numbers.setdefault(n, set()).add(dom)
        shared = {n: sorted(d) for n, d in numbers.items() if len(d) > 1}
        return {
            "domains": {d: [e.id for e in items] for d, items in by_domain.items()},
            "independent_domains": len(by_domain),
            "numbers_reported_by_multiple_domains": shared,
            "dated_sources": sum(1 for items in by_domain.values() for e in items if e.published_at),
        }

    @staticmethod
    def validate_web_citations(evidence: list[Evidence], pages: dict[str, WebPage]) -> list[dict[str, Any]]:
        """Tool: validate_web_citations — excerpt must occur verbatim in the fetched page text."""
        out = []
        for e in evidence:
            if e.source_type != "web":
                continue
            page = pages.get(e.url or "")
            if e.snippet_only or page is None:
                out.append({"evidence_id": e.id, "ok": False, "reason": "page not fetched; only a search snippet is available"})
                continue
            ok = match_key(e.text) in match_key(page.text)
            out.append({"evidence_id": e.id, "ok": ok, "reason": "excerpt found in page" if ok else "excerpt not found in fetched page"})
        return out

    # ---------------------------------------------------------------- composite step used by the agent
    async def research_many(
        self, queries: list[tuple[str, str | None]], pages_cache: dict[str, WebPage] | None = None, *, concurrency: int = 3
    ) -> tuple[list[Evidence], list[str]]:
        """Search every query, fetch each distinct result URL once, then extract verbatim passages per query."""
        s = self.settings()
        cache = pages_cache if pages_cache is not None else {}
        sem = asyncio.Semaphore(concurrency)
        warnings: list[str] = []

        async def search(q: str) -> list[WebSearchResult]:
            async with sem:
                try:
                    return (await self.web_search(q))[: s.max_pages_per_query]
                except (ProviderError, NoProviderAvailableError) as exc:
                    warnings.append(f"web search failed for '{q[:80]}': {exc}")
                    return []

        hits = await asyncio.gather(*(search(q) for q, _ in queries))
        failed: dict[str, str] = {}
        urls = list(dict.fromkeys(r.url for rs in hits for r in rs if r.url not in cache))

        async def fetch(url: str) -> None:
            async with sem:
                try:
                    page = await self.fetch_webpage(url)
                    if page.text.strip():
                        cache[url] = page
                    else:
                        failed[url] = "empty page"
                except (ProviderError, NoProviderAvailableError) as exc:
                    failed[url] = str(exc)

        await asyncio.gather(*(fetch(u) for u in urls))
        for url, err in failed.items():
            warnings.append(f"could not fetch {url}: {err}")
        evidence: list[Evidence] = []
        for (query, sqid), results in zip(queries, hits):
            evidence += self._passages(query, sqid, results, cache)
        return evidence, warnings

    async def research(self, query: str, subquery_id: str | None = None, pages_cache: dict[str, WebPage] | None = None) -> tuple[list[Evidence], list[str]]:
        return await self.research_many([(query, subquery_id)], pages_cache)

    def _passages(self, query: str, subquery_id: str | None, results: list[WebSearchResult], cache: dict[str, WebPage]) -> list[Evidence]:
        s = self.settings()
        evidence: list[Evidence] = []
        for r in results:
            page = cache.get(r.url)
            if page is not None:
                meta = self.get_web_source_metadata(page)
                for passage in self.extract_webpage_content(page, query, s.passages_per_page):
                    evidence.append(
                        Evidence(
                            key=web_key(r.url, passage),
                            source_type="web",
                            text=passage,
                            subquery=subquery_id,
                            url=r.url,
                            final_url=page.final_url,
                            canonical_url=page.canonical_url,
                            domain=meta["domain"],
                            web_title=page.title or r.title,
                            published_at=page.published_at or r.published_at,
                            retrieved_at=page.fetched_at,
                            search_engine=r.source_engine,
                            retrieval={"query": query, "search_score": r.score},
                        )
                    )
            elif r.snippet:
                evidence.append(
                    Evidence(
                        key=web_key(r.url, r.snippet),
                        source_type="web",
                        text=r.snippet,
                        subquery=subquery_id,
                        url=r.url,
                        domain=domain_of(r.url),
                        web_title=r.title,
                        published_at=r.published_at,
                        retrieved_at=utcnow(),
                        search_engine=r.source_engine,
                        snippet_only=True,
                        retrieval={"query": query, "search_score": r.score},
                    )
                )
        return evidence
