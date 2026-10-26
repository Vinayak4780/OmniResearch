"""Built-in webpage extraction: safe fetch + HTML/PDF/text content extraction, fully local."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from bs4 import BeautifulSoup
from pydantic import Field

from app.core.safe_fetch import FetchError, SafeFetcher
from app.providers.base import (
    Capability,
    HealthResult,
    ProviderConfigBase,
    ProviderError,
    ProviderErrorKind,
    WebExtractionRequest,
    WebPage,
    WebPageExtractionProvider,
)

_DROP_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe", "svg", "button", "template"]
_DATE_META = [
    ("meta", {"property": "article:published_time"}),
    ("meta", {"name": "citation_publication_date"}),
    ("meta", {"name": "citation_date"}),
    ("meta", {"name": "dc.date"}),
    ("meta", {"name": "DC.date.issued"}),
    ("meta", {"name": "date"}),
    ("meta", {"itemprop": "datePublished"}),
]

FETCH_ERROR_KIND = {
    "blocked": ProviderErrorKind.BLOCKED,
    "restricted": ProviderErrorKind.AUTH,
    "robots": ProviderErrorKind.BLOCKED,
    "timeout": ProviderErrorKind.TIMEOUT,
    "network": ProviderErrorKind.NETWORK,
    "too_large": ProviderErrorKind.BAD_REQUEST,
    "unsupported_type": ProviderErrorKind.UNSUPPORTED,
}


def extract_html(html: str, base_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip() or title
    canonical = None
    link = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v, list) else [v]))
    if link and link.get("href", "").startswith(("http://", "https://")):
        canonical = link["href"]
    site = soup.find("meta", property="og:site_name")
    author = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"name": "citation_author"})
    published = None
    for tag, attrs in _DATE_META:
        m = soup.find(tag, attrs=attrs)
        if m and m.get("content"):
            published = m["content"].strip()
            break
    if not published:
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(s.string or "")
            except (ValueError, TypeError):
                continue
            items = data if isinstance(data, list) else [data]
            for it in items:
                if isinstance(it, dict) and it.get("datePublished"):
                    published = str(it["datePublished"])
                    break
            if published:
                break
    if not published:
        t = soup.find("time", attrs={"datetime": True})
        if t:
            published = t["datetime"]

    for t in soup(_DROP_TAGS):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote", "td", "th", "figcaption", "caption"]):
        if el.find(["p", "li"]) and el.name in ("li", "blockquote", "td"):
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if len(text) < 2:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            text = "#" * int(el.name[1]) + " " + text
        blocks.append(text)
    if not blocks:
        blocks = [re.sub(r"\s+", " ", root.get_text(" ", strip=True))]
    deduped: list[str] = []
    seen: set[str] = set()
    for b in blocks:
        if b not in seen:
            seen.add(b)
            deduped.append(b)
    return {
        "title": title,
        "canonical_url": canonical,
        "site_name": site.get("content") if site else None,
        "author": author.get("content") if author else None,
        "published_at": published,
        "text": "\n\n".join(deduped),
    }


def extract_pdf(data: bytes) -> dict[str, Any]:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        meta = doc.metadata or {}
        pages = [f"[page {i + 1}]\n{p.get_text('text')}" for i, p in enumerate(doc)]
    return {"title": meta.get("title") or "", "author": meta.get("author") or None, "published_at": meta.get("creationDate") or None, "text": "\n\n".join(pages)}


class BuiltinExtractorConfig(ProviderConfigBase):
    respect_robots_txt: bool = True
    max_bytes: int = Field(default=3_000_000, ge=10_000, le=50_000_000)
    timeout_s: float = Field(default=20.0, gt=0, le=120)


class BuiltinWebExtractor(WebPageExtractionProvider):
    adapter = "builtin_web_extractor"
    display_name = "Built-in webpage extractor"
    protocol = "local"
    capabilities = frozenset({Capability.WEB_EXTRACTION})
    modalities = ["text"]
    description = "Fetches pages with SSRF protection and extracts main content, title, canonical URL and publication date (HTML, PDF, plain text)."
    limitations = [
        "Does not execute JavaScript; client-rendered pages may yield little text.",
        "Does not bypass logins, paywalls or robots.txt.",
    ]
    config_model = BuiltinExtractorConfig
    config: BuiltinExtractorConfig

    def __init__(self, *args: Any, fetcher_factory: Any = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._fetcher: SafeFetcher | None = None
        self._fetcher_factory = fetcher_factory

    def model_for(self, capability: Capability) -> str | None:
        return "builtin-html-pdf"

    def _get_fetcher(self) -> SafeFetcher:
        if self._fetcher is None:
            c = self.config
            if self._fetcher_factory:
                self._fetcher = self._fetcher_factory(c)
            else:
                self._fetcher = SafeFetcher(self.http.policy, max_bytes=c.max_bytes, timeout_s=c.timeout_s, respect_robots=c.respect_robots_txt)
        return self._fetcher

    async def extract(self, request: WebExtractionRequest) -> WebPage:
        try:
            res = await self._get_fetcher().fetch(request.url)
        except FetchError as exc:
            raise ProviderError(FETCH_ERROR_KIND.get(exc.kind, ProviderErrorKind.BAD_REQUEST), str(exc), provider_id=self.provider_id, status_code=exc.status_code) from exc
        ctype = res.content_type
        if ctype == "application/pdf" or res.body[:5] == b"%PDF-":
            try:
                info = extract_pdf(res.body)
            except Exception as exc:
                raise ProviderError(ProviderErrorKind.INVALID_RESPONSE, f"Could not parse PDF: {type(exc).__name__}", provider_id=self.provider_id) from exc
        elif ctype in ("text/html", "application/xhtml+xml", "") or ctype.endswith("+xml"):
            info = extract_html(res.body.decode("utf-8", "replace"), res.final_url)
        elif ctype.startswith("text/"):
            info = {"title": "", "text": res.body.decode("utf-8", "replace")}
        else:
            raise ProviderError(ProviderErrorKind.UNSUPPORTED, f"Unsupported content type {ctype}", provider_id=self.provider_id)
        text = info.get("text", "")
        truncated = res.truncated or len(text) > request.max_chars
        return WebPage(
            url=request.url,
            final_url=res.final_url,
            canonical_url=info.get("canonical_url"),
            title=info.get("title") or "",
            text=text[: request.max_chars],
            site_name=info.get("site_name"),
            author=info.get("author"),
            published_at=info.get("published_at"),
            fetched_at=res.fetched_at,
            status_code=res.status_code,
            content_type=ctype or None,
            truncated=truncated,
            extractor=self.adapter,
        )

    async def health_check(self) -> HealthResult:
        t0 = time.perf_counter()
        page = await self.extract(WebExtractionRequest(url="https://example.com/"))
        return HealthResult(ok=bool(page.text), latency_ms=(time.perf_counter() - t0) * 1000, detail=page.title or "fetched", checked=["fetch"])

    async def aclose(self) -> None:
        if self._fetcher:
            await self._fetcher.aclose()
