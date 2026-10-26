"""Notebook sources from the web: add a link (PDF or webpage) and discover sources with web search.

A PDF link is downloaded and processed exactly like an upload. A webpage's main text is captured
into a PDF snapshot (title, original address and capture time on page 1), so it gets the same
parsing, chunking, page viewer and page-level citations as any other source, and citations keep
working if the page later changes or disappears.

Discovery only proposes links: nothing is added until the user picks results. Search results and
snippets are untrusted text; the ranking model sees them inside data fences and can only return
result numbers.
"""

from __future__ import annotations

import asyncio
import html
import io
import re
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field

from app.core.safe_fetch import FetchError, SafeFetcher
from app.observability.metrics import current_job_id
from app.providers.base import NoProviderAvailableError, ProviderError, Task
from app.providers.webpage_extraction.builtin import extract_html
from app.services.research.evidence import _attr, domain_of
from app.services.research.llm import StructuredOutputError, chat_json

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

MAX_LINKS = 10
MIN_PAGE_CHARS = 200
SNAPSHOT_MAX_CHARS = 200_000
NO_SEARCH = "Web search needs a search service. Connect Tavily, Brave or Exa in Settings → Connect AI."

FETCH_MESSAGES = {
    "robots": "This site does not allow automated access (robots.txt).",
    "restricted": "The page needs a login or refuses access.",
    "blocked": "That address is not allowed (private network or blocked site).",
    "timeout": "The site took too long to respond.",
    "network": "Could not reach the site.",
    "too_large": "The file is larger than the upload limit.",
}


def snapshot_pdf(title: str, url: str, text: str, captured_at: str) -> bytes:
    """Render extracted page text (paragraphs, '#'-prefixed headings) into a simple, searchable PDF."""
    import pymupdf

    parts = [f"<h1>{html.escape(title)}</h1>", f"<p class='src'>Captured from {html.escape(url)} on {html.escape(captured_at)}</p>"]
    for block in re.split(r"\n\s*\n", text[:SNAPSHOT_MAX_CHARS]):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", block)
        if m:
            level = min(len(m.group(1)) + 1, 4)
            parts.append(f"<h{level}>{html.escape(m.group(2))}</h{level}>")
        else:
            parts.append("<p>" + html.escape(block).replace("\n", "<br/>") + "</p>")
    css = "body{font-family:sans-serif;font-size:10.5pt;line-height:1.35} h1{font-size:17pt} h2{font-size:14pt} h3,h4{font-size:12pt} p{margin-bottom:6pt} .src{color:#555;font-size:9pt}"
    story = pymupdf.Story(html="".join(parts), user_css=css)
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    page = pymupdf.paper_rect("a4")
    where = page + (54, 54, -54, -54)
    more = True
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    with pymupdf.open(stream=buf.getvalue(), filetype="pdf") as doc:
        doc.set_metadata({"title": title[:200], "subject": url[:500], "creator": "OmniResearch web capture"})
        return doc.tobytes(garbage=3, deflate=True)


def _filename(url: str, title: str | None, ext: str = ".pdf") -> str:
    if title:
        base = title
    else:
        path = unquote(urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]) or domain_of(url)
        base = re.sub(r"\.pdf$", "", path, flags=re.I)
    base = re.sub(r"[^\w\s.-]+", " ", base).strip()
    base = re.sub(r"\s+", " ", base)[:100] or "web source"
    return base + ext


class DiscoverQueries(BaseModel):
    queries: list[str] = Field(description="3–4 diverse web search queries, each under 12 words")


class DiscoverPick(BaseModel):
    n: int = Field(description="Result number")
    reason: str = Field(default="", description="One short sentence: why this result is useful for the topic")


class DiscoverPicks(BaseModel):
    picks: list[DiscoverPick] = Field(default_factory=list)


class NotebookWebService:
    def __init__(self, c: "Container"):
        self.c = c
        self.db = c.db
        self.make_fetcher = lambda: SafeFetcher(c.policy, max_bytes=c.settings.max_upload_mb * 1024 * 1024, timeout_s=30)

    # ---------------------------------------------------------------- add links
    async def add_links(self, nb_id: str, urls: list[str]) -> list[dict[str, Any]]:
        self.c.notebooks._row(nb_id)
        urls = list(dict.fromkeys(u.strip() for u in urls if u and u.strip()))
        if not urls:
            raise ValueError("Paste at least one link")
        if len(urls) > MAX_LINKS:
            raise ValueError(f"Add at most {MAX_LINKS} links at a time")
        fetcher = self.make_fetcher()
        sem = asyncio.Semaphore(3)

        async def one(url: str) -> dict[str, Any]:
            async with sem:
                try:
                    return {"url": url, "ok": True, **await self._add_one(nb_id, url, fetcher)}
                except ValueError as exc:
                    return {"url": url, "ok": False, "error": str(exc)}

        try:
            results = await asyncio.gather(*(one(u) for u in urls))
        finally:
            await fetcher.aclose()
        if any(r["ok"] for r in results):
            self.c.notebooks.touch(nb_id)
        return results

    async def _add_one(self, nb_id: str, url: str, fetcher: SafeFetcher) -> dict[str, Any]:
        if not re.match(r"^https?://[^\s/]+", url, re.I):
            raise ValueError("Links must start with http:// or https://")
        dup = self.db.one("SELECT * FROM documents WHERE collection=? AND source_url=?", (nb_id, url))
        if dup:
            return {"document": dup, "duplicate": True, "kind": "pdf" if url.lower().endswith(".pdf") else "web"}
        try:
            res = await fetcher.fetch(url)
        except FetchError as exc:
            raise ValueError(FETCH_MESSAGES.get(exc.kind, f"The site returned an error ({exc})")) from exc
        ctype = res.content_type
        if ctype == "application/pdf" or res.body[:5] == b"%PDF-":
            if res.truncated:
                raise ValueError(FETCH_MESSAGES["too_large"])
            out = self.c.ingestion.submit(res.body, _filename(res.final_url, None), nb_id, source_url=url)
            return {"document": out["document"], "duplicate": out["duplicate"], "kind": "pdf"}
        if ctype in ("text/html", "application/xhtml+xml", "") or ctype.endswith("+xml"):
            info = extract_html(res.body.decode("utf-8", "replace"), res.final_url)
        elif ctype.startswith("text/"):
            info = {"title": "", "text": res.body.decode("utf-8", "replace")}
        else:
            raise ValueError(f"Unsupported content ({ctype or 'unknown type'}); links must point to a webpage or a PDF")
        text = (info.get("text") or "").strip()
        if len(text) < MIN_PAGE_CHARS:
            raise ValueError("The page has almost no readable text — it may need a login or JavaScript. Save it as a PDF and upload that instead.")
        title = (info.get("title") or "").strip() or domain_of(res.final_url)
        pdf = await asyncio.to_thread(snapshot_pdf, title, res.final_url, text, res.fetched_at)
        out = self.c.ingestion.submit(pdf, _filename(url, title), nb_id, source_url=url)
        return {"document": out["document"], "duplicate": out["duplicate"], "kind": "web"}

    # ---------------------------------------------------------------- discover
    async def discover(self, nb_id: str, query: str, mode: Literal["fast", "deep"] = "fast") -> dict[str, Any]:
        self.c.notebooks._row(nb_id)
        query = (query or "").strip()
        if not 2 <= len(query) <= 400:
            raise ValueError("Describe what to look for in 2–400 characters")
        ok, _ = self.c.web.available()
        if not ok:
            raise ValueError(NO_SEARCH)
        current_job_id.set(f"notebook:{nb_id}")
        warnings: list[str] = []
        queries = [query]
        if mode == "deep":
            titles = [r["title"] or r["filename"] for r in self.db.all("SELECT title, filename FROM documents WHERE collection=? LIMIT 15", (nb_id,))]
            try:
                q = await chat_json(
                    self.c.router,
                    Task.PLANNING,
                    DiscoverQueries,
                    system="You write web search queries that together find the best, most authoritative sources on a topic: primary papers, official docs, and strong overviews. Make the queries differ in angle.",
                    user=f"<topic>\n{query}\n</topic>\n<existing_sources>\n" + "\n".join(f"- {_attr(t)}" for t in titles) + "\n</existing_sources>",
                    max_tokens=1200,
                    temperature=0.3,
                )
                queries += [s.strip() for s in q.queries if s.strip() and s.strip().lower() != query.lower()][:4]
            except (ProviderError, NoProviderAvailableError, StructuredOutputError) as exc:
                warnings.append(f"Could not plan extra searches ({exc}); used your query only.")

        async def search(q: str) -> list:
            try:
                return await self.c.web.web_search(q, max_results=10 if mode == "fast" else 8)
            except (ProviderError, NoProviderAvailableError) as exc:
                warnings.append(f"Search failed for '{q[:60]}': {exc}")
                return []

        hits = await asyncio.gather(*(search(q) for q in queries))
        seen: dict[str, Any] = {}
        for rs in hits:
            for r in rs:
                if r.url and r.url.startswith(("http://", "https://")) and r.url not in seen:
                    seen[r.url] = r
        candidates = list(seen.values())
        if not candidates and warnings and len(warnings) >= len(queries):
            raise ValueError(warnings[0])
        reasons: dict[str, str] = {}
        if mode == "deep" and len(candidates) > 1:
            listing = "\n".join(
                f'<result n="{i}" domain="{_attr(domain_of(r.url))}" title="{_attr((r.title or "")[:150])}">{_attr((r.snippet or "")[:300])}</result>'
                for i, r in enumerate(candidates, 1)
            )
            try:
                picks = await chat_json(
                    self.c.router,
                    Task.PLANNING,
                    DiscoverPicks,
                    system="Pick the up to 10 search results most useful as research sources on the topic. Prefer primary sources, official documentation and substantive articles over listicles, forums and duplicates.",
                    user=f"<topic>\n{query}\n</topic>\n\n{listing}",
                    max_tokens=1500,
                    temperature=0.1,
                )
                order = [p for p in picks.picks if 1 <= p.n <= len(candidates)]
                if order:
                    ranked = list(dict.fromkeys(candidates[p.n - 1].url for p in order))[:10]
                    reasons = {candidates[p.n - 1].url: p.reason.strip()[:200] for p in order}
                    candidates = [seen[u] for u in ranked]
            except (ProviderError, NoProviderAvailableError, StructuredOutputError) as exc:
                warnings.append(f"Could not rank results ({exc}); showing them in search order.")
        added = {r["source_url"] for r in self.db.all("SELECT source_url FROM documents WHERE collection=? AND source_url IS NOT NULL", (nb_id,))}
        results = [
            {
                "url": r.url,
                "title": r.title or domain_of(r.url),
                "snippet": r.snippet or "",
                "domain": domain_of(r.url),
                "published_at": r.published_at,
                "is_pdf": urlsplit(r.url).path.lower().endswith(".pdf"),
                "reason": reasons.get(r.url),
                "already_added": r.url in added,
            }
            for r in candidates[:10]
        ]
        return {"mode": mode, "query": query, "queries": queries, "results": results, "warnings": warnings}
