"""Notebook sources from links, web discovery (fast/deep), web-assisted chat and Studio outputs."""

from __future__ import annotations

import json
import re

import httpx

from app.core.netguard import OutboundPolicy
from app.core.safe_fetch import SafeFetcher
from app.services.notebook_studio import uncited_statements
from app.services.notebook_web import snapshot_pdf
from tests import fakes
from tests.test_agent import find, sources
from tests.test_api import CORPUS, make_client, wait_until

ARTICLE = (
    "<html><head><title>Serving LLMs on a budget</title></head><body><nav>menu</nav><article>"
    "<h1>Serving LLMs on a budget</h1><p>Weight quantization to INT4 lets a 13B model fit in 7.4 GB of GPU memory, "
    "which makes single-GPU serving practical for small teams.</p><h2>Paging</h2><p>Paged KV caches reduce fragmentation "
    "and raise throughput when many requests share one GPU. This article compares both approaches.</p></article></body></html>"
)


def site(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/paper.pdf":
        return httpx.Response(200, content=(CORPUS / "synthetic_kvcache.pdf").read_bytes(), headers={"content-type": "application/pdf"})
    if path == "/article":
        return httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html; charset=utf-8"})
    if path == "/login":
        return httpx.Response(403, text="forbidden")
    if path == "/empty":
        return httpx.Response(200, text="<html><body><div id=app></div></body></html>", headers={"content-type": "text/html"})
    return httpx.Response(404)


def setup(client, c, web: bool = False, script: str = "nbweb"):
    providers = [("llm", "fake_chat", {"label": "l", "script": script}), ("emb", "fake_embed", {"label": "e", "dims": 24}), ("rr", "local_lexical_rerank", {})]
    routes = [("synthesis", "llm"), ("text_embedding", "emb"), ("reranking", "rr")]
    if web:
        providers.append(("web", "fake_web", {"label": "w"}))
        routes += [("web_search", "web"), ("web_extraction", "web")]
    for pid, adapter, cfg in providers:
        client.post("/api/providers", json={"id": pid, "name": pid, "adapter": adapter, "config": cfg})
    for task, pid in routes:
        client.put(f"/api/routes/{task}", json={"task": task, "provider_ids": [pid]})
    c.notebook_web.make_fetcher = lambda: SafeFetcher(OutboundPolicy(resolve_dns=False), respect_robots=False, client=httpx.AsyncClient(transport=httpx.MockTransport(site)))


def test_snapshot_pdf_is_searchable():
    import pymupdf

    pdf = snapshot_pdf("A <title>", "https://example.org/a", "# Heading\n\nFirst paragraph with 7.4 GB.\n\n" + "More text. " * 900, "2026-10-25T00:00:00Z")
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        assert doc.page_count >= 2 and doc.metadata["title"] == "A <title>"
        first = doc[0].get_text()
        assert "Captured from https://example.org/a" in first and "7.4 GB" in first and "Heading" in first


def test_uncited_statements_counts_only_claims():
    md = "## Key findings\n- INT4 needs 7.4 GB of memory on one GPU [E1]\n- Paged caches reduce fragmentation for many requests\n### What is paging?\n\n"
    assert uncited_statements(md) == 1


def test_add_links_pdf_and_webpage(tmp_path):
    client, c = make_client(tmp_path)
    with client:
        setup(client, c)
        nb = client.post("/api/notebooks", json={"title": "links"}).json()["id"]
        urls = ["https://example.org/paper.pdf", "https://example.org/article", "https://example.org/login", "https://example.org/empty", "ftp://x"]
        res = {r["url"]: r for r in client.post(f"/api/notebooks/{nb}/sources/links", json={"urls": urls}).json()["results"]}
        assert res[urls[0]]["ok"] and res[urls[0]]["kind"] == "pdf"
        assert res[urls[1]]["ok"] and res[urls[1]]["kind"] == "web"
        assert not res[urls[2]]["ok"] and "login" in res[urls[2]]["error"]
        assert not res[urls[3]]["ok"] and "readable text" in res[urls[3]]["error"]
        assert not res[urls[4]]["ok"] and "http" in res[urls[4]]["error"]
        wait_until(lambda: all(s["status"].startswith("ready") for s in client.get(f"/api/notebooks/{nb}/sources").json()))
        srcs = {s["source_url"]: s for s in client.get(f"/api/notebooks/{nb}/sources").json()}
        assert set(srcs) == {urls[0], urls[1]}
        web_doc = srcs[urls[1]]["id"]
        text = " ".join(ch["text"] for ch in client.get(f"/api/documents/{web_doc}/chunks").json())
        assert "7.4 GB" in text and "Captured from" in text and "menu" not in text
        again = client.post(f"/api/notebooks/{nb}/sources/links", json={"urls": [urls[1]]}).json()["results"][0]
        assert again["duplicate"] is True and again["document"]["id"] == web_doc

        # the same file in another notebook is a separate source, not the first notebook's copy
        nb2 = client.post("/api/notebooks", json={"title": "other"}).json()["id"]
        up = client.post("/api/documents", files={"file": ("k.pdf", (CORPUS / "synthetic_kvcache.pdf").read_bytes(), "application/pdf")}, data={"collection": nb2}).json()
        assert up["duplicate"] is False and up["document"]["collection"] == nb2


def nbweb_llm(state: dict):
    def handler(req, label):
        prompt = "\n".join(m.content for m in req.messages)
        if "web search queries that together" in prompt:
            return json.dumps({"queries": ["int4 quantization memory", "paged kv cache"]})
        if "Pick the up to 10" in prompt:
            return json.dumps({"picks": [{"n": 2, "reason": "Primary source on paging"}, {"n": 1, "reason": "Overview"}]})
        if "answering questions about the user's sources" in prompt:
            src = sources(prompt)
            q = find(src, "7.4 GB")
            state["web_prompt"] = prompt
            return json.dumps({"answer": f"INT4 needs 7.4 GB [{q}].", "citations": [q]})
        if "You build a mind map" in prompt:
            src = sources(prompt)
            first = next(iter(src))
            return json.dumps({"central": "LLM serving", "branches": [{"label": "Memory", "citations": [first], "children": [{"label": "INT4", "citations": [first]}, {"label": "Uncited leaf"}]}]})
        if "study and briefing material" in prompt:
            src = sources(prompt)
            q = find(src, "7.4 GB") or next(iter(src))
            state["studio_prompt"] = prompt
            return json.dumps({"content": f"## Key findings\n- INT4 weights fit in 7.4 GB [{q}]\n- Throughput rises by 123.4 percent overall [{q}]\n- A statement with no citation at all here\n- Bad ref [E99]"})
        return json.dumps({"summary": "s", "questions": ["a?", "b?", "c?"]})

    return handler


def test_discover_and_web_chat(tmp_path):
    client, c = make_client(tmp_path)
    st: dict = {}
    fakes.CHAT_HANDLERS["nbweb"] = nbweb_llm(st)
    with client:
        setup(client, c)
        nb = client.post("/api/notebooks", json={"title": "web"}).json()["id"]
        r = client.post(f"/api/notebooks/{nb}/discover", json={"query": "llm serving", "mode": "fast"})
        assert r.status_code == 400 and "search service" in r.json()["detail"]
        r = client.post(f"/api/notebooks/{nb}/messages", json={"message": "How much memory?", "web": True})
        assert r.status_code == 400 and "search service" in r.json()["detail"]

        setup(client, c, web=True)
        fakes.FakeSearchProvider.PAGES = {
            "https://a.example/int4": {"title": "INT4 overview", "text": "Quantizing weights to INT4 needs 7.4 GB for a 13B model, measured on one GPU."},
            "https://b.example/paging": {"title": "Paged attention", "text": "Paged KV caches cut fragmentation and raise serving throughput substantially."},
        }
        fast = client.post(f"/api/notebooks/{nb}/discover", json={"query": "llm serving", "mode": "fast"}).json()
        assert [x["url"] for x in fast["results"]] == list(fakes.FakeSearchProvider.PAGES) and fast["queries"] == ["llm serving"]
        deep = client.post(f"/api/notebooks/{nb}/discover", json={"query": "llm serving", "mode": "deep"}).json()
        assert len(deep["queries"]) == 3 and deep["results"][0]["url"] == "https://b.example/paging" and deep["results"][0]["reason"] == "Primary source on paging"

        # fast answer from the web alone (no sources yet): the cited passage links to the page
        ans = client.post(f"/api/notebooks/{nb}/messages", json={"message": "How much memory does INT4 need?", "web": True}).json()
        assert "7.4 GB" in ans["content"] and ans["check"]["web"] is True
        cite = next(iter(ans["citations"].values()))
        assert cite["source_type"] == "web" and cite["link"] == "https://a.example/int4"
        assert "type=web" in st["web_prompt"]


def test_studio_outputs(tmp_path):
    client, c = make_client(tmp_path)
    st: dict = {}
    fakes.CHAT_HANDLERS["nbweb"] = nbweb_llm(st)
    with client:
        setup(client, c)
        nb = client.post("/api/notebooks", json={"title": "studio"}).json()["id"]
        assert client.post(f"/api/notebooks/{nb}/outputs", json={"kind": "summary"}).status_code == 400  # no sources yet
        for f in ("synthetic_quantization.pdf", "synthetic_kvcache.pdf"):
            client.post("/api/documents", files={"file": (f, (CORPUS / f).read_bytes(), "application/pdf")}, data={"collection": nb})
        wait_until(lambda: all(s["status"].startswith("ready") for s in client.get(f"/api/notebooks/{nb}/sources").json()))
        assert client.post(f"/api/notebooks/{nb}/outputs", json={"kind": "podcast"}).status_code == 422

        brief = client.post(f"/api/notebooks/{nb}/outputs", json={"kind": "briefing"}).json()
        assert brief["status"] == "running" and brief["title"] == "Briefing doc"
        done = wait_until(lambda: (o := client.get(f"/api/notebooks/{nb}/outputs/{brief['id']}").json())["status"] != "running" and o)
        assert done["status"] == "completed", done
        assert done["check"]["unknown_citations"] == ["E99"] and "123.4" in done["check"]["numbers_not_in_cited_evidence"]
        assert done["check"]["uncited_statements"] == 1
        cite = next(iter(done["citations"].values()))
        assert cite["link"].startswith("/documents/")
        # passages come from both sources, not just one search
        assert len(set(re.findall(r'doc="([^"]+)"', st["studio_prompt"]))) == 2

        mm = client.post(f"/api/notebooks/{nb}/outputs", json={"kind": "mindmap"}).json()
        mm = wait_until(lambda: (o := client.get(f"/api/notebooks/{nb}/outputs/{mm['id']}").json())["status"] != "running" and o)
        assert mm["status"] == "completed" and mm["data"]["central"] == "LLM serving"
        assert mm["data"]["branches"][0]["children"][0]["label"] == "INT4" and mm["check"]["uncited_statements"] == 1

        assert len(client.get(f"/api/notebooks/{nb}/outputs").json()) == 2
        assert client.delete(f"/api/notebooks/{nb}/outputs/{mm['id']}").status_code == 204
        assert [o["id"] for o in client.get(f"/api/notebooks/{nb}/outputs").json()] == [brief["id"]]
