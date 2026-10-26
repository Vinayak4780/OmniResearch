"""HTTP API: auth, credential handling, document lifecycle and research endpoints."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.core.container import Container
from app.core.netguard import OutboundPolicy
from app.main import create_app
from app.services.parsing.pymupdf_parser import PyMuPDFParser
from app.services.retrieval.vector_store import VectorStore
from tests import fakes
from tests.test_agent import scripted_llm

CORPUS = Path(__file__).resolve().parent / "data" / "corpus"


def make_client(tmp_path, admin_token: str = "", client=("127.0.0.1", 5000)):
    settings = Settings(data_dir=tmp_path, parser="pymupdf", admin_token=admin_token)
    c = Container(settings, extra_adapters=fakes.FAKE_ADAPTERS, policy=OutboundPolicy(resolve_dns=False), vector_store=VectorStore(), parser=PyMuPDFParser(), checkpoints=False, backoff_base_s=0)
    c.db.kv_set("bootstrapped", True)
    app = create_app(settings, container=c)
    return TestClient(app, client=client), c


def wait_until(fn, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.1)
    raise TimeoutError


def test_provider_crud_never_returns_secrets(tmp_path):
    client, c = make_client(tmp_path)
    with client:
        cat = client.get("/api/providers/catalog").json()
        assert {"openai_compatible", "anthropic", "custom_http", "paddleocr_local", "tavily"} <= {a["adapter"] for a in cat}
        r = client.post("/api/providers", json={"name": "NIM", "adapter": "openai_compatible", "config": {"base_url": "https://integrate.api.nvidia.com/v1", "chat_model": "m"}, "secrets": {"api_key": "nvapi-TOPSECRET-1234"}})
        assert r.status_code == 201, r.text
        pid = r.json()["id"]
        for resp in (r, client.get("/api/providers"), client.get(f"/api/providers/{pid}")):
            assert "nvapi-TOPSECRET-1234" not in resp.text
        assert client.get(f"/api/providers/{pid}").json()["secrets_set"] == {"api_key": True}
        r = client.patch(f"/api/providers/{pid}", json={"config": {"base_url": "https://integrate.api.nvidia.com/v1", "chat_model": "m2"}})
        assert r.json()["config"]["chat_model"] == "m2" and r.json()["secrets_set"]["api_key"] is True
        # route + safe delete
        assert client.put("/api/routes/synthesis", json={"task": "synthesis", "provider_ids": [pid]}).status_code == 200
        routes = {x["task"]: x for x in client.get("/api/routes").json()}
        assert routes["planning"]["inherits_from"] == "synthesis" and routes["planning"]["effective_provider_ids"] == [pid]
        bad = client.put("/api/routes/text_embedding", json={"task": "text_embedding", "provider_ids": [pid]})
        assert bad.status_code == 400  # chat model can't serve embeddings unless configured with an embedding model... capability check
        d = client.delete(f"/api/providers/{pid}")
        assert d.status_code == 409 and d.json()["references"] == ["route:synthesis"]
        assert client.delete(f"/api/providers/{pid}?force=true").status_code == 204
        assert client.post("/api/providers", json={"name": "x", "adapter": "openai_compatible", "config": {"base_url": "ftp://x"}}).status_code == 400


def test_admin_auth(tmp_path):
    client, _ = make_client(tmp_path, client=("203.0.113.9", 5000))
    with client:
        assert client.get("/api/providers").status_code == 403  # remote, no token configured
        assert client.get("/api/health").status_code == 200
    client2, _ = make_client(tmp_path / "b", admin_token="s3cret-token", client=("203.0.113.9", 5000))
    with client2:
        assert client2.get("/api/providers").status_code == 401
        assert client2.get("/api/providers", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client2.get("/api/providers", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200


def test_documents_and_research(tmp_path):
    client, c = make_client(tmp_path)
    fakes.CHAT_HANDLERS["main"] = scripted_llm({})
    with client:
        for pid, adapter, cfg in [("llm", "fake_chat", {"label": "l", "script": "main"}), ("emb", "fake_embed", {"label": "e", "dims": 24}), ("ocr", "fake_ocr", {"label": "o"}), ("rr", "local_lexical_rerank", {})]:
            assert client.post("/api/providers", json={"id": pid, "name": pid, "adapter": adapter, "config": cfg}).status_code == 201
        for task, pid in [("synthesis", "llm"), ("text_embedding", "emb"), ("ocr", "ocr"), ("reranking", "rr")]:
            assert client.put(f"/api/routes/{task}", json={"task": task, "provider_ids": [pid]}).status_code == 200

        pdf = (CORPUS / "synthetic_kvcache.pdf").read_bytes()
        r = client.post("/api/documents", files={"file": ("kv.pdf", pdf, "application/pdf")}, data={"collection": "papers"})
        assert r.status_code == 201, r.text
        doc_id = r.json()["document"]["id"]
        assert client.post("/api/documents", files={"file": ("x.pdf", b"hello", "application/pdf")}).status_code == 400
        wait_until(lambda: client.get(f"/api/documents/{doc_id}").json()["status"] == "ready")
        d = client.get(f"/api/documents/{doc_id}").json()
        assert d["page_count"] == 2 and d["pages"][0]["extraction_method"] == "text_layer"
        assert client.get(f"/api/documents/{doc_id}/pdf").headers["content-type"] == "application/pdf"
        png = client.get(f"/api/documents/{doc_id}/pages/2.png")
        assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
        assert client.get("/api/collections").json() == [{"collection": "papers", "documents": 1}]
        s = client.post("/api/search", json={"query": "paged KV cache waste", "k": 3}).json()
        assert s["results"] and s["results"][0]["document_id"] == doc_id and s["reranked"]
        st = client.get("/api/index/status").json()
        assert st["active"]["spec"]["dimensions"] == 24 and st["missing_vectors"] == 0

        r = client.post("/api/research", json={"question": "How much does continuous batching help throughput?", "web_mode": "never", "collections": ["papers"]})
        assert r.status_code == 201, r.text
        job = r.json()["id"]
        wait_until(lambda: client.get(f"/api/research/{job}").json()["status"] in ("completed", "failed"))
        j = client.get(f"/api/research/{job}").json()
        assert j["status"] == "completed", j["error"]
        assert client.get(f"/api/research/{job}/events").json()
        ev = client.get(f"/api/research/{job}/evidence").json()
        assert ev and all(e["document_id"] == doc_id for e in ev)
        md = client.get(f"/api/research/{job}/report.md")
        assert md.status_code == 200 and md.text.startswith("# ")

        assert client.get("/api/evaluation/runs").status_code == 404  # feature removed
        assert client.get("/api/provider-health").json()["metrics"]  # call metrics are still recorded

        assert client.get("/api/research/does-not-exist").status_code == 404
        assert client.delete(f"/api/documents/{doc_id}").status_code == 204
        assert client.get(f"/api/documents/{doc_id}").status_code == 404
