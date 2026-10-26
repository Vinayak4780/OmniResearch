"""Ingestion, OCR routing, provenance, hybrid retrieval, and embedding-index safety."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.config.settings import Settings
from app.providers.base import Task
from app.providers.registry import RouteConfig
from app.services.ingestion.service import IngestionService, UploadRejected
from app.services.parsing.pymupdf_parser import PyMuPDFParser
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.hybrid import Retriever
from app.services.retrieval.index_manager import IndexManager
from app.services.retrieval.vector_store import VectorStore
from tests import fakes

CORPUS = Path(__file__).resolve().parent / "data" / "corpus"


@pytest.fixture(scope="session", autouse=True)
def corpus():
    if not (CORPUS / "synthetic_kvcache.pdf").exists():
        import subprocess, sys

        subprocess.check_call([sys.executable, str(CORPUS.parent / "make_corpus.py"), str(CORPUS)])
    return CORPUS


@pytest.fixture
async def svc(tmp_path, db, registry, router):
    settings = Settings(data_dir=tmp_path, parser="pymupdf", ocr_concurrency=1, ingestion_concurrency=1)
    settings.ensure_dirs()
    registry.create("emb", "fake_embed", {"label": "s1", "dims": 32}, provider_id="emb-a")
    registry.create("ocr", "fake_ocr", {"label": "o"}, provider_id="ocr-fake")
    registry.set_route(RouteConfig(task=Task.TEXT_EMBEDDING, provider_ids=["emb-a"]))
    registry.set_route(RouteConfig(task=Task.OCR, provider_ids=["ocr-fake"]))
    store = VectorStore()
    index = IndexManager(db, router, store, batch_size=4)
    ingestion = IngestionService(settings, db, router, index, parser=PyMuPDFParser())
    ingestion.start()
    bm25 = BM25Index(db)
    retriever = Retriever(db, router, index, bm25)
    yield {"ingestion": ingestion, "index": index, "retriever": retriever, "store": store, "settings": settings}
    await ingestion.stop()


async def _ingest_all(ingestion):
    ids = {}
    for f in sorted(CORPUS.glob("*.pdf")):
        r = ingestion.submit(f.read_bytes(), f.name)
        ids[f.name] = r["document"]["id"]
    await ingestion.wait_idle(120)
    return ids


async def test_ingestion_provenance_and_selective_ocr(svc, db):
    ids = await _ingest_all(svc["ingestion"])
    docs = {d["filename"]: d for d in db.all("SELECT * FROM documents")}
    assert all(d["status"] == "ready" for d in docs.values()), [(d["filename"], d["status"], d["error"]) for d in docs.values()]

    scanned = ids["synthetic_attention_scanned.pdf"]
    pages = db.all("SELECT page_no, extraction_method, ocr_status FROM pages WHERE document_id=? ORDER BY page_no", (scanned,))
    assert pages[0]["extraction_method"] == "text_layer" and pages[0]["ocr_status"] is None  # page 1 not OCR'd
    assert pages[1]["extraction_method"].startswith("ocr:") and pages[1]["ocr_status"] == "done"
    assert fakes.CALLS["ocr-fake"] == 1  # only the scanned page went to OCR

    ocr_chunks = db.all("SELECT * FROM chunks WHERE document_id=? AND extraction_method LIKE 'ocr:%'", (scanned,))
    assert len(ocr_chunks) == 1 and ocr_chunks[0]["page_no"] == 2 and "2.4x" in ocr_chunks[0]["text"]

    # Every chunk carries complete provenance
    for c in db.all("SELECT * FROM chunks"):
        assert c["document_id"] and c["page_no"] >= 1 and c["element_type"] and c["extraction_method"] and c["chunk_version"]
    kv = ids["synthetic_kvcache.pdf"]
    assert db.one("SELECT COUNT(*) n FROM chunks WHERE document_id=? AND page_no=2 AND element_type='caption' AND text LIKE 'Figure 1%'", (kv,))["n"] == 1


async def test_idempotent_upload_and_validation(svc, db):
    data = (CORPUS / "synthetic_quantization.pdf").read_bytes()
    a = svc["ingestion"].submit(data, "q.pdf")
    await svc["ingestion"].wait_idle(60)
    b = svc["ingestion"].submit(data, "q-again.pdf")
    assert b["duplicate"] and b["document"]["id"] == a["document"]["id"]
    assert db.one("SELECT COUNT(*) n FROM documents")["n"] == 1
    with pytest.raises(UploadRejected):
        svc["ingestion"].submit(b"not a pdf at all", "x.pdf")
    with pytest.raises(UploadRejected):
        svc["ingestion"].submit(b"%PDF-1.4 garbage", "y.pdf")


async def test_ocr_cache_reused(svc, db):
    data = (CORPUS / "synthetic_attention_scanned.pdf").read_bytes()
    r = svc["ingestion"].submit(data, "s.pdf")
    await svc["ingestion"].wait_idle(60)
    svc["ingestion"].retry(r["document"]["id"])
    await svc["ingestion"].wait_idle(60)
    assert fakes.CALLS["ocr-fake"] == 1  # second run served from cache
    meta = json.loads(db.one("SELECT metadata_json FROM documents")["metadata_json"])
    assert meta["ocr"]["ocr_pages"][0]["cached"] is True


async def test_ocr_failure_recorded_and_fallback(svc, db, registry):
    registry.update("ocr-fake", config={"label": "o", "fail_kind": "unavailable", "max_retries": 0})
    data = (CORPUS / "synthetic_attention_scanned.pdf").read_bytes()
    r = svc["ingestion"].submit(data, "s.pdf")
    await svc["ingestion"].wait_idle(60)
    doc = db.one("SELECT * FROM documents WHERE id=?", (r["document"]["id"],))
    assert doc["status"] == "ready_with_errors"
    page = db.one("SELECT * FROM pages WHERE document_id=? AND page_no=2", (doc["id"],))
    assert page["ocr_status"] == "failed" and "unavailable" in page["ocr_error"]
    # configured fallback OCR provider rescues the page on retry
    registry.create("ocr2", "fake_ocr", {"label": "local"}, provider_id="ocr-local")
    registry.set_route(RouteConfig(task=Task.OCR, provider_ids=["ocr-fake", "ocr-local"]))
    svc["ingestion"].retry(doc["id"])
    await svc["ingestion"].wait_idle(60)
    page = db.one("SELECT * FROM pages WHERE document_id=? AND page_no=2", (doc["id"],))
    assert page["ocr_status"] == "done" and "local" in page["ocr_provider"]


async def test_hybrid_retrieval_and_filters(svc):
    ids = await _ingest_all(svc["ingestion"])
    r = svc["retriever"]
    res = await r.search("paged KV cache memory waste fragmentation", k=5, mode="hybrid")
    assert res.chunks and res.chunks[0].document_id == ids["synthetic_kvcache.pdf"]
    assert {"dense", "bm25"} & set(res.chunks[0].scores)
    bm = await r.search("WikiText-2 perplexity INT4", k=3, mode="bm25")
    assert bm.chunks[0].document_id == ids["synthetic_quantization.pdf"]
    filtered = await r.search("throughput", k=10, document_ids=[ids["synthetic_quantization.pdf"]])
    assert filtered.chunks and all(c.document_id == ids["synthetic_quantization.pdf"] for c in filtered.chunks)


async def test_reranking_through_route(svc, registry):
    await _ingest_all(svc["ingestion"])
    registry.create("rr", "local_lexical_rerank", {}, provider_id="rr-local")
    registry.set_route(RouteConfig(task=Task.RERANKING, provider_ids=["rr-local"]))
    res = await svc["retriever"].search("continuous batching throughput", k=5)
    assert res.reranked and "rerank" in res.chunks[0].scores


async def test_document_deletion_removes_vectors(svc, db):
    ids = await _ingest_all(svc["ingestion"])
    active = svc["index"].active()
    before = svc["store"].count(active.collection_name)
    doc = ids["synthetic_kvcache.pdf"]
    n = db.one("SELECT COUNT(*) n FROM chunks WHERE document_id=?", (doc,))["n"]
    svc["ingestion"].delete(doc)
    assert svc["store"].count(active.collection_name) == before - n
    assert db.one("SELECT COUNT(*) n FROM chunks WHERE document_id=?", (doc,))["n"] == 0
    res = await svc["retriever"].search("paged KV cache", k=10)
    assert all(c.document_id != doc for c in res.chunks)


async def test_embedding_change_never_mixes_spaces(svc, registry, db):
    await _ingest_all(svc["ingestion"])
    index = svc["index"]
    first = index.active()
    assert first.spec["dimensions"] == 32 and first.spec["provider_id"] == "emb-a"

    # Switch the route to a different embedding model (different dims and space).
    registry.create("emb-b", "fake_embed", {"label": "s2", "dims": 48}, provider_id="emb-b")
    registry.set_route(RouteConfig(task=Task.TEXT_EMBEDDING, provider_ids=["emb-b"]))
    st = index.status()
    assert st["reindex_recommended"] and not st["route_matches_active"]
    # New uploads still go to the active index *with its pinned provider* (no mixing).
    fakes.CALLS.clear()
    svc["ingestion"].submit((CORPUS / "synthetic_quantization.pdf").read_bytes() + b"\n%x", "q2.pdf")
    await svc["ingestion"].wait_idle(60)
    assert fakes.CALLS.get("emb-a") and not fakes.CALLS.get("emb-b")

    # Safe migration: build, validate, switch; previous kept for rollback.
    index.start_reindex()
    await index.wait_idle()
    new = index.active()
    assert new.id != first.id and new.spec["dimensions"] == 48 and new.spec["provider_id"] == "emb-b"
    assert index.previous().id == first.id
    assert svc["store"].count(new.collection_name) == db.one("SELECT COUNT(*) n FROM chunks")["n"]
    res = await svc["retriever"].search("paged attention block table", k=3, mode="dense")
    assert res.chunks and not res.warnings

    rolled = index.rollback()
    assert rolled.id == first.id
    index.start_reindex()
    await index.wait_idle()
    index.confirm_migration()
    assert index.previous() is None
    assert db.one("SELECT status FROM vector_indexes WHERE id=?", (first.id,))["status"] == "retired"


async def test_endpoint_change_same_model_requires_confirmation(svc, registry):
    await _ingest_all(svc["ingestion"])
    index = svc["index"]
    # Same adapter, same model label, but the provider configuration changes its identity (simulated endpoint change).
    registry.update("emb-a", config={"label": "s1", "dims": 32, "compatibility_key": None, "max_retries": 1})
    assert index.compatibility()["ok"]  # unrelated settings do not affect the fingerprint
    registry.update("emb-a", config={"label": "s1-new-endpoint", "dims": 32})
    compat = index.compatibility()
    assert not compat["ok"] and compat["reason"] == "provider_config_changed"
    res = await svc["retriever"].search("paged KV cache", k=3, mode="hybrid")
    assert any("not usable" in w for w in res.warnings)  # dense skipped, BM25 still answers
    assert res.chunks
    r = await svc["ingestion"].index.index_document("nonexistent")
    assert r == {"indexed": False, "reason": "provider_config_changed"}
    index.confirm_compatibility()
    assert index.compatibility()["ok"]
    assert index.active().spec["compatibility_confirmations"]


async def test_failed_reindex_does_not_switch(svc, registry, db):
    await _ingest_all(svc["ingestion"])
    index = svc["index"]
    first = index.active()
    # The dimension probe succeeds, every embedding batch afterwards fails.
    registry.create("emb-bad", "fake_embed", {"label": "bad", "dims": 16, "fail_kind": "unavailable", "fail_after": 1, "max_retries": 0}, provider_id="emb-bad")
    index.start_reindex("emb-bad")
    await index.wait_idle()
    assert index.active().id == first.id
    bad = [i for i in index.list() if i.spec.get("provider_id") == "emb-bad"]
    assert len(bad) == 1 and bad[0].status == "failed" and bad[0].failed_batches > 0
    assert "failed" in bad[0].error
    res = await svc["retriever"].search("paged KV cache", k=3, mode="dense")
    assert res.chunks and not res.warnings  # still served by the original index


@pytest.mark.ml
@pytest.mark.skipif(os.environ.get("OMNI_ML_TESTS") != "1", reason="set OMNI_ML_TESTS=1 to run Docling + PaddleOCR")
async def test_real_docling_and_paddleocr(tmp_path, db, registry, router):
    from app.services.parsing.docling_parser import DoclingParser

    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    registry.create("paddle", "paddleocr_local", {}, provider_id="paddle")
    registry.create("hash", "local_hashing", {}, provider_id="hash")
    registry.set_route(RouteConfig(task=Task.OCR, provider_ids=["paddle"]))
    registry.set_route(RouteConfig(task=Task.TEXT_EMBEDDING, provider_ids=["hash"]))
    index = IndexManager(db, router, VectorStore())
    ing = IngestionService(settings, db, router, index, parser=DoclingParser())
    ing.start()
    r = ing.submit((CORPUS / "synthetic_attention_scanned.pdf").read_bytes(), "scan.pdf")
    k = ing.submit((CORPUS / "synthetic_kvcache.pdf").read_bytes(), "kv.pdf")
    await ing.wait_idle(900)
    await ing.stop()
    doc = db.one("SELECT * FROM documents WHERE id=?", (r["document"]["id"],))
    assert doc["parser"] == "docling" and doc["status"] == "ready", doc["error"]
    ocr = db.one("SELECT * FROM chunks WHERE document_id=? AND extraction_method LIKE 'ocr:%'", (doc["id"],))
    assert ocr and "2.4x" in ocr["text"] and ocr["page_no"] == 2
    tables = db.all("SELECT * FROM chunks WHERE document_id=? AND element_type='table'", (k["document"]["id"],))
    assert tables and "23.0" in tables[0]["text"] and tables[0]["text"].startswith("Table 2")
