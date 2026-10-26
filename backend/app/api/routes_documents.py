"""Document library, ingestion jobs, OCR policy, vector index lifecycle, and retrieval."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.api.deps import container, require_admin
from app.core.container import Container
from app.services.ingestion.service import DEFAULT_OCR_POLICY, OCR_POLICY_KEY

router = APIRouter(prefix="/api", tags=["documents"])


def _doc(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
    return row


@router.post("/documents", status_code=201)
async def upload(file: UploadFile = File(...), collection: str = Form("default"), c: Container = Depends(container)):
    limit = c.settings.max_upload_mb * 1024 * 1024
    data = bytearray()
    while chunk := await file.read(1 << 20):
        data += chunk
        if len(data) > limit:
            raise HTTPException(413, f"File exceeds {c.settings.max_upload_mb} MB")
    collection = (collection or "default").strip()[:64] or "default"
    if not c.db.one("SELECT 1 FROM notebooks WHERE id=?", (collection,)):
        c.db.execute("INSERT INTO notebooks(id, title, created_at, updated_at) VALUES (?,?,datetime('now'),datetime('now'))", (collection, collection))
    res = c.ingestion.submit(bytes(data), file.filename or "document.pdf", collection)
    c.notebooks.touch(collection)
    return {"document": _doc(res["document"]), "duplicate": res["duplicate"], "job_id": res["job_id"]}


@router.get("/documents")
def list_documents(q: str | None = None, collection: str | None = None, c: Container = Depends(container)):
    sql = "SELECT * FROM documents WHERE 1=1"
    params: list[Any] = []
    if q:
        sql += " AND (filename LIKE ? OR title LIKE ?)"
        params += [f"%{q}%", f"%{q}%"]
    if collection:
        sql += " AND collection=?"
        params.append(collection)
    rows = c.db.all(sql + " ORDER BY created_at DESC", tuple(params))
    jobs = {r["document_id"]: r for r in c.db.all("SELECT * FROM ingestion_jobs WHERE id IN (SELECT id FROM ingestion_jobs j WHERE created_at = (SELECT MAX(created_at) FROM ingestion_jobs WHERE document_id=j.document_id))")}
    out = []
    for r in rows:
        d = _doc(r)
        d["job"] = jobs.get(r["id"])
        d["chunk_count"] = c.db.one("SELECT COUNT(*) n FROM chunks WHERE document_id=?", (r["id"],))["n"]
        out.append(d)
    return out


@router.get("/collections")
def collections(c: Container = Depends(container)):
    return c.db.all("SELECT collection, COUNT(*) AS documents FROM documents GROUP BY collection ORDER BY collection")


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, c: Container = Depends(container)):
    row = c.db.one("SELECT * FROM documents WHERE id=?", (doc_id,))
    if not row:
        raise KeyError(doc_id)
    d = _doc(row)
    d["pages"] = c.db.all("SELECT page_no, width, height, text_chars, extraction_method, ocr_status, ocr_error, ocr_provider, ocr_latency_ms FROM pages WHERE document_id=? ORDER BY page_no", (doc_id,))
    d["jobs"] = c.db.all("SELECT * FROM ingestion_jobs WHERE document_id=? ORDER BY created_at DESC", (doc_id,))
    d["element_types"] = c.db.all("SELECT element_type, extraction_method, COUNT(*) AS n FROM chunks WHERE document_id=? GROUP BY element_type, extraction_method", (doc_id,))
    return d


@router.get("/documents/{doc_id}/chunks")
def document_chunks(doc_id: str, page: int | None = None, c: Container = Depends(container)):
    if page:
        return c.db.all("SELECT * FROM chunks WHERE document_id=? AND page_no<=? AND page_end>=? ORDER BY chunk_index", (doc_id, page, page))
    return c.db.all("SELECT * FROM chunks WHERE document_id=? ORDER BY chunk_index", (doc_id,))


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, c: Container = Depends(container)):
    c.ingestion.delete(doc_id)


@router.post("/documents/{doc_id}/retry")
async def retry_document(doc_id: str, c: Container = Depends(container)):
    res = c.ingestion.retry(doc_id)
    return {"document": _doc(res["document"]), "job_id": res["job_id"]}


@router.get("/documents/{doc_id}/pdf")
def document_pdf(doc_id: str, c: Container = Depends(container)):
    row = c.db.one("SELECT filename FROM documents WHERE id=?", (doc_id,))
    path = c.ingestion.pdf_path(doc_id)
    if not row or not path.exists():
        raise KeyError(doc_id)
    return FileResponse(path, media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{row["filename"]}"', "X-Content-Type-Options": "nosniff"})


@router.get("/documents/{doc_id}/pages/{page_no}.png")
def page_image(doc_id: str, page_no: int, dpi: int = 110, c: Container = Depends(container)):
    if not c.db.one("SELECT 1 FROM documents WHERE id=?", (doc_id,)):
        raise KeyError(doc_id)
    dpi = max(50, min(dpi, 220))
    return Response(c.ingestion.render_page(doc_id, page_no, dpi), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@router.get("/ingestion/jobs")
def ingestion_jobs(c: Container = Depends(container)):
    return c.db.all("SELECT j.*, d.filename FROM ingestion_jobs j JOIN documents d ON d.id=j.document_id ORDER BY j.created_at DESC LIMIT 100")


class OCRPolicy(BaseModel):
    mode: Literal["auto", "off", "force"] = "auto"
    language: str | None = None
    dpi: int | None = Field(default=None, ge=72, le=400)
    min_chars: int | None = Field(default=None, ge=0, le=5000)
    max_garbled_ratio: float = Field(default=0.3, ge=0, le=1)


@router.get("/settings/ocr")
def get_ocr(c: Container = Depends(container)):
    return {**DEFAULT_OCR_POLICY, **(c.db.kv_get(OCR_POLICY_KEY, {}) or {})}


@router.put("/settings/ocr", dependencies=[Depends(require_admin)])
def put_ocr(body: OCRPolicy, c: Container = Depends(container)):
    c.db.kv_set(OCR_POLICY_KEY, body.model_dump())
    return body.model_dump()


# ---------------------------------------------------------------- vector index
@router.get("/index/status")
def index_status(c: Container = Depends(container)):
    return c.index.status()


class ReindexIn(BaseModel):
    provider_id: str | None = None


@router.post("/index/reindex", dependencies=[Depends(require_admin)])
async def reindex(body: ReindexIn, c: Container = Depends(container)):
    return {"job": c.index.start_reindex(body.provider_id)}


@router.post("/index/rollback", dependencies=[Depends(require_admin)])
def rollback(c: Container = Depends(container)):
    return c.index.rollback().__dict__


@router.post("/index/confirm-migration", dependencies=[Depends(require_admin)])
def confirm_migration(c: Container = Depends(container)):
    c.index.confirm_migration()
    return c.index.status()


@router.post("/index/confirm-compatibility", dependencies=[Depends(require_admin)])
def confirm_compat(c: Container = Depends(container)):
    return c.index.confirm_compatibility().__dict__


@router.post("/index/backfill", dependencies=[Depends(require_admin)])
async def backfill(c: Container = Depends(container)):
    return await c.index.backfill_missing()


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=50)
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    rerank: bool = True
    document_ids: list[str] | None = None
    collections: list[str] | None = None


@router.post("/search")
async def search(body: SearchIn, c: Container = Depends(container)):
    res = await c.retriever.search(body.query, k=body.k, mode=body.mode, rerank=body.rerank, document_ids=body.document_ids, collections=body.collections)
    return {"query": res.query, "mode": res.mode, "reranked": res.reranked, "warnings": res.warnings, "results": [ch.to_dict() for ch in res.chunks]}
