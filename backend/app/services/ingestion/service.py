"""Document ingestion: validation → Docling parse → selective OCR → structure-aware chunking → indexing.

* Idempotent: documents are keyed by SHA-256; re-uploading a ready document returns it unchanged.
* Only pages whose text layer is missing/garbled are rendered and sent to the OCR route.
* OCR results are cached by (page image hash, OCR provider identity).
* Bounded concurrency for ingestion jobs and for OCR pages.
* Interrupted jobs are resumed at startup; failed jobs can be retried.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.core.text import match_key
from app.observability.logging import get_logger, log_event
from app.providers.base import Capability, ImageInput, NoProviderAvailableError, ProviderError, Task
from app.providers.router import ProviderRouter
from app.repositories.db import Database, utcnow
from app.services.chunking.chunker import chunk_elements
from app.services.parsing.base import ParsedDocument, ParsedElement
from app.services.parsing.docling_parser import DoclingParser
from app.services.parsing.pymupdf_parser import PyMuPDFParser
from app.services.retrieval.bm25 import bump_chunks_version
from app.services.retrieval.index_manager import IndexManager

log = get_logger("omni.ingestion")

OCR_POLICY_KEY = "ocr_policy"
DEFAULT_OCR_POLICY = {"mode": "auto", "language": None, "dpi": None, "min_chars": None, "max_garbled_ratio": 0.3}


class IngestionError(Exception):
    pass


class UploadRejected(ValueError):
    pass


class IngestionService:
    def __init__(self, settings: Settings, db: Database, router: ProviderRouter, index: IndexManager, parser: Any = None):
        self.settings = settings
        self.db = db
        self.router = router
        self.index = index
        self._parser = parser
        self.upload_dir = settings.data_dir / "uploads"
        self.page_cache = settings.data_dir / "cache" / "pages"
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._ocr_sem = asyncio.Semaphore(max(1, settings.ocr_concurrency))
        self._parse_sem = asyncio.Semaphore(1)  # Docling models are large; parse one document at a time

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        for _ in range(max(1, self.settings.ingestion_concurrency)):
            self._workers.append(asyncio.create_task(self._worker()))
        for r in self.db.all("SELECT id FROM ingestion_jobs WHERE status IN ('queued','running')"):
            self.db.execute("UPDATE ingestion_jobs SET status='queued', stage='resumed', updated_at=? WHERE id=?", (utcnow(), r["id"]))
            self._queue.put_nowait(r["id"])

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def wait_idle(self, timeout: float = 600) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._queue.empty() and not self.db.one("SELECT 1 FROM ingestion_jobs WHERE status IN ('queued','running')"):
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("ingestion did not finish")

    def parser(self):
        if self._parser is not None:
            return self._parser
        choice = self.settings.parser
        if choice in ("auto", "docling") and DoclingParser.available():
            self._parser = DoclingParser()
        elif choice == "docling":
            raise IngestionError("Docling is not installed but OMNI_PARSER=docling")
        else:
            self._parser = PyMuPDFParser()
        return self._parser

    # ---------------------------------------------------------------- submission
    def validate_pdf(self, data: bytes, filename: str) -> int:
        import pymupdf

        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise UploadRejected(f"File exceeds {self.settings.max_upload_mb} MB limit")
        if not data.startswith(b"%PDF-"):
            raise UploadRejected("Not a PDF file (missing %PDF- header)")
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise UploadRejected("Encrypted PDFs are not supported")
                n = doc.page_count
        except UploadRejected:
            raise
        except Exception as exc:
            raise UploadRejected(f"Unreadable PDF: {type(exc).__name__}") from exc
        if n == 0:
            raise UploadRejected("PDF has no pages")
        if n > self.settings.max_pages_per_document:
            raise UploadRejected(f"PDF has {n} pages; limit is {self.settings.max_pages_per_document}")
        return n

    def submit(self, data: bytes, filename: str, collection: str = "default", source_url: str | None = None) -> dict[str, Any]:
        safe_name = Path(filename or "document.pdf").name[:200] or "document.pdf"
        collection = collection or "default"
        sha = hashlib.sha256(data).hexdigest()
        # The same file may live in several notebooks; it is a duplicate only within one collection.
        existing = self.db.one("SELECT * FROM documents WHERE sha256 IN (?, ?) AND collection=?", (sha, f"{sha}:{collection}", collection))
        if self.db.one("SELECT 1 FROM documents WHERE sha256=?", (sha,)) and not existing:
            sha = f"{sha}:{collection}"
        if existing:
            if existing["status"] == "failed":
                return self.retry(existing["id"])
            return {"document": existing, "duplicate": True, "job_id": None}
        pages = self.validate_pdf(data, safe_name)
        doc_id = uuid.uuid4().hex
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        # Write via a temp file in the same directory, then atomically move into place.
        with tempfile.NamedTemporaryFile(dir=self.upload_dir, delete=False, suffix=".part") as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.upload_dir / f"{doc_id}.pdf")
        now = utcnow()
        self.db.execute(
            "INSERT INTO documents(id, filename, sha256, size_bytes, collection, page_count, status, source_url, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (doc_id, safe_name, sha, len(data), collection, pages, "queued", source_url, now, now),
        )
        job_id = self._new_job(doc_id)
        return {"document": self.db.one("SELECT * FROM documents WHERE id=?", (doc_id,)), "duplicate": False, "job_id": job_id}

    def _new_job(self, doc_id: str) -> str:
        job_id = uuid.uuid4().hex[:16]
        now = utcnow()
        self.db.execute("INSERT INTO ingestion_jobs(id, document_id, status, stage, created_at, updated_at) VALUES (?,?,?,?,?,?)", (job_id, doc_id, "queued", "queued", now, now))
        self._queue.put_nowait(job_id)
        return job_id

    def retry(self, doc_id: str) -> dict[str, Any]:
        doc = self.db.one("SELECT * FROM documents WHERE id=?", (doc_id,))
        if not doc:
            raise KeyError(doc_id)
        if self.db.one("SELECT 1 FROM ingestion_jobs WHERE document_id=? AND status IN ('queued','running')", (doc_id,)):
            raise ValueError("Document is already being processed")
        self.db.execute("UPDATE documents SET status='queued', error=NULL, updated_at=? WHERE id=?", (utcnow(), doc_id))
        return {"document": self.db.one("SELECT * FROM documents WHERE id=?", (doc_id,)), "duplicate": False, "job_id": self._new_job(doc_id)}

    def delete(self, doc_id: str) -> None:
        doc = self.db.one("SELECT * FROM documents WHERE id=?", (doc_id,))
        if not doc:
            raise KeyError(doc_id)
        if self.db.one("SELECT 1 FROM ingestion_jobs WHERE document_id=? AND status='running'", (doc_id,)):
            raise ValueError("Document is being processed; retry deletion when the job finishes")
        chunk_ids = [r["id"] for r in self.db.all("SELECT id FROM chunks WHERE document_id=?", (doc_id,))]
        self.index.delete_document(doc_id, chunk_ids)
        self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        bump_chunks_version(self.db)
        (self.upload_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
        shutil.rmtree(self.page_cache / doc_id, ignore_errors=True)

    def pdf_path(self, doc_id: str) -> Path:
        return self.upload_dir / f"{doc_id}.pdf"

    # ---------------------------------------------------------------- processing
    def _job(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = utcnow()
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE ingestion_jobs SET {sets} WHERE id=?", (*fields.values(), job_id))

    def _doc(self, doc_id: str, **fields: Any) -> None:
        fields["updated_at"] = utcnow()
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE documents SET {sets} WHERE id=?", (*fields.values(), doc_id))

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self.process(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ingestion worker error")
            finally:
                self._queue.task_done()

    async def process(self, job_id: str) -> None:
        job = self.db.one("SELECT * FROM ingestion_jobs WHERE id=?", (job_id,))
        if not job or job["status"] not in ("queued", "running"):
            return
        doc_id = job["document_id"]
        doc = self.db.one("SELECT * FROM documents WHERE id=?", (doc_id,))
        if not doc:
            self._job(job_id, status="failed", error="document deleted")
            return
        self._job(job_id, status="running", stage="parsing", progress=0.05, attempts=job["attempts"] + 1)
        self._doc(doc_id, status="processing", error=None)
        path = str(self.pdf_path(doc_id))
        t0 = time.perf_counter()
        try:
            parsed = await self._parse(path)
            self._job(job_id, stage="ocr", progress=0.35)
            ocr_report = await self._run_ocr(doc_id, path, parsed, job_id)
            self._job(job_id, stage="chunking", progress=0.7)
            chunks = chunk_elements(doc_id, parsed.elements, max_chars=self.settings.chunk_max_chars, overlap=self.settings.chunk_overlap_chars)
            self._store(doc_id, parsed, chunks)
            self._job(job_id, stage="indexing", progress=0.8)
            index_info: dict[str, Any]
            try:
                index_info = await self.index.index_document(doc_id)
            except (ProviderError, NoProviderAvailableError) as exc:
                index_info = {"indexed": False, "reason": f"embedding failed: {exc}"}
            meta = {
                **parsed.metadata,
                "ocr": ocr_report,
                "index": index_info,
                "parse_seconds": round(time.perf_counter() - t0, 2),
                "element_counts": _counts(parsed.elements),
                "chunk_count": len(chunks),
            }
            status = "ready" if index_info.get("indexed") else "ready_lexical_only"
            if ocr_report.get("failed_pages"):
                status = "ready_with_errors"
            self._doc(doc_id, status=status, title=parsed.title, parser=parsed.parser, page_count=parsed.page_count, metadata_json=json.dumps(meta, default=str))
            self._job(job_id, status="succeeded", stage="done", progress=1.0)
            log_event(log, "document ingested", document_id=doc_id, chunks=len(chunks), status=status, seconds=meta["parse_seconds"])
        except Exception as exc:
            log.exception("ingestion failed")
            msg = f"{type(exc).__name__}: {exc}"[:1000]
            self._doc(doc_id, status="failed", error=msg)
            self._job(job_id, status="failed", error=msg)

    async def _parse(self, path: str) -> ParsedDocument:
        async with self._parse_sem:
            parser = self.parser()
            try:
                return await asyncio.to_thread(parser.parse, path)
            except Exception as exc:
                if isinstance(parser, PyMuPDFParser):
                    raise
                log_event(log, "docling failed; falling back to PyMuPDF", error=f"{type(exc).__name__}: {exc}"[:300])
                parsed = await asyncio.to_thread(PyMuPDFParser().parse, path)
                parsed.metadata["parser_fallback_reason"] = f"docling: {type(exc).__name__}"
                return parsed

    # ---------------------------------------------------------------- OCR
    def ocr_policy(self) -> dict[str, Any]:
        return {**DEFAULT_OCR_POLICY, **(self.db.kv_get(OCR_POLICY_KEY, {}) or {})}

    def pages_needing_ocr(self, parsed: ParsedDocument) -> list[int]:
        pol = self.ocr_policy()
        if pol["mode"] == "off":
            return []
        if pol["mode"] == "force":
            return sorted(parsed.pages)
        min_chars = pol["min_chars"] if pol.get("min_chars") is not None else self.settings.ocr_text_threshold_chars
        need = []
        for no, info in parsed.pages.items():
            if info.text_layer_chars < min_chars and info.parsed_chars < min_chars:
                need.append(no)
            elif info.garbled_ratio > pol["max_garbled_ratio"]:
                need.append(no)
        return sorted(need)

    def render_page(self, doc_id: str, page_no: int, dpi: int) -> bytes:
        import pymupdf

        cache = self.page_cache / doc_id / f"p{page_no}_{dpi}.png"
        if cache.exists():
            return cache.read_bytes()
        with pymupdf.open(self.pdf_path(doc_id)) as pdf:
            if not 1 <= page_no <= pdf.page_count:
                raise ValueError(f"Page {page_no} out of range")
            png = pdf[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(png)
        return png

    async def _run_ocr(self, doc_id: str, path: str, parsed: ParsedDocument, job_id: str) -> dict[str, Any]:
        pages = self.pages_needing_ocr(parsed)
        for no, info in parsed.pages.items():
            method = "text_layer" if no not in pages else "pending_ocr"
            self.db.execute(
                """INSERT INTO pages(document_id, page_no, width, height, text_chars, extraction_method) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(document_id, page_no) DO UPDATE SET width=excluded.width, height=excluded.height,
                   text_chars=excluded.text_chars, extraction_method=excluded.extraction_method, ocr_status=NULL, ocr_error=NULL""",
                (doc_id, no, info.width, info.height, info.text_layer_chars, method),
            )
        report: dict[str, Any] = {"pages_needing_ocr": pages, "ocr_pages": [], "failed_pages": [], "skipped_reason": None}
        if not pages:
            return report
        if not self.router.is_configured(Task.OCR):
            report["skipped_reason"] = "no OCR provider configured"
            for no in pages:
                self.db.execute("UPDATE pages SET ocr_status='skipped', ocr_error=? WHERE document_id=? AND page_no=?", ("no OCR provider configured", doc_id, no))
            report["failed_pages"] = pages
            return report
        pol = self.ocr_policy()
        dpi = pol.get("dpi") or self.settings.ocr_render_dpi
        primary = self.router.primary(Task.OCR)
        identity = f"{primary.adapter}:{primary.model_for(Capability.OCR)}" if primary else "unknown"

        async def one(no: int) -> None:
            async with self._ocr_sem:
                try:
                    png = await asyncio.to_thread(self.render_page, doc_id, no, dpi)
                    key = hashlib.sha256(png).hexdigest() + "|" + identity
                    cached = self.db.one("SELECT * FROM ocr_cache WHERE cache_key=?", (key,))
                    t0 = time.perf_counter()
                    if cached:
                        text, engine, latency = cached["text"], cached["provider_fingerprint"], 0.0
                    else:
                        resp = await self.router.ocr(ImageInput(data=png, mime_type="image/png"), pol.get("language"))
                        text, engine = resp.text, f"{resp.method}:{resp.engine}"
                        latency = (time.perf_counter() - t0) * 1000
                        self.db.execute(
                            "INSERT OR REPLACE INTO ocr_cache(cache_key, provider_fingerprint, text, lines_json, created_at) VALUES (?,?,?,?,?)",
                            (key, engine, text, json.dumps([l.model_dump() for l in resp.lines]), utcnow()),
                        )
                    self._merge_ocr(parsed, no, text, engine, dpi)
                    self.db.execute(
                        "UPDATE pages SET ocr_status='done', extraction_method=?, ocr_provider=?, ocr_latency_ms=? WHERE document_id=? AND page_no=?",
                        (f"ocr:{engine}", engine, latency, doc_id, no),
                    )
                    report["ocr_pages"].append({"page": no, "engine": engine, "chars": len(text), "latency_ms": round(latency, 1), "cached": bool(cached)})
                except (ProviderError, NoProviderAvailableError, ValueError) as exc:
                    report["failed_pages"].append(no)
                    self.db.execute("UPDATE pages SET ocr_status='failed', ocr_error=? WHERE document_id=? AND page_no=?", (str(exc)[:500], doc_id, no))

        done = 0
        for coro in asyncio.as_completed([one(n) for n in pages]):
            await coro
            done += 1
            self._job(job_id, progress=0.35 + 0.3 * done / len(pages))
        report["ocr_pages"].sort(key=lambda r: r["page"])
        return report

    @staticmethod
    def _merge_ocr(parsed: ParsedDocument, page_no: int, text: str, engine: str, dpi: int) -> None:
        """Add OCR text for a page, dropping paragraphs already present in the parsed text (dedup)."""
        existing = match_key(" ".join(e.text for e in parsed.elements if e.page_no == page_no))
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        new_paras = [p for p in paras if match_key(p) not in existing]
        if not new_paras:
            return
        el = ParsedElement(
            element_id=f"#/ocr/page{page_no}",
            type="ocr_text",
            text="\n\n".join(new_paras),
            page_no=page_no,
            extraction_method=f"ocr:{engine}",
        )
        # insert after the last element of the same or an earlier page to keep reading order
        idx = 0
        for i, e in enumerate(parsed.elements):
            if e.page_no <= page_no:
                idx = i + 1
        parsed.elements.insert(idx, el)
        parsed.pages[page_no].parsed_chars += len(el.text)

    def _store(self, doc_id: str, parsed: ParsedDocument, chunks: list) -> None:
        old_ids = [r["id"] for r in self.db.all("SELECT id FROM chunks WHERE document_id=?", (doc_id,))]
        if old_ids:
            self.index.delete_document(doc_id, old_ids)
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
            conn.executemany(
                """INSERT INTO chunks(id, document_id, chunk_index, page_no, page_end, section_path, element_type, extraction_method,
                   text, bbox_json, source_element_id, chunk_version, content_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (c.id, c.document_id, c.chunk_index, c.page_no, c.page_end, c.section_path, c.element_type, c.extraction_method, c.text,
                     json.dumps(c.bbox) if c.bbox else None, c.source_element_id, c.chunk_version, c.content_hash)
                    for c in chunks
                ],
            )
            for no in parsed.pages:
                # Page text (parsed elements + merged OCR text) is what citation excerpts are validated against.
                page_text = "\n\n".join(e.text for e in parsed.elements if e.page_no == no)
                conn.execute("UPDATE pages SET text=? WHERE document_id=? AND page_no=?", (page_text, doc_id, no))
        bump_chunks_version(self.db)


def _counts(elements: list[ParsedElement]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in elements:
        out[e.type] = out.get(e.type, 0) + 1
    return out
