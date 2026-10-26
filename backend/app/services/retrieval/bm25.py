"""In-process BM25 over all chunks, rebuilt lazily when the chunk set changes."""

from __future__ import annotations

import threading

from rank_bm25 import BM25Okapi

from app.core.text import tokenize
from app.repositories.db import Database
from app.services.retrieval.index_manager import embedding_input

VERSION_KEY = "chunks_version"


def bump_chunks_version(db: Database) -> None:
    db.kv_set(VERSION_KEY, (db.kv_get(VERSION_KEY, 0) or 0) + 1)


class BM25Index:
    def __init__(self, db: Database):
        self.db = db
        self._version = -1
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        v = self.db.kv_get(VERSION_KEY, 0)
        if v == self._version and self._bm25 is not None:
            return
        rows = self.db.all(
            "SELECT c.id, c.document_id, c.text, c.section_path, d.title FROM chunks c JOIN documents d ON d.id=c.document_id ORDER BY c.document_id, c.chunk_index"
        )
        self._ids = [r["id"] for r in rows]
        self._docs = [r["document_id"] for r in rows]
        corpus = [tokenize(embedding_input(r["text"], r["title"], r["section_path"])) or ["_"] for r in rows]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._version = v

    def search(self, query: str, limit: int, document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        with self._lock:
            self._ensure()
            if self._bm25 is None:
                return []
            q = tokenize(query)
            if not q:
                return []
            scores = self._bm25.get_scores(q)
            allowed = set(document_ids) if document_ids else None
            ranked = sorted(
                ((cid, float(s)) for cid, doc, s in zip(self._ids, self._docs, scores) if s > 0 and (allowed is None or doc in allowed)),
                key=lambda p: p[1],
                reverse=True,
            )
            return ranked[:limit]
