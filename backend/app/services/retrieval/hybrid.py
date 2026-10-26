"""Hybrid retrieval: dense (active vector index) + BM25, fused with Reciprocal Rank Fusion,
then optionally reranked through the RerankingProvider route."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from app.providers.base import NoProviderAvailableError, ProviderError, RerankCandidate, Task
from app.providers.router import ProviderRouter
from app.repositories.db import Database
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.index_manager import IndexManager

RRF_K = 60
Mode = Literal["dense", "bm25", "hybrid"]


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    title: str | None
    page_no: int
    page_end: int
    section_path: str
    element_type: str
    extraction_method: str
    text: str
    bbox: list[float] | None
    source_element_id: str | None
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    final_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RetrievalResult:
    query: str
    mode: str
    chunks: list[RetrievedChunk]
    reranked: bool
    warnings: list[str]


def rrf(rankings: dict[str, list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in rankings.values():
        for r, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r)
    return scores


class Retriever:
    def __init__(self, db: Database, router: ProviderRouter, index: IndexManager, bm25: BM25Index):
        self.db = db
        self.router = router
        self.index = index
        self.bm25 = bm25

    def _load(self, ids: list[str]) -> dict[str, RetrievedChunk]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.db.all(
            f"""SELECT c.*, d.filename, d.title FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id IN ({q})""", tuple(ids)
        )
        return {
            r["id"]: RetrievedChunk(
                chunk_id=r["id"],
                document_id=r["document_id"],
                filename=r["filename"],
                title=r["title"],
                page_no=r["page_no"],
                page_end=r["page_end"],
                section_path=r["section_path"],
                element_type=r["element_type"],
                extraction_method=r["extraction_method"],
                text=r["text"],
                bbox=json.loads(r["bbox_json"]) if r["bbox_json"] else None,
                source_element_id=r["source_element_id"],
            )
            for r in rows
        }

    def document_ids_for(self, collections: list[str] | None, document_ids: list[str] | None) -> list[str] | None:
        if document_ids:
            return document_ids
        if collections:
            q = ",".join("?" * len(collections))
            return [r["id"] for r in self.db.all(f"SELECT id FROM documents WHERE collection IN ({q})", tuple(collections))] or ["__none__"]
        return None

    async def search(
        self,
        query: str,
        *,
        k: int = 8,
        mode: Mode = "hybrid",
        rerank: bool = True,
        document_ids: list[str] | None = None,
        collections: list[str] | None = None,
        candidates: int = 40,
        element_types: list[str] | None = None,
    ) -> RetrievalResult:
        warnings: list[str] = []
        doc_filter = self.document_ids_for(collections, document_ids)
        rankings: dict[str, list[str]] = {}
        raw: dict[str, dict[str, float]] = {}
        if mode in ("dense", "hybrid"):
            hits, warn = await self.index.search(query, candidates, doc_filter)
            if warn:
                warnings.append(warn)
            rankings["dense"] = [c for c, _ in hits]
            raw["dense"] = dict(hits)
        if mode in ("bm25", "hybrid") or (mode == "dense" and not rankings.get("dense")):
            if mode == "dense":
                warnings.append("Dense retrieval returned nothing; fell back to BM25.")
            hits = self.bm25.search(query, candidates, doc_filter)
            rankings["bm25"] = [c for c, _ in hits]
            raw["bm25"] = dict(hits)

        fused = rrf(rankings)
        order = sorted(fused, key=lambda c: fused[c], reverse=True)
        loaded = self._load(order)
        results: list[RetrievedChunk] = []
        seen_hashes: set[str] = set()
        for cid in order:
            ch = loaded.get(cid)
            if ch is None:
                continue
            if element_types and ch.element_type not in element_types:
                continue
            key = ch.text.strip()[:500]
            if key in seen_hashes:  # deduplicate identical text (e.g. repeated boilerplate)
                continue
            seen_hashes.add(key)
            for name, ranked in rankings.items():
                if cid in raw.get(name, {}):
                    ch.scores[name] = raw[name][cid]
                    ch.ranks[name] = ranked.index(cid) + 1
            ch.scores["rrf"] = fused[cid]
            ch.final_score = fused[cid]
            results.append(ch)

        reranked = False
        if rerank and results and self.router.is_configured(Task.RERANKING):
            pool = results[: min(len(results), 30)]
            try:
                rr = await self.router.rerank(query, [RerankCandidate(id=c.chunk_id, text=c.text[:2000]) for c in pool], top_n=len(pool))
                by_id = {c.chunk_id: c for c in pool}
                new: list[RetrievedChunk] = []
                for r in rr.results:
                    c = by_id.get(r.id)
                    if c:
                        c.scores["rerank"] = r.score
                        c.ranks["rerank"] = r.rank
                        c.final_score = r.score
                        new.append(c)
                results = new + [c for c in pool if c.chunk_id not in {x.chunk_id for x in new}] + results[len(pool):]
                reranked = True
            except (ProviderError, NoProviderAvailableError) as exc:
                warnings.append(f"Reranking failed, using fused order: {exc}")
        return RetrievalResult(query=query, mode=mode, chunks=results[:k], reranked=reranked, warnings=warnings)
