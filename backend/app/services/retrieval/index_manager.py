"""Vector-index lifecycle with embedding-compatibility guarantees.

Invariants
 * Every vector collection records the full fingerprint of the embedding space that produced it
   (provider identity, endpoint, model, dimensions, normalization, distance, chunking version,
   preprocessing). A collection only ever contains vectors from that space.
 * The active index is always written and queried with *its own pinned provider*. Changing the
   text-embedding route never silently changes what the active index is fed with.
 * If the pinned provider's configuration changes (e.g. a new endpoint with the same model name),
   the index becomes ``incompatible`` until the user confirms compatibility or reindexes.
 * Reindexing builds a new collection in bounded batches, records progress/errors, validates, and
   only then switches. The previous index is kept for rollback until the user confirms.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any

from app.observability.logging import get_logger, log_event
from app.providers.base import NoProviderAvailableError, ProviderError, ProviderErrorKind, Task, fingerprint_hash
from app.providers.router import ProviderRouter
from app.repositories.db import Database, utcnow
from app.services.chunking.chunker import CHUNK_VERSION
from app.services.retrieval.vector_store import VectorStore

log = get_logger("omni.index")
PREPROCESSING_VERSION = "title-section-prefix-v1"
ACTIVE_KEY = "active_vector_index"
LAST_REINDEX_ERROR_KEY = "last_reindex_error"
PREVIOUS_KEY = "previous_vector_index"


class IndexIncompatibleError(Exception):
    pass


@dataclass
class IndexInfo:
    id: str
    collection_name: str
    fingerprint: str
    spec: dict[str, Any]
    status: str
    progress_done: int
    progress_total: int
    failed_batches: int
    error: str | None
    created_at: str
    activated_at: str | None

    @classmethod
    def from_row(cls, r: dict[str, Any]) -> "IndexInfo":
        return cls(
            id=r["id"],
            collection_name=r["collection_name"],
            fingerprint=r["fingerprint"],
            spec=json.loads(r["fingerprint_json"]),
            status=r["status"],
            progress_done=r["progress_done"],
            progress_total=r["progress_total"],
            failed_batches=r["failed_batches"],
            error=r["error"],
            created_at=r["created_at"],
            activated_at=r["activated_at"],
        )


def embedding_input(text: str, title: str | None, section: str) -> str:
    ctx = " > ".join(x for x in [title or "", section] if x)
    return f"{ctx}\n{text}" if ctx else text


class IndexManager:
    def __init__(self, db: Database, router: ProviderRouter, store: VectorStore, *, batch_size: int = 64, distance: str = "cosine"):
        self.db = db
        self.router = router
        self.store = store
        self.batch_size = batch_size
        self.distance = distance
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------- spec / fingerprint
    async def spec_for_provider(self, provider_id: str, *, probe: bool = True) -> dict[str, Any]:
        fp = self.router.embedding_fingerprint(provider_id)
        if fp is None:
            raise ProviderError(ProviderErrorKind.NOT_CONFIGURED, f"Embedding provider {provider_id} unavailable", provider_id=provider_id)
        spec: dict[str, Any] = {
            "provider_id": provider_id,
            "provider_fingerprint": fp,
            "distance": self.distance,
            "chunking_version": CHUNK_VERSION,
            "preprocessing": PREPROCESSING_VERSION,
        }
        if probe:
            resp = await self.router.embed(["dimension probe"], input_type="document", provider_id=provider_id)
            spec["dimensions"] = resp.dims
            spec["model"] = resp.model
        return spec

    @staticmethod
    def _space_key(spec: dict[str, Any]) -> str:
        """Hash of everything that determines vector compatibility (not the provider row id)."""
        return fingerprint_hash({k: spec.get(k) for k in ("provider_fingerprint", "dimensions", "distance", "chunking_version", "preprocessing")})

    # ---------------------------------------------------------------- lookup
    def get(self, index_id: str | None) -> IndexInfo | None:
        if not index_id:
            return None
        r = self.db.one("SELECT * FROM vector_indexes WHERE id=?", (index_id,))
        return IndexInfo.from_row(r) if r else None

    def active(self) -> IndexInfo | None:
        return self.get(self.db.kv_get(ACTIVE_KEY))

    def previous(self) -> IndexInfo | None:
        return self.get(self.db.kv_get(PREVIOUS_KEY))

    def building(self) -> list[IndexInfo]:
        return [IndexInfo.from_row(r) for r in self.db.all("SELECT * FROM vector_indexes WHERE status IN ('building','validating') ORDER BY created_at")]

    def list(self) -> list[IndexInfo]:
        return [IndexInfo.from_row(r) for r in self.db.all("SELECT * FROM vector_indexes ORDER BY created_at DESC")]

    def compatibility(self, index: IndexInfo | None = None) -> dict[str, Any]:
        """Is the given (default: active) index usable with its pinned provider's *current* config?"""
        index = index or self.active()
        if index is None:
            return {"ok": False, "reason": "no_active_index"}
        pid = index.spec["provider_id"]
        if not self.router.registry.is_enabled(pid):
            return {"ok": False, "reason": "pinned_provider_unavailable", "provider_id": pid}
        current = self.router.embedding_fingerprint(pid)
        if current is None:
            return {"ok": False, "reason": "pinned_provider_unavailable", "provider_id": pid}
        if fingerprint_hash(current) != fingerprint_hash(index.spec["provider_fingerprint"]):
            return {"ok": False, "reason": "provider_config_changed", "provider_id": pid, "indexed_with": index.spec["provider_fingerprint"], "current": current}
        if index.spec.get("chunking_version") != CHUNK_VERSION:
            return {"ok": False, "reason": "chunking_version_changed"}
        return {"ok": True, "provider_id": pid}

    def status(self) -> dict[str, Any]:
        active = self.active()
        route = self.router.route(Task.TEXT_EMBEDDING)
        route_primary = next((p for p in route.provider_ids if self.router.registry.is_enabled(p)), None)
        total_chunks = self.db.one("SELECT COUNT(*) AS n FROM chunks")["n"]
        out: dict[str, Any] = {
            "active": active.__dict__ if active else None,
            "previous": self.previous().__dict__ if self.previous() else None,
            "building": [b.__dict__ for b in self.building()],
            "last_reindex_error": self.db.kv_get(LAST_REINDEX_ERROR_KEY),
            "recent_failed": [IndexInfo.from_row(r).__dict__ for r in self.db.all("SELECT * FROM vector_indexes WHERE status='failed' ORDER BY updated_at DESC LIMIT 5")],
            "route_primary_provider": route_primary,
            "compatibility": self.compatibility(active) if active else {"ok": False, "reason": "no_active_index"},
            "total_chunks": total_chunks,
        }
        if active:
            out["indexed_chunks"] = self.db.one("SELECT COUNT(*) AS n FROM indexed_chunks WHERE index_id=?", (active.id,))["n"]
            out["missing_vectors"] = total_chunks - out["indexed_chunks"]
            out["vector_count"] = self.store.count(active.collection_name)
            out["route_matches_active"] = route_primary == active.spec["provider_id"] or (
                route_primary is not None
                and self.router.embedding_fingerprint(route_primary) is not None
                and fingerprint_hash(self.router.embedding_fingerprint(route_primary)) == fingerprint_hash(active.spec["provider_fingerprint"])
            )
            out["reindex_recommended"] = not out["route_matches_active"] or not out["compatibility"]["ok"]
        return out

    # ---------------------------------------------------------------- creation
    async def _create(self, provider_id: str, status: str) -> IndexInfo:
        spec = await self.spec_for_provider(provider_id)
        space = self._space_key(spec)
        iid = uuid.uuid4().hex[:12]
        name = f"omni_{space}_{iid}"
        self.store.ensure_collection(name, spec["dimensions"], self.distance)
        now = utcnow()
        self.db.execute(
            "INSERT INTO vector_indexes(id, collection_name, fingerprint, fingerprint_json, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (iid, name, space, json.dumps(spec), status, now, now),
        )
        log_event(log, "vector index created", index_id=iid, provider_id=provider_id, dims=spec["dimensions"], status=status)
        return self.get(iid)  # type: ignore[return-value]

    async def ensure_active(self) -> IndexInfo | None:
        """Create the first index from the embedding route if none exists. Never replaces an existing one."""
        async with self._lock:
            active = self.active()
            if active is not None:
                return active
            route = self.router.route(Task.TEXT_EMBEDDING)
            pid = next((p for p in route.provider_ids if self.router.registry.is_enabled(p)), None)
            if pid is None:
                return None
            idx = await self._create(pid, "active")
            self.db.execute("UPDATE vector_indexes SET activated_at=? WHERE id=?", (utcnow(), idx.id))
            self.db.kv_set(ACTIVE_KEY, idx.id)
            return self.get(idx.id)

    # ---------------------------------------------------------------- writing
    async def _embed_into(self, index: IndexInfo, rows: list[dict[str, Any]]) -> int:
        """Embed chunk rows with the index's pinned provider and upsert. Returns count written."""
        if not rows:
            return 0
        titles = {r["document_id"]: r.get("doc_title") for r in rows}
        texts = [embedding_input(r["text"], titles.get(r["document_id"]), r["section_path"]) for r in rows]
        resp = await self.router.embed(texts, input_type="document", provider_id=index.spec["provider_id"])
        dims = index.spec["dimensions"]
        if len(resp.vectors) != len(rows) or any(len(v) != dims for v in resp.vectors):
            raise IndexIncompatibleError(f"Embedding dimension mismatch: index expects {dims}, provider returned {resp.dims}")
        self.store.upsert(
            index.collection_name,
            [(r["id"], v, {"document_id": r["document_id"], "page_no": r["page_no"], "element_type": r["element_type"]}) for r, v in zip(rows, resp.vectors)],
        )
        self.db.executemany("INSERT OR IGNORE INTO indexed_chunks(index_id, chunk_id) VALUES (?,?)", [(index.id, r["id"]) for r in rows])
        return len(rows)

    def _chunk_rows(self, where: str = "", params: tuple = ()) -> list[dict[str, Any]]:
        return self.db.all(
            f"""SELECT c.id, c.document_id, c.page_no, c.element_type, c.section_path, c.text, d.title AS doc_title
                FROM chunks c JOIN documents d ON d.id=c.document_id {where} ORDER BY c.document_id, c.chunk_index""",
            params,
        )

    async def index_document(self, document_id: str) -> dict[str, Any]:
        """Index a document's chunks into the active index (and any index being built)."""
        active = await self.ensure_active()
        if active is None:
            return {"indexed": False, "reason": "no_embedding_provider"}
        compat = self.compatibility(active)
        if not compat["ok"]:
            return {"indexed": False, "reason": compat["reason"]}
        rows = self._chunk_rows("WHERE c.document_id=?", (document_id,))
        written = 0
        for i in range(0, len(rows), self.batch_size):
            written += await self._embed_into(active, rows[i : i + self.batch_size])
        for b in self.building():
            try:
                for i in range(0, len(rows), self.batch_size):
                    await self._embed_into(b, rows[i : i + self.batch_size])
            except (ProviderError, IndexIncompatibleError) as exc:
                log_event(log, "failed to mirror document into building index", index_id=b.id, error=str(exc))
        return {"indexed": True, "chunks": written, "index_id": active.id}

    async def backfill_missing(self) -> dict[str, Any]:
        active = self.active()
        if not active or not self.compatibility(active)["ok"]:
            return {"backfilled": 0, "reason": "active index unavailable or incompatible"}
        rows = self._chunk_rows("WHERE c.id NOT IN (SELECT chunk_id FROM indexed_chunks WHERE index_id=?)", (active.id,))
        n = 0
        for i in range(0, len(rows), self.batch_size):
            n += await self._embed_into(active, rows[i : i + self.batch_size])
        return {"backfilled": n}

    def delete_document(self, document_id: str, chunk_ids: list[str]) -> None:
        for idx in self.list():
            if idx.status in ("active", "building", "validating", "ready", "previous"):
                self.store.delete_document(idx.collection_name, document_id)
        if chunk_ids:
            self.db.executemany("DELETE FROM indexed_chunks WHERE chunk_id=?", [(c,) for c in chunk_ids])

    # ---------------------------------------------------------------- querying
    async def search(self, query: str, limit: int, document_ids: list[str] | None = None) -> tuple[list[tuple[str, float]], str | None]:
        """Dense search on the active index. Returns (hits, warning)."""
        active = self.active()
        if active is None:
            return [], "No vector index: dense retrieval unavailable (configure a text-embedding provider)."
        compat = self.compatibility(active)
        if not compat["ok"]:
            return [], f"Active vector index is not usable ({compat['reason']}); dense retrieval skipped. Reindex or confirm compatibility."
        try:
            resp = await self.router.embed([query], input_type="query", provider_id=active.spec["provider_id"])
        except (ProviderError, NoProviderAvailableError) as exc:
            return [], f"Query embedding failed: {exc}"
        vec = resp.vectors[0]
        if len(vec) != active.spec["dimensions"]:
            return [], f"Query embedding has {len(vec)} dims; index expects {active.spec['dimensions']}"
        return self.store.search(active.collection_name, vec, limit, document_ids), None

    # ---------------------------------------------------------------- migration
    def start_reindex(self, provider_id: str | None = None) -> str:
        if self.building():
            raise ValueError("A reindex is already in progress")
        route = self.router.route(Task.TEXT_EMBEDDING)
        pid = provider_id or next((p for p in route.provider_ids if self.router.registry.is_enabled(p)), None)
        if pid is None:
            raise ValueError("No enabled text-embedding provider is configured")
        job_id = uuid.uuid4().hex[:8]
        self.db.kv_set(LAST_REINDEX_ERROR_KEY, None)
        self._tasks[job_id] = asyncio.create_task(self._reindex(pid))
        return job_id

    async def resume_interrupted(self) -> None:
        for b in self.building():
            log_event(log, "resuming interrupted reindex", index_id=b.id)
            self._tasks[b.id] = asyncio.create_task(self._reindex(b.spec["provider_id"], resume=b))

    async def _reindex(self, provider_id: str, resume: IndexInfo | None = None) -> None:
        idx: IndexInfo | None = resume
        try:
            if idx is None:
                idx = await self._create(provider_id, "building")
            rows = self._chunk_rows("WHERE c.id NOT IN (SELECT chunk_id FROM indexed_chunks WHERE index_id=?)", (idx.id,))
            done = self.db.one("SELECT COUNT(*) AS n FROM indexed_chunks WHERE index_id=?", (idx.id,))["n"]
            total = done + len(rows)
            self.db.execute("UPDATE vector_indexes SET progress_total=?, progress_done=?, updated_at=? WHERE id=?", (total, done, utcnow(), idx.id))
            failed = 0
            for i in range(0, len(rows), self.batch_size):
                batch = rows[i : i + self.batch_size]
                for attempt in range(3):
                    try:
                        done += await self._embed_into(idx, batch)
                        break
                    except (ProviderError, NoProviderAvailableError) as exc:
                        if attempt == 2:
                            failed += 1
                            self.db.execute("UPDATE vector_indexes SET failed_batches=?, error=? WHERE id=?", (failed, str(exc)[:500], idx.id))
                        else:
                            await asyncio.sleep(1.0 * (attempt + 1))
                self.db.execute("UPDATE vector_indexes SET progress_done=?, updated_at=? WHERE id=?", (done, utcnow(), idx.id))
            self.db.execute("UPDATE vector_indexes SET status='validating', updated_at=? WHERE id=?", (utcnow(), idx.id))
            ok, detail = await self._validate(idx)
            if not ok:
                self.db.execute("UPDATE vector_indexes SET status='failed', error=?, updated_at=? WHERE id=?", (detail, utcnow(), idx.id))
                return
            self._switch(idx.id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("reindex failed")
            # Failures before the new collection exists (e.g. the embedding probe) must still be visible.
            self.db.kv_set(LAST_REINDEX_ERROR_KEY, {"at": utcnow(), "provider_id": provider_id, "error": f"{type(exc).__name__}: {exc}"[:500]})
            if idx is not None:
                self.db.execute("UPDATE vector_indexes SET status='failed', error=?, updated_at=? WHERE id=?", (f"{type(exc).__name__}: {exc}"[:500], utcnow(), idx.id))

    async def _validate(self, idx: IndexInfo) -> tuple[bool, str]:
        idx = self.get(idx.id)  # type: ignore[assignment]
        total = self.db.one("SELECT COUNT(*) AS n FROM chunks")["n"]
        indexed = self.db.one("SELECT COUNT(*) AS n FROM indexed_chunks WHERE index_id=?", (idx.id,))["n"]
        stored = self.store.count(idx.collection_name)
        if idx.failed_batches:
            return False, f"{idx.failed_batches} batch(es) failed; {indexed}/{total} chunks embedded"
        if indexed != total or stored < total:
            return False, f"Incomplete index: {indexed}/{total} recorded, {stored} vectors stored"
        # Self-retrieval sanity check: a chunk's own text should retrieve it.
        sample = self.db.all("SELECT c.id, c.text FROM chunks c ORDER BY RANDOM() LIMIT 5")
        hits = 0
        for s in sample:
            resp = await self.router.embed([s["text"]], input_type="document", provider_id=idx.spec["provider_id"])
            ids = [cid for cid, _ in self.store.search(idx.collection_name, resp.vectors[0], 5)]
            hits += s["id"] in ids
        if sample and hits < max(1, int(0.6 * len(sample))):
            return False, f"Validation failed: self-retrieval {hits}/{len(sample)}"
        return True, f"{indexed} chunks, self-retrieval {hits}/{len(sample)}"

    def _switch(self, new_id: str) -> None:
        old = self.active()
        prev = self.previous()
        if prev and prev.id != (old.id if old else None):
            self._retire(prev)
        if old:
            self.db.execute("UPDATE vector_indexes SET status='previous', updated_at=? WHERE id=?", (utcnow(), old.id))
            self.db.kv_set(PREVIOUS_KEY, old.id)
        self.db.execute("UPDATE vector_indexes SET status='active', activated_at=?, updated_at=? WHERE id=?", (utcnow(), utcnow(), new_id))
        self.db.kv_set(ACTIVE_KEY, new_id)
        log_event(log, "vector index switched", new=new_id, previous=old.id if old else None)

    def rollback(self) -> IndexInfo:
        prev, cur = self.previous(), self.active()
        if prev is None:
            raise ValueError("No previous index to roll back to")
        if cur:
            self.db.execute("UPDATE vector_indexes SET status='rolled_back', updated_at=? WHERE id=?", (utcnow(), cur.id))
        self.db.execute("UPDATE vector_indexes SET status='active', activated_at=?, updated_at=? WHERE id=?", (utcnow(), utcnow(), prev.id))
        self.db.kv_set(ACTIVE_KEY, prev.id)
        self.db.kv_set(PREVIOUS_KEY, None)
        return self.active()  # type: ignore[return-value]

    def confirm_migration(self) -> None:
        prev = self.previous()
        if prev is None:
            raise ValueError("No migration awaiting confirmation")
        self._retire(prev)
        self.db.kv_set(PREVIOUS_KEY, None)

    def _retire(self, idx: IndexInfo) -> None:
        self.store.drop(idx.collection_name)
        self.db.execute("UPDATE vector_indexes SET status='retired', updated_at=? WHERE id=?", (utcnow(), idx.id))
        self.db.execute("DELETE FROM indexed_chunks WHERE index_id=?", (idx.id,))

    def confirm_compatibility(self) -> IndexInfo:
        """User asserts the pinned provider's *changed* configuration still produces the same vector space."""
        active = self.active()
        if active is None:
            raise ValueError("No active index")
        current = self.router.embedding_fingerprint(active.spec["provider_id"])
        if current is None:
            raise ValueError("Pinned provider unavailable")
        spec = dict(active.spec)
        history = spec.setdefault("compatibility_confirmations", [])
        history.append({"at": utcnow(), "from": spec["provider_fingerprint"], "to": current})
        spec["provider_fingerprint"] = current
        self.db.execute("UPDATE vector_indexes SET fingerprint_json=?, updated_at=? WHERE id=?", (json.dumps(spec), utcnow(), active.id))
        return self.active()  # type: ignore[return-value]

    async def wait_idle(self) -> None:
        tasks = [t for t in self._tasks.values() if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
