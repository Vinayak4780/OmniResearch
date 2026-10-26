"""Thin Qdrant wrapper. Uses a Qdrant server when OMNI_QDRANT_URL is set, else embedded on-disk mode."""

from __future__ import annotations

import threading
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

_NS = uuid.UUID("6f1d5a52-5d0b-4d7e-9a43-0c2f1f6c7e11")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


class VectorStore:
    def __init__(self, *, url: str = "", api_key: str = "", path: str | None = None, client: QdrantClient | None = None):
        self.remote = bool(url)
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(location=":memory:")
        self._lock = threading.RLock()  # embedded mode is not safe for concurrent writers

    def ensure_collection(self, name: str, dims: int, distance: str = "cosine") -> None:
        with self._lock:
            if self.client.collection_exists(name):
                info = self.client.get_collection(name)
                vec = info.config.params.vectors
                size = vec.size if hasattr(vec, "size") else None
                if size is not None and size != dims:
                    raise ValueError(f"Collection {name} has {size} dims, expected {dims}")
                return
            dist = {"cosine": qm.Distance.COSINE, "dot": qm.Distance.DOT, "euclid": qm.Distance.EUCLID}[distance]
            self.client.create_collection(name, vectors_config=qm.VectorParams(size=dims, distance=dist))
            if self.remote:  # payload indexes are a no-op in embedded mode
                self.client.create_payload_index(name, "document_id", qm.PayloadSchemaType.KEYWORD)

    def upsert(self, name: str, items: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        if not items:
            return
        points = [qm.PointStruct(id=point_id(cid), vector=vec, payload={**payload, "chunk_id": cid}) for cid, vec, payload in items]
        with self._lock:
            self.client.upsert(name, points=points, wait=True)

    def search(self, name: str, vector: list[float], limit: int, document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        flt = None
        if document_ids:
            flt = qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchAny(any=document_ids))])
        with self._lock:
            res = self.client.query_points(name, query=vector, limit=limit, query_filter=flt, with_payload=["chunk_id"])
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def delete_document(self, name: str, document_id: str) -> None:
        with self._lock:
            if self.client.collection_exists(name):
                self.client.delete(name, points_selector=qm.FilterSelector(filter=qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))])), wait=True)

    def delete_points(self, name: str, chunk_ids: list[str]) -> None:
        with self._lock:
            if chunk_ids and self.client.collection_exists(name):
                self.client.delete(name, points_selector=qm.PointIdsList(points=[point_id(c) for c in chunk_ids]), wait=True)

    def count(self, name: str) -> int:
        with self._lock:
            if not self.client.collection_exists(name):
                return 0
            return int(self.client.count(name, exact=True).count)

    def drop(self, name: str) -> None:
        with self._lock:
            if self.client.collection_exists(name):
                self.client.delete_collection(name)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
