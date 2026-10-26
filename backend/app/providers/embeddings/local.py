"""CPU-local embedding adapters."""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
from typing import Any, ClassVar

from pydantic import Field

from app.core.text import tokenize
from app.providers.base import (
    Capability,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResult,
    ProviderConfigBase,
    ProviderError,
    ProviderErrorKind,
)


class HashingEmbeddingConfig(ProviderConfigBase):
    dimensions: int = Field(default=512, ge=64, le=4096)
    use_bigrams: bool = True


class HashingEmbeddingProvider(EmbeddingProvider):
    """Feature-hashing bag-of-words vectors. Dependency-free and deterministic, but lexical only."""

    adapter = "local_hashing"
    display_name = "Local hashing embeddings (lexical, dependency-free)"
    protocol = "local"
    capabilities = frozenset({Capability.EMBEDDING})
    description = "Deterministic signed feature hashing of unigrams/bigrams. Works fully offline with no model download."
    limitations = [
        "Captures lexical overlap only — no semantic similarity. Use a neural embedder (fastembed or an API) for real dense retrieval.",
    ]
    config_model = HashingEmbeddingConfig
    requires_network = False
    config: HashingEmbeddingConfig

    def model_for(self, capability: Capability) -> str | None:
        return f"hashing-{self.config.dimensions}{'-bigram' if self.config.use_bigrams else ''}"

    def embedding_fingerprint(self) -> dict[str, Any]:
        return {"adapter": self.adapter, "model": self.model_for(Capability.EMBEDDING), "dimensions": self.config.dimensions, "normalize": True, "version": 1}

    def _vec(self, text: str) -> list[float]:
        dims = self.config.dimensions
        v = [0.0] * dims
        toks = tokenize(text)
        feats = toks + ([f"{a}_{b}" for a, b in zip(toks, toks[1:])] if self.config.use_bigrams else [])
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % dims] += 1.0 if (h >> 63) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(vectors=[self._vec(t) for t in request.texts], model=self.model_for(Capability.EMBEDDING))

    async def health_check(self) -> HealthResult:
        return HealthResult(ok=True, latency_ms=0.0, detail=f"dims={self.config.dimensions}", checked=["embedding"])


class FastEmbedConfig(ProviderConfigBase):
    model: str = Field(default="BAAI/bge-small-en-v1.5", description="Any model supported by fastembed TextEmbedding")
    cache_dir: str | None = None
    threads: int | None = Field(default=None, ge=1, le=64)
    batch_size: int = Field(default=32, ge=1, le=512)
    query_prefix: str = Field(default="", description="Some models expect e.g. 'query: '")
    document_prefix: str = ""


class FastEmbedProvider(EmbeddingProvider):
    adapter = "fastembed"
    display_name = "FastEmbed (local ONNX on CPU)"
    protocol = "local"
    capabilities = frozenset({Capability.EMBEDDING})
    description = "Local CPU embeddings via Qdrant fastembed (ONNX Runtime). Model weights are downloaded once."
    limitations = ["Requires the optional 'fastembed' package.", "First use downloads the model (~50–500 MB)."]
    config_model = FastEmbedConfig
    requires_network = False
    config: FastEmbedConfig

    _models: ClassVar[dict[tuple, Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def model_for(self, capability: Capability) -> str | None:
        return self.config.model

    def embedding_fingerprint(self) -> dict[str, Any]:
        c = self.config
        return {"adapter": self.adapter, "model": c.model, "normalize": True, "query_prefix": c.query_prefix, "document_prefix": c.document_prefix}

    def _model(self):
        key = (self.config.model, self.config.cache_dir, self.config.threads)
        with self._lock:
            if key not in self._models:
                try:
                    from fastembed import TextEmbedding
                except ImportError as exc:
                    raise ProviderError(ProviderErrorKind.CONFIG, "fastembed is not installed (pip install fastembed)", provider_id=self.provider_id) from exc
                self._models[key] = TextEmbedding(model_name=self.config.model, cache_dir=self.config.cache_dir, threads=self.config.threads)
            return self._models[key]

    def _embed_sync(self, texts: list[str], input_type: str) -> list[list[float]]:
        prefix = self.config.query_prefix if input_type == "query" else self.config.document_prefix
        model = self._model()
        out = []
        for v in model.embed([prefix + t for t in texts], batch_size=self.config.batch_size):
            n = float((v**2).sum()) ** 0.5
            out.append((v / n if n else v).tolist())
        return out

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        vecs = await asyncio.to_thread(self._embed_sync, request.texts, request.input_type)
        return EmbeddingResponse(vectors=vecs, model=self.config.model)

    async def health_check(self) -> HealthResult:
        r = await self.embed(EmbeddingRequest(texts=["health check"], input_type="query"))
        return HealthResult(ok=True, detail=f"dims={r.dims}", checked=["embedding"])
