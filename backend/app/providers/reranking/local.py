"""CPU-local reranking adapters."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, ClassVar

from pydantic import Field
from rank_bm25 import BM25Okapi

from app.core.text import tokenize
from app.providers.base import (
    Capability,
    HealthResult,
    ProviderConfigBase,
    ProviderError,
    ProviderErrorKind,
    RerankingProvider,
    RerankRequest,
    RerankResponse,
    RerankResult,
)


class LexicalRerankConfig(ProviderConfigBase):
    coverage_weight: float = Field(default=0.5, ge=0, le=1, description="Weight of query-term coverage vs BM25 score")


class LexicalRerankProvider(RerankingProvider):
    adapter = "local_lexical_rerank"
    display_name = "Local lexical reranker (BM25 + term coverage)"
    protocol = "local"
    capabilities = frozenset({Capability.RERANKING})
    description = "Re-scores candidates by BM25 over the candidate set plus query-term coverage. No model required."
    limitations = ["Lexical only; does not understand paraphrase. A cross-encoder reranker is more accurate."]
    config_model = LexicalRerankConfig
    requires_network = False
    config: LexicalRerankConfig

    def model_for(self, capability: Capability) -> str | None:
        return "bm25-coverage"

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        if not request.candidates:
            return RerankResponse(results=[], model="bm25-coverage")
        q = set(tokenize(request.query))
        docs = [tokenize(c.text) for c in request.candidates]
        bm = BM25Okapi([d or ["_"] for d in docs]).get_scores(list(q)) if q else [0.0] * len(docs)
        top = max(bm) or 1.0
        w = self.config.coverage_weight
        scores = []
        for c, d, b in zip(request.candidates, docs, bm):
            cov = len(q & set(d)) / len(q) if q else 0.0
            scores.append((c.id, (1 - w) * (b / top) + w * cov))
        scores.sort(key=lambda s: s[1], reverse=True)
        n = request.top_n or len(scores)
        return RerankResponse(results=[RerankResult(id=i, score=s, rank=r) for r, (i, s) in enumerate(scores[:n], 1)], model="bm25-coverage")

    async def health_check(self) -> HealthResult:
        return HealthResult(ok=True, latency_ms=0.0, detail="local", checked=["rerank"])


class FastEmbedRerankConfig(ProviderConfigBase):
    model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2", description="fastembed TextCrossEncoder model")
    cache_dir: str | None = None
    threads: int | None = Field(default=None, ge=1, le=64)


class FastEmbedRerankProvider(RerankingProvider):
    adapter = "fastembed_rerank"
    display_name = "FastEmbed cross-encoder (local ONNX on CPU)"
    protocol = "local"
    capabilities = frozenset({Capability.RERANKING})
    description = "Local cross-encoder reranking via fastembed TextCrossEncoder."
    limitations = ["Requires the optional 'fastembed' package.", "Cost grows linearly with candidates on CPU."]
    config_model = FastEmbedRerankConfig
    requires_network = False
    config: FastEmbedRerankConfig

    _models: ClassVar[dict[tuple, Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def model_for(self, capability: Capability) -> str | None:
        return self.config.model

    def _model(self):
        key = (self.config.model, self.config.cache_dir, self.config.threads)
        with self._lock:
            if key not in self._models:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder
                except ImportError as exc:
                    raise ProviderError(ProviderErrorKind.CONFIG, "fastembed is not installed", provider_id=self.provider_id) from exc
                self._models[key] = TextCrossEncoder(model_name=self.config.model, cache_dir=self.config.cache_dir, threads=self.config.threads)
            return self._models[key]

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        if not request.candidates:
            return RerankResponse(results=[], model=self.config.model)

        def run() -> list[float]:
            return [float(s) for s in self._model().rerank(request.query, [c.text for c in request.candidates])]

        scores = await asyncio.to_thread(run)
        ranked = sorted(zip(request.candidates, scores), key=lambda p: p[1], reverse=True)[: request.top_n or len(scores)]
        return RerankResponse(results=[RerankResult(id=c.id, score=s, rank=r) for r, (c, s) in enumerate(ranked, 1)], model=self.config.model)
