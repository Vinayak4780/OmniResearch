"""Mode B: native reranking endpoints with differing request/response schemas.

All dialects are normalized to ``RerankResponse(results=[RerankResult(id, score, rank)])``.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import Field, field_validator

from app.providers.base import (
    Capability,
    HealthResult,
    ProviderConfigBase,
    RerankCandidate,
    RerankingProvider,
    RerankRequest,
    RerankResponse,
    RerankResult,
)
from app.providers.common import invalid, validate_headers
from app.providers.http import join_url

Dialect = Literal["cohere", "jina", "voyage", "nvidia_nim", "tei"]

DEFAULT_PATHS: dict[str, str] = {
    "cohere": "v2/rerank",
    "jina": "v1/rerank",
    "voyage": "v1/rerank",
    "nvidia_nim": "v1/ranking",
    "tei": "rerank",
}


class RerankAPIConfig(ProviderConfigBase):
    dialect: Dialect = Field(default="cohere", description="Wire format: cohere | jina | voyage | nvidia_nim | tei")
    base_url: str = Field(..., description="API root, e.g. https://api.cohere.com or http://localhost:8080 (TEI)")
    path: str = Field(default="", description="Override endpoint path (defaults per dialect)")
    model: str = Field(default="", description="Reranker model ID (not used by TEI)")
    max_candidates: int = Field(default=100, ge=1, le=1000)
    truncate: bool = True
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_headers")
    @classmethod
    def _headers(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_headers(v)


class RerankAPIProvider(RerankingProvider):
    adapter = "rerank_api"
    display_name = "Reranking API (Cohere / Jina / Voyage / NVIDIA NIM / TEI)"
    protocol = "native"
    capabilities = frozenset({Capability.RERANKING})
    description = "Dedicated reranking endpoints. Each dialect maps its own wire format to the normalized rerank contract."
    limitations = ["Candidates beyond max_candidates are dropped (lowest fused rank first).", "NIM returns logits; scores are not probabilities."]
    config_model = RerankAPIConfig
    secret_fields = ["api_key"]
    config: RerankAPIConfig

    def model_for(self, capability: Capability) -> str | None:
        return self.config.model or self.config.dialect

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json", **self.config.extra_headers}
        if key := self.secrets.get("api_key"):
            h["Authorization"] = f"Bearer {key}"
        return h

    def _body(self, query: str, docs: list[str], top_n: int) -> dict[str, Any]:
        d, m = self.config.dialect, self.config.model
        if d == "nvidia_nim":
            return {"model": m, "query": {"text": query}, "passages": [{"text": t} for t in docs], "truncate": "END" if self.config.truncate else "NONE"}
        if d == "tei":
            return {"query": query, "texts": docs, "truncate": self.config.truncate, "raw_scores": False}
        body: dict[str, Any] = {"model": m, "query": query, "documents": docs, "top_n": top_n}
        if d == "voyage":
            body["top_k"] = body.pop("top_n")
            body["truncation"] = self.config.truncate
        return body

    def _parse(self, data: Any) -> list[tuple[int, float]]:
        d = self.config.dialect
        if d == "nvidia_nim":
            items = data.get("rankings") if isinstance(data, dict) else None
            key = "logit"
        elif d == "tei":
            items, key = data, "score"
        elif d == "voyage":
            items, key = (data.get("data") if isinstance(data, dict) else None), "relevance_score"
        else:
            items, key = (data.get("results") if isinstance(data, dict) else None), "relevance_score"
        if not isinstance(items, list):
            raise invalid(self.provider_id, f"{d} rerank results missing")
        out = []
        for it in items:
            if not isinstance(it, dict) or "index" not in it or key not in it:
                raise invalid(self.provider_id, f"{d} rerank item lacks index/{key}")
            out.append((int(it["index"]), float(it[key])))
        return out

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        cands: list[RerankCandidate] = request.candidates[: self.config.max_candidates]
        if not cands:
            return RerankResponse(results=[], model=self.config.model)
        top_n = min(request.top_n or len(cands), len(cands))
        path = self.config.path or DEFAULT_PATHS[self.config.dialect]
        data = await self.http.request_json(
            self.provider_id,
            "POST",
            join_url(self.config.base_url, path),
            headers=self._headers(),
            json=self._body(request.query, [c.text for c in cands], top_n),
            timeout=self.config.timeout_s,
        )
        pairs = [(i, s) for i, s in self._parse(data) if 0 <= i < len(cands)]
        pairs.sort(key=lambda p: p[1], reverse=True)
        results = [RerankResult(id=cands[i].id, score=s, rank=r) for r, (i, s) in enumerate(pairs[:top_n], start=1)]
        return RerankResponse(results=results, model=self.config.model or self.config.dialect)

    async def health_check(self) -> HealthResult:
        t0 = time.perf_counter()
        r = await self.rerank(
            RerankRequest(query="gpu memory", candidates=[RerankCandidate(id="a", text="GPU memory usage"), RerankCandidate(id="b", text="Cooking pasta")])
        )
        ok = bool(r.results)
        return HealthResult(ok=ok, latency_ms=(time.perf_counter() - t0) * 1000, detail=f"top={r.results[0].id}" if ok else "empty", checked=["rerank"])
