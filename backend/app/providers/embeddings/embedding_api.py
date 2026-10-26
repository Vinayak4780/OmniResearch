"""Mode B: native (non-OpenAI) embedding endpoints: Hugging Face TEI, Cohere v2, Ollama native."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import Field, field_validator

from app.providers.base import (
    Capability,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResult,
    ProviderConfigBase,
    Usage,
)
from app.providers.common import batched, invalid, l2_normalize, validate_headers
from app.providers.http import join_url


class EmbeddingAPIConfig(ProviderConfigBase):
    dialect: Literal["tei", "cohere", "ollama"] = "tei"
    base_url: str = Field(..., description="e.g. http://localhost:8080 (TEI), https://api.cohere.com, http://localhost:11434")
    model: str = Field(default="", description="Model ID (TEI serves a single model; used only for identity)")
    normalize_embeddings: bool = True
    batch_size: int = Field(default=32, ge=1, le=1024)
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_headers")
    @classmethod
    def _headers(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_headers(v)


class EmbeddingAPIProvider(EmbeddingProvider):
    adapter = "embedding_api"
    display_name = "Embedding API (TEI / Cohere / Ollama native)"
    protocol = "native"
    capabilities = frozenset({Capability.EMBEDDING})
    description = "Native embedding endpoints whose wire formats differ from OpenAI's /embeddings."
    limitations = ["TEI identity depends on the served model; set 'model' so index fingerprints are meaningful."]
    config_model = EmbeddingAPIConfig
    secret_fields = ["api_key"]
    config: EmbeddingAPIConfig

    def model_for(self, capability: Capability) -> str | None:
        return self.config.model or None

    def embedding_fingerprint(self) -> dict[str, Any]:
        c = self.config
        return {"adapter": self.adapter, "dialect": c.dialect, "endpoint": c.base_url.rstrip("/"), "model": c.model, "normalize": c.normalize_embeddings}

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self.config.extra_headers}
        if key := self.secrets.get("api_key"):
            h["Authorization"] = f"Bearer {key}"
        return h

    async def _batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        c = self.config
        if c.dialect == "tei":
            path, body = "embed", {"inputs": texts, "truncate": True}
        elif c.dialect == "cohere":
            path = "v2/embed"
            body = {"model": c.model, "texts": texts, "input_type": "search_query" if input_type == "query" else "search_document", "embedding_types": ["float"]}
        else:
            path, body = "api/embed", {"model": c.model, "input": texts}
        data = await self.http.request_json(self.provider_id, "POST", join_url(c.base_url, path), headers=self._headers(), json=body, timeout=c.timeout_s)
        if c.dialect == "tei":
            vecs = data
        elif c.dialect == "cohere":
            vecs = (data.get("embeddings") or {}).get("float") if isinstance(data, dict) else None
        else:
            vecs = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vecs, list) or len(vecs) != len(texts) or not all(isinstance(v, list) for v in vecs):
            raise invalid(self.provider_id, f"{c.dialect} embeddings missing or count mismatch")
        return vecs

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        out: list[list[float]] = []
        for batch in batched(request.texts, self.config.batch_size):
            for v in await self._batch(batch, request.input_type):
                out.append(l2_normalize(v) if self.config.normalize_embeddings else v)
        return EmbeddingResponse(vectors=out, model=self.config.model or self.config.dialect, usage=Usage())

    async def health_check(self) -> HealthResult:
        t0 = time.perf_counter()
        r = await self.embed(EmbeddingRequest(texts=["health check"], input_type="query"))
        return HealthResult(ok=True, latency_ms=(time.perf_counter() - t0) * 1000, detail=f"dims={r.dims}", checked=["embedding"])
