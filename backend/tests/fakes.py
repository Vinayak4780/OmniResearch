"""Deterministic mock adapters used to prove provider independence in tests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable

from pydantic import Field

from app.providers.base import (
    Capability,
    ChatModelProvider,
    ChatRequest,
    ChatResponse,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    OCRProvider,
    OCRRequest,
    OCRResponse,
    ProviderConfigBase,
    ProviderError,
    ProviderErrorKind,
    Usage,
    VisionProvider,
    VisionRequest,
    VisionResponse,
    WebExtractionRequest,
    WebPage,
    WebPageExtractionProvider,
    WebSearchProvider,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)


class FakeConfig(ProviderConfigBase):
    label: str = "fake"
    fail_kind: str | None = None
    fail_times: int = 0
    fail_after: int | None = None  # succeed for the first N calls, then fail with fail_kind
    dims: int = 16
    script: str = Field(default="", description="name of a scripted chat behaviour")


CALLS: dict[str, int] = {}
CHAT_HANDLERS: dict[str, Callable[[ChatRequest, str], str]] = {}


def _maybe_fail(provider_id: str, cfg: FakeConfig) -> None:
    CALLS[provider_id] = CALLS.get(provider_id, 0) + 1
    if cfg.fail_kind and cfg.fail_after is not None:
        if CALLS[provider_id] > cfg.fail_after:
            raise ProviderError(ProviderErrorKind(cfg.fail_kind), f"scripted {cfg.fail_kind}", provider_id=provider_id)
        return
    if cfg.fail_kind and (cfg.fail_times == 0 or CALLS[provider_id] <= cfg.fail_times):
        raise ProviderError(ProviderErrorKind(cfg.fail_kind), f"scripted {cfg.fail_kind}", provider_id=provider_id)


class FakeChatProvider(ChatModelProvider, VisionProvider):
    adapter = "fake_chat"
    display_name = "Fake chat"
    protocol = "local"
    capabilities = frozenset({Capability.CHAT, Capability.VISION})
    config_model = FakeConfig
    requires_network = False
    config: FakeConfig

    def model_for(self, capability: Capability) -> str | None:
        return f"fake-{self.config.label}"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        _maybe_fail(self.provider_id, self.config)
        handler = CHAT_HANDLERS.get(self.config.script)
        text = handler(request, self.config.label) if handler else f"[{self.config.label}] {request.messages[-1].content[:40]}"
        return ChatResponse(text=text, model=request.model or f"fake-{self.config.label}", usage=Usage(input_tokens=10, output_tokens=5))

    async def analyze(self, request: VisionRequest) -> VisionResponse:
        _maybe_fail(self.provider_id, self.config)
        return VisionResponse(text=f"[{self.config.label} vision] image of {len(request.images)} item(s)", model="fake-vl")


class FakeEmbeddingProvider(EmbeddingProvider):
    adapter = "fake_embed"
    display_name = "Fake embeddings"
    protocol = "local"
    capabilities = frozenset({Capability.EMBEDDING})
    config_model = FakeConfig
    requires_network = False
    config: FakeConfig

    def model_for(self, capability: Capability) -> str | None:
        return f"fake-embed-{self.config.label}-{self.config.dims}"

    def embedding_fingerprint(self) -> dict[str, Any]:
        return {"adapter": self.adapter, "label": self.config.label, "dims": self.config.dims}

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        _maybe_fail(self.provider_id, self.config)
        out = []
        for t in request.texts:
            v = [0.0] * self.config.dims
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5((self.config.label + tok).encode()).hexdigest(), 16)
                v[h % self.config.dims] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return EmbeddingResponse(vectors=out, model=self.model_for(Capability.EMBEDDING))


class FakeOCRProvider(OCRProvider):
    adapter = "fake_ocr"
    display_name = "Fake OCR"
    protocol = "local"
    capabilities = frozenset({Capability.OCR})
    config_model = FakeConfig
    requires_network = False
    config: FakeConfig

    async def ocr(self, request: OCRRequest) -> OCRResponse:
        _maybe_fail(self.provider_id, self.config)
        return OCRResponse(text=f"OCR TEXT from {self.config.label}: scanned page mentions FlashAttention speedup of 2.4x", engine=f"fake-{self.config.label}")


class FakeSearchProvider(WebSearchProvider, WebPageExtractionProvider):
    adapter = "fake_web"
    display_name = "Fake web"
    protocol = "local"
    capabilities = frozenset({Capability.WEB_SEARCH, Capability.WEB_EXTRACTION})
    config_model = FakeConfig
    requires_network = False
    config: FakeConfig
    PAGES: dict[str, dict[str, str]] = {}

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        _maybe_fail(self.provider_id, self.config)
        results = [
            WebSearchResult(url=url, title=p["title"], snippet=p["text"][:120], published_at=p.get("published_at"), source_engine="fake")
            for url, p in self.PAGES.items()
        ]
        return WebSearchResponse(results=results[: request.max_results], engine="fake")

    async def extract(self, request: WebExtractionRequest) -> WebPage:
        p = self.PAGES.get(request.url)
        if not p:
            raise ProviderError(ProviderErrorKind.BAD_REQUEST, "unknown page", provider_id=self.provider_id)
        return WebPage(url=request.url, final_url=request.url, title=p["title"], text=p["text"], fetched_at="2026-10-23T00:00:00Z", published_at=p.get("published_at"), extractor="fake")


FAKE_ADAPTERS = [FakeChatProvider, FakeEmbeddingProvider, FakeOCRProvider, FakeSearchProvider]


def json_handler(payload_by_marker: dict[str, Any]) -> Callable[[ChatRequest, str], str]:
    """Build a chat handler that returns JSON chosen by a marker string found in the prompt."""

    def handler(req: ChatRequest, label: str) -> str:
        prompt = "\n".join(m.content for m in req.messages)
        for marker, payload in payload_by_marker.items():
            if marker in prompt:
                value = payload(prompt) if callable(payload) else payload
                return value if isinstance(value, str) else json.dumps(value)
        return "{}"

    return handler
