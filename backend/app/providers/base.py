"""Capability interfaces and normalized I/O contracts for every AI integration.

Application code (agents, retrieval, ingestion, reports) depends only on the types in this module and
on :class:`app.providers.router.ProviderRouter`. Vendor-specific request/response formats live inside
adapters under ``app/providers/<capability>/``.
"""

from __future__ import annotations

import abc
import hashlib
import json
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- capabilities & tasks


class Capability(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE_EMBEDDING = "image_embedding"
    RERANKING = "reranking"
    OCR = "ocr"
    VISION = "vision"
    WEB_SEARCH = "web_search"
    WEB_EXTRACTION = "web_extraction"


class Task(str, Enum):
    """Units of work that are routed independently. Several tasks may share a capability."""

    PLANNING = "planning"
    DECOMPOSITION = "decomposition"
    REWRITING = "rewriting"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    CHAT = "chat"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_EMBEDDING = "image_embedding"
    RERANKING = "reranking"
    OCR = "ocr"
    VISION = "vision"
    WEB_SEARCH = "web_search"
    WEB_EXTRACTION = "web_extraction"


TASK_CAPABILITY: dict[Task, Capability] = {
    Task.PLANNING: Capability.CHAT,
    Task.DECOMPOSITION: Capability.CHAT,
    Task.REWRITING: Capability.CHAT,
    Task.SYNTHESIS: Capability.CHAT,
    Task.VERIFICATION: Capability.CHAT,
    Task.CHAT: Capability.CHAT,
    Task.TEXT_EMBEDDING: Capability.EMBEDDING,
    Task.IMAGE_EMBEDDING: Capability.IMAGE_EMBEDDING,
    Task.RERANKING: Capability.RERANKING,
    Task.OCR: Capability.OCR,
    Task.VISION: Capability.VISION,
    Task.WEB_SEARCH: Capability.WEB_SEARCH,
    Task.WEB_EXTRACTION: Capability.WEB_EXTRACTION,
}

TASK_DESCRIPTIONS: dict[Task, str] = {
    Task.PLANNING: "Research planning and question analysis",
    Task.DECOMPOSITION: "Breaking the question into sub-queries",
    Task.REWRITING: "Rewriting follow-up search queries",
    Task.SYNTHESIS: "Final report generation",
    Task.VERIFICATION: "Semantic claim verification",
    Task.CHAT: "Answering questions about a notebook's sources",
    Task.TEXT_EMBEDDING: "Text embeddings for dense retrieval",
    Task.IMAGE_EMBEDDING: "Page-image / multimodal embeddings",
    Task.RERANKING: "Reranking retrieved candidates",
    Task.OCR: "OCR for scanned or poorly extracted pages",
    Task.VISION: "Vision-language inspection of figures and pages",
    Task.WEB_SEARCH: "Live web search",
    Task.WEB_EXTRACTION: "Webpage fetching and content extraction",
}


# --------------------------------------------------------------------------- errors


class ProviderErrorKind(str, Enum):
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    BAD_REQUEST = "bad_request"
    INVALID_RESPONSE = "invalid_response"
    UNSUPPORTED = "unsupported"
    NETWORK = "network"
    CONFIG = "config"
    BLOCKED = "blocked"
    CIRCUIT_OPEN = "circuit_open"
    NOT_CONFIGURED = "not_configured"


RETRYABLE_KINDS = {ProviderErrorKind.RATE_LIMIT, ProviderErrorKind.TIMEOUT, ProviderErrorKind.UNAVAILABLE, ProviderErrorKind.NETWORK}


class ProviderError(Exception):
    """Normalized provider failure. Messages must never contain credentials."""

    def __init__(
        self,
        kind: ProviderErrorKind,
        message: str,
        *,
        provider_id: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.provider_id = provider_id
        self.status_code = status_code
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "message": self.message, "provider_id": self.provider_id, "status_code": self.status_code}

    def __str__(self) -> str:  # pragma: no cover - trivial
        prefix = f"[{self.provider_id}] " if self.provider_id else ""
        return f"{prefix}{self.kind.value}: {self.message}"


class NoProviderAvailableError(ProviderError):
    def __init__(self, task: str, attempts: list[dict[str, Any]] | None = None, message: str | None = None):
        super().__init__(
            ProviderErrorKind.NOT_CONFIGURED if not attempts else ProviderErrorKind.UNAVAILABLE,
            message or (f"No provider configured for task '{task}'" if not attempts else f"All providers failed for task '{task}'"),
        )
        self.task = task
        self.attempts = attempts or []

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "task": self.task, "attempts": self.attempts}


# --------------------------------------------------------------------------- normalized contracts


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class ImageInput(BaseModel):
    data: bytes
    mime_type: str = "image/png"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images: list[ImageInput] = Field(default_factory=list)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float | None = 0.2
    max_tokens: int | None = 2048
    json_mode: bool = False
    json_schema: dict[str, Any] | None = None
    stop: list[str] | None = None
    model: str | None = None  # route-level override; adapters fall back to their configured model


class ChatResponse(BaseModel):
    text: str
    model: str | None = None
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)


class EmbeddingRequest(BaseModel):
    texts: list[str]
    input_type: Literal["query", "document"] = "document"


class EmbeddingResponse(BaseModel):
    vectors: list[list[float]]
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)

    @property
    def dims(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


class ImageEmbeddingRequest(BaseModel):
    images: list[ImageInput] = Field(default_factory=list)
    texts: list[str] = Field(default_factory=list)  # text queries embedded into the same space


class RerankCandidate(BaseModel):
    id: str
    text: str


class RerankRequest(BaseModel):
    query: str
    candidates: list[RerankCandidate]
    top_n: int | None = None


class RerankResult(BaseModel):
    id: str
    score: float
    rank: int


class RerankResponse(BaseModel):
    results: list[RerankResult]
    model: str | None = None


class OCRRequest(BaseModel):
    image: ImageInput
    language: str | None = None


class OCRLine(BaseModel):
    text: str
    confidence: float | None = None
    bbox: list[float] | None = None  # [x0, y0, x1, y1] in image pixel coordinates


class OCRResponse(BaseModel):
    text: str
    lines: list[OCRLine] = Field(default_factory=list)
    engine: str
    method: Literal["ocr_engine", "vlm_transcription"] = "ocr_engine"


class VisionRequest(BaseModel):
    prompt: str
    images: list[ImageInput]
    system: str | None = None
    max_tokens: int | None = 1024
    model: str | None = None


class VisionResponse(BaseModel):
    text: str
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5
    recency_days: int | None = None
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)


class WebSearchResult(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    published_at: str | None = None
    score: float | None = None
    source_engine: str | None = None


class WebSearchResponse(BaseModel):
    results: list[WebSearchResult]
    engine: str


class WebExtractionRequest(BaseModel):
    url: str
    max_chars: int = 60_000


class WebPage(BaseModel):
    url: str
    final_url: str
    canonical_url: str | None = None
    title: str = ""
    text: str
    site_name: str | None = None
    author: str | None = None
    published_at: str | None = None
    fetched_at: str
    status_code: int | None = None
    content_type: str | None = None
    truncated: bool = False
    extractor: str


class HealthResult(BaseModel):
    ok: bool
    latency_ms: float | None = None
    detail: str = ""
    checked: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- provider base


class ProviderDescriptor(BaseModel):
    """Static metadata the UI and router use to reason about an adapter type."""

    adapter: str
    display_name: str
    protocol: Literal["openai_compatible", "native", "custom", "local"]
    capabilities: list[Capability]
    modalities: list[str]
    description: str
    limitations: list[str]
    config_schema: dict[str, Any]
    secret_fields: list[str]
    requires_network: bool = True


class ProviderConfigBase(BaseModel):
    """Fields common to every adapter configuration."""

    enabled_capabilities: list[Capability] = Field(
        default_factory=list, description="Subset of the adapter's capabilities to expose. Empty = all."
    )
    timeout_s: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=6)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    price_input_per_1m: float | None = Field(default=None, ge=0, description="USD per 1M input tokens (for cost estimates)")
    price_output_per_1m: float | None = Field(default=None, ge=0, description="USD per 1M output tokens")
    compatibility_key: str | None = Field(
        default=None,
        description="Embeddings only: explicit confirmation that providers sharing this key produce identical vector spaces.",
    )


class BaseProvider(abc.ABC):
    adapter: ClassVar[str]
    display_name: ClassVar[str]
    protocol: ClassVar[str]
    capabilities: ClassVar[frozenset[Capability]]
    modalities: ClassVar[list[str]] = ["text"]
    description: ClassVar[str] = ""
    limitations: ClassVar[list[str]] = []
    config_model: ClassVar[type[ProviderConfigBase]] = ProviderConfigBase
    secret_fields: ClassVar[list[str]] = []
    requires_network: ClassVar[bool] = True

    def __init__(self, provider_id: str, config: ProviderConfigBase, secrets: dict[str, str], http: Any = None):
        self.provider_id = provider_id
        self.config = config
        self.secrets = secrets
        self.http = http

    # ---- metadata
    @classmethod
    def describe(cls) -> ProviderDescriptor:
        return ProviderDescriptor(
            adapter=cls.adapter,
            display_name=cls.display_name,
            protocol=cls.protocol,  # type: ignore[arg-type]
            capabilities=sorted(cls.capabilities, key=lambda c: c.value),
            modalities=cls.modalities,
            description=cls.description,
            limitations=cls.limitations,
            config_schema=cls.config_model.model_json_schema(),
            secret_fields=cls.secret_fields,
            requires_network=cls.requires_network,
        )

    @property
    def active_capabilities(self) -> frozenset[Capability]:
        enabled = set(self.config.enabled_capabilities or self.capabilities)
        return frozenset(enabled & set(self.capabilities))

    def supports(self, capability: Capability) -> bool:
        return capability in self.active_capabilities

    def model_for(self, capability: Capability) -> str | None:
        """The model identity used for a capability (for display, metrics and fingerprints)."""
        return None

    def embedding_fingerprint(self) -> dict[str, Any]:
        """Identity of the embedding output space. Two providers are vector-compatible only if equal."""
        raise ProviderError(ProviderErrorKind.UNSUPPORTED, "Provider does not produce embeddings", provider_id=self.provider_id)

    async def health_check(self) -> HealthResult:
        return HealthResult(ok=True, detail="No remote health check implemented for this adapter")

    async def aclose(self) -> None:
        return None

    def estimate_cost(self, usage: Usage) -> float | None:
        c = self.config
        if c.price_input_per_1m is None and c.price_output_per_1m is None:
            return None
        if usage.input_tokens is None and usage.output_tokens is None:
            return None
        return ((usage.input_tokens or 0) * (c.price_input_per_1m or 0) + (usage.output_tokens or 0) * (c.price_output_per_1m or 0)) / 1e6


def fingerprint_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- capability interfaces


class ChatModelProvider(BaseProvider):
    @abc.abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse: ...


class EmbeddingProvider(BaseProvider):
    @abc.abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


class ImageEmbeddingProvider(BaseProvider):
    @abc.abstractmethod
    async def embed_images(self, request: ImageEmbeddingRequest) -> EmbeddingResponse: ...


class RerankingProvider(BaseProvider):
    @abc.abstractmethod
    async def rerank(self, request: RerankRequest) -> RerankResponse: ...


class OCRProvider(BaseProvider):
    @abc.abstractmethod
    async def ocr(self, request: OCRRequest) -> OCRResponse: ...


class VisionProvider(BaseProvider):
    @abc.abstractmethod
    async def analyze(self, request: VisionRequest) -> VisionResponse: ...


class WebSearchProvider(BaseProvider):
    @abc.abstractmethod
    async def search(self, request: WebSearchRequest) -> WebSearchResponse: ...


class WebPageExtractionProvider(BaseProvider):
    @abc.abstractmethod
    async def extract(self, request: WebExtractionRequest) -> WebPage: ...


CAPABILITY_METHOD: dict[Capability, str] = {
    Capability.CHAT: "chat",
    Capability.EMBEDDING: "embed",
    Capability.IMAGE_EMBEDDING: "embed_images",
    Capability.RERANKING: "rerank",
    Capability.OCR: "ocr",
    Capability.VISION: "analyze",
    Capability.WEB_SEARCH: "search",
    Capability.WEB_EXTRACTION: "extract",
}
